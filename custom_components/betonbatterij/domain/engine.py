"""DigitalTwinEngine facade for Home Assistant (`ha_bridge`) and Layer 2.

Implements Component 5 §4 (`05-software-architectuur-opslag-en-interfaces.md`).
100% pure Python stdlib — zero `homeassistant.*`, `numpy`, or `scipy` imports.
"""
from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import math
import pathlib
from typing import Any, Literal

from custom_components.betonbatterij import const
from custom_components.betonbatterij import house_config
from custom_components.betonbatterij.domain import calibration
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import cop_model
from custom_components.betonbatterij.domain import evaluation
from custom_components.betonbatterij.domain import hydraulics
from custom_components.betonbatterij.domain import solar_weather
from custom_components.betonbatterij.domain import zoh_kalman

SNAPSHOT_STACK_MAX_DEPTH: int = 8
MAX_CALIBRATION_HISTORY: int = 32


def _opt_row_temp(
    raw_row: Mapping[str, float | int],
    key: str,
) -> float | None:
    """Reads an optional finite temperature (`>= -100 °C`) from `raw_row`."""
    if key not in raw_row:
        return None
    val = raw_row[key]
    if val is None:
        return None
    fval = float(val)
    if not math.isfinite(fval) or fval < -100.0:
        return None
    return fval


def _row_float(
    raw_row: Mapping[str, float | int],
    key: str,
    default: float = 0.0,
    alt_key: str | None = None,
) -> float:
    """Reads a finite float from `raw_row[key]` (or `alt_key`)."""
    val = raw_row.get(key)
    if val is None and alt_key is not None:
        val = raw_row.get(alt_key)
    if val is None:
        return default
    fval = float(val)
    if not math.isfinite(fval):
        return default
    return fval


def _zone_params_to_dict(params: contracts.ZoneParameters) -> dict[str, float]:
    """Serializes `ZoneParameters` to a float dictionary."""
    return {k: float(v) for k, v in dataclasses.asdict(params).items()}


def _zone_params_from_dict(
    data: Mapping[str, Any],
    fallback: contracts.ZoneParameters,
) -> contracts.ZoneParameters:
    """Deserializes `ZoneParameters` with `fallback` defaults."""
    base = dataclasses.asdict(fallback)
    for k in base:
        if k in data and data[k] is not None:
            base[k] = float(data[k])
    return contracts.ZoneParameters(**base)


def _cop_params_to_dict(params: contracts.CopParameters) -> dict[str, float]:
    """Serializes `CopParameters` to a float dictionary."""
    return {k: float(v) for k, v in dataclasses.asdict(params).items()}


def _cop_params_from_dict(
    data: Mapping[str, Any] | None,
) -> contracts.CopParameters:
    """Deserializes `CopParameters` with `DAY1_COP_PARAMS` defaults."""
    if not data:
        return house_config.DAY1_COP_PARAMS
    base = dataclasses.asdict(house_config.DAY1_COP_PARAMS)
    for k in base:
        if k in data and data[k] is not None:
            base[k] = float(data[k])
    return contracts.CopParameters(**base)


def _discretize_2x2(
    params: contracts.ZoneParameters,
    dt_s: float,
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[float, float],
    tuple[float, float],
]:
    """Extracts exact 2x2 ZOH matrices `(phi_2x2, gamma_fh, gamma_ac_el_ref)`."""
    a_d, b_d, _ = zoh_kalman.discretize_zoh_3x3(
        params=params,
        dt_s=dt_s,
        v_wind_ms=3.0,
    )
    phi_2x2 = (
        (float(a_d[0][0]), float(a_d[0][1])),
        (float(a_d[1][0]), float(a_d[1][1])),
    )
    gamma_fh = (float(b_d[0][4]), float(b_d[1][4]))
    gamma_ac = (
        params.eta_ac * float(b_d[0][6]),
        params.eta_ac * float(b_d[1][6]),
    )
    return phi_2x2, gamma_fh, gamma_ac


def _fit_cop_with_mode(
    columns: Mapping[str, Sequence[float | int]],
    current_cop: contracts.CopParameters,
    memory_cop: contracts.ParameterPosteriorMemory | None,
    mode: Literal["manual", "auto_weekly"],
    lat_deg: float = house_config.LATITUDE_DEG,
    lon_deg: float = house_config.LONGITUDE_DEG,
) -> tuple[
    contracts.CopParameters,
    contracts.ParameterPosteriorMemory,
    dict[str, Any],
]:
    """Fits `CopParameters` and applies `0.80 / 0.20` damping in `auto_weekly`."""
    fitted_cop, updated_mem, diag_raw = cop_model.fit_cop_parameters(
        columns=columns,
        current_params=current_cop,
        memory=memory_cop,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )
    diag = dict(diag_raw)

    if mode == "auto_weekly" and fitted_cop != current_cop:
        bnds = house_config.COP_PARAM_BOUNDS
        final_cop = contracts.CopParameters(
            eta_carnot_cv=min(
                bnds["eta_carnot_cv"].upper,
                max(
                    bnds["eta_carnot_cv"].lower,
                    0.80 * current_cop.eta_carnot_cv
                    + 0.20 * fitted_cop.eta_carnot_cv,
                ),
            ),
            beta_defrost=min(
                bnds["beta_defrost"].upper,
                max(
                    bnds["beta_defrost"].lower,
                    0.80 * current_cop.beta_defrost
                    + 0.20 * fitted_cop.beta_defrost,
                ),
            ),
            eta_carnot_dhw=min(
                bnds["eta_carnot_dhw"].upper,
                max(
                    bnds["eta_carnot_dhw"].lower,
                    0.80 * current_cop.eta_carnot_dhw
                    + 0.20 * fitted_cop.eta_carnot_dhw,
                ),
            ),
            delta_p_wp_bias_w=min(
                bnds["delta_p_wp_bias_w"].upper,
                max(
                    bnds["delta_p_wp_bias_w"].lower,
                    0.80 * current_cop.delta_p_wp_bias_w
                    + 0.20 * fitted_cop.delta_p_wp_bias_w,
                ),
            ),
        )
    else:
        final_cop = fitted_cop

    diag["sample_20hz_corr_el_w"] = round(
        cop_model.corrected_wp_electrical_power_w(
            final_cop, 600.0, 20.0, False
        ),
        1,
    )
    diag["cop_cv_7c_27c"] = round(
        cop_model.predict_cop_cv(final_cop, 7.0, 27.0, 80.0), 3
    )
    diag["cop_dhw_7c_42_50c"] = round(
        cop_model.predict_cop_dhw(final_cop, 7.0, 42.0, 50.0, 80.0), 3
    )
    return final_cop, updated_mem, diag


class DigitalTwinEngine:
    """Complete pure-Python Layer-1 Digital Twin facade for `ha_bridge` & Layer 2.

    Attributes:
        estimator_wk: 3-state augmented Kalman filter for `wk`.
        estimator_kt: 3-state augmented Kalman filter for `kt`.
        hydraulics: Secondary manifold and 200 L buffer `HydraulicModel`.
        cop_params: Active heat pump `CopParameters`.
        hindcast_wk: Live 72-step 6h open-loop FIFO tracker for `wk`.
        hindcast_kt: Live 72-step 6h open-loop FIFO tracker for `kt`.
        episode_bank: Choice D3 365-day `SeasonalEpisodeBank`.
        memory_wk: Choice D2 seasonal posterior memory for `wk`.
        memory_kt: Choice D2 seasonal posterior memory for `kt`.
        memory_cop: Choice D2 seasonal posterior memory for COP parameters.
        eval_report_wk: Latest 14-day multi-regime evaluation report for `wk`.
        eval_report_kt: Latest 14-day multi-regime evaluation report for `kt`.
        cop_diagnostics: Latest COP fit and RMSE diagnostics dictionary.
        calibration_history: Bounded audit log of calibration/rollback events.
        snapshot_stack: 8-deep parameter snapshot deque for rollback safety.
    """

    def __init__(
        self,
        params_wk: contracts.ZoneParameters | None = None,
        params_kt: contracts.ZoneParameters | None = None,
        flow_rates_l_min: Mapping[str, float] | None = None,
        k_hx_by_group_w_k: Mapping[str, float] | None = None,
        cop_params: contracts.CopParameters | None = None,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
    ) -> None:
        """Initializes `DigitalTwinEngine` with Day-1 priors or custom overrides.

        Args:
            params_wk: Optional `ZoneParameters` override for `wk`.
            params_kt: Optional `ZoneParameters` override for `kt`.
            flow_rates_l_min: Optional manifold rotameter flow rates [L/min].
            k_hx_by_group_w_k: Optional `K_hx` [W/K] per hydraulic group.
            cop_params: Optional heat pump `CopParameters` override.
            latitude_deg: Optional site latitude [deg N] (non-zero overrides
                `house_config.LATITUDE_DEG`).
            longitude_deg: Optional site longitude [deg E] (non-zero overrides
                `house_config.LONGITUDE_DEG`).
        """
        self.latitude_deg: float = (
            float(latitude_deg)
            if latitude_deg is not None
            and math.isfinite(latitude_deg)
            and float(latitude_deg) != 0.0
            else house_config.LATITUDE_DEG
        )
        self.longitude_deg: float = (
            float(longitude_deg)
            if longitude_deg is not None
            and math.isfinite(longitude_deg)
            and float(longitude_deg) != 0.0
            else house_config.LONGITUDE_DEG
        )
        self._flow_rates_l_min: dict[str, float] = dict(
            const.DEFAULT_FLOW_RATES_L_MIN
        )
        if flow_rates_l_min:
            self._flow_rates_l_min.update(
                {k: float(v) for k, v in flow_rates_l_min.items()}
            )

        self._k_hx_by_group_w_k: dict[str, float] = dict(
            house_config.DAY1_K_HX_BY_GROUP_W_K
        )
        if k_hx_by_group_w_k:
            for g, v in k_hx_by_group_w_k.items():
                self._k_hx_by_group_w_k["kantoor" if g == "kt" else g] = float(
                    v
                )

        p_wk = params_wk or dataclasses.replace(
            house_config.DAY1_PRIORS_WK,
            k_hx_w_k=self._k_hx_by_group_w_k["wk"],
        )
        p_kt = params_kt or dataclasses.replace(
            house_config.DAY1_PRIORS_KT,
            k_hx_w_k=self._k_hx_by_group_w_k["kantoor"],
        )
        self._k_hx_by_group_w_k["wk"] = p_wk.k_hx_w_k
        self._k_hx_by_group_w_k["kantoor"] = p_kt.k_hx_w_k

        self._params_wk: contracts.ZoneParameters = p_wk
        self._params_kt: contracts.ZoneParameters = p_kt

        self.estimator_wk: zoh_kalman.ZoneKalmanEstimator = (
            zoh_kalman.ZoneKalmanEstimator("wk", p_wk)
        )
        self.estimator_kt: zoh_kalman.ZoneKalmanEstimator = (
            zoh_kalman.ZoneKalmanEstimator("kt", p_kt)
        )
        self.hydraulics: hydraulics.HydraulicModel = hydraulics.HydraulicModel(
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group_w_k=self._k_hx_by_group_w_k,
        )
        self.cop_params: contracts.CopParameters = (
            cop_params or house_config.DAY1_COP_PARAMS
        )
        self.hindcast_wk: evaluation.LiveHindcastTracker = (
            evaluation.LiveHindcastTracker(
                horizon_steps=evaluation.HORIZON_6H_STEPS
            )
        )
        self.hindcast_kt: evaluation.LiveHindcastTracker = (
            evaluation.LiveHindcastTracker(
                horizon_steps=evaluation.HORIZON_6H_STEPS
            )
        )
        self.episode_bank: calibration.SeasonalEpisodeBank = (
            calibration.SeasonalEpisodeBank()
        )

        self.memory_wk: contracts.ParameterPosteriorMemory | None = None
        self.memory_kt: contracts.ParameterPosteriorMemory | None = None
        self.memory_cop: contracts.ParameterPosteriorMemory | None = None

        self.eval_report_wk: contracts.ZoneEvaluationReport | None = None
        self.eval_report_kt: contracts.ZoneEvaluationReport | None = None
        self.cop_diagnostics: dict[str, Any] = {}
        self.calibration_history: list[dict[str, Any]] = []
        self.snapshot_stack: collections.deque[dict[str, Any]] = (
            collections.deque(maxlen=SNAPSHOT_STACK_MAX_DEPTH)
        )

        self._hm_ema_c: float | None = None
        self._interfloor_cache: dict[str, tuple[float, int]] = {}
        self._desk_state_kt: dict[str, Any] = {}
        self._last_ts_utc: int = 0
        self._last_q_fh_wk_w: float = 0.0
        self._last_q_fh_kt_w: float = 0.0
        self._last_t_air_meas_wk_c: float | None = None
        self._last_t_air_meas_kt_c: float | None = None
        self._last_posterior_std_wk: dict[str, float] = {}
        self._last_posterior_std_kt: dict[str, float] = {}
        self._last_corr_wk: dict[str, dict[str, float]] = {}
        self._last_corr_kt: dict[str, dict[str, float]] = {}
        self._last_collinear_wk: list[str] = []
        self._last_collinear_kt: list[str] = []

        self._snapshot_wk: contracts.ZoneSnapshot | None = None
        self._snapshot_kt: contracts.ZoneSnapshot | None = None

    def _build_snapshot(
        self,
        zone: contracts.ZoneId,
    ) -> contracts.ZoneSnapshot:
        """Constructs an immutable `ZoneSnapshot` for `zone`."""
        est = self.estimator_wk if zone == "wk" else self.estimator_kt
        params = self._params_wk if zone == "wk" else self._params_kt
        hind = self.hindcast_wk if zone == "wk" else self.hindcast_kt
        ev = self.eval_report_wk if zone == "wk" else self.eval_report_kt
        q_fh = self._last_q_fh_wk_w if zone == "wk" else self._last_q_fh_kt_w
        t_meas = (
            self._last_t_air_meas_wk_c
            if zone == "wk"
            else self._last_t_air_meas_kt_c
        )

        st = est.state
        t_surf = evaluation.estimate_zone_surface_temp(
            zone, st.t_air_c, st.t_floor_c, params
        )
        stored_wh = params.c_floor_wh_k * (st.t_floor_c - st.t_air_c)
        rmse_6h = ev.overall_6h.rmse_end_c if ev is not None else None

        return contracts.ZoneSnapshot(
            zone=zone,
            ts_utc=self._last_ts_utc,
            state=st,
            t_surface_est_c=t_surf,
            t_air_meas_c=t_meas,
            t_floor_std_c=float(est.t_floor_std_c),
            q_fh_now_w=q_fh,
            t_hindcast_6h_c=hind.last_published_c,
            sensor_stale=bool(est.sensor_stale),
            open_window_active=bool(est.open_window_active),
            stored_slab_energy_wh=stored_wh,
            params=params,
            rmse_6h_c=rmse_6h,
        )

    def _refresh_cached_snapshots(self) -> None:
        """Refreshes cached O(1) snapshots for both zones."""
        self._snapshot_wk = self._build_snapshot("wk")
        self._snapshot_kt = self._build_snapshot("kt")

    # ========================================================================
    # Real-Time 5-Minute Pipeline & 48-Hour Parameter Replay
    # ========================================================================
    def process_5m_window(
        self,
        raw_row: Mapping[str, float | int],
        temp_changed_wk: bool = True,
        hum_changed_wk: bool = False,
        stale_wk: bool = False,
        temp_changed_kt: bool = True,
        hum_changed_kt: bool = False,
        stale_kt: bool = False,
        dt_s: float = 300.0,
    ) -> dict[str, float]:
        """Processes one 5-minute telemetry window and updates online estimators.

        Args:
            raw_row: Mapping of accumulated 5-minute sensor values.
            temp_changed_wk: True if `wk` Zigbee temperature reported a new value.
            hum_changed_wk: True if `wk` Zigbee humidity reported a heartbeat.
            stale_wk: True if `wk` room sensor exceeded staleness threshold.
            temp_changed_kt: True if `kt` Zigbee temperature reported a new value.
            hum_changed_kt: True if `kt` Zigbee humidity reported a heartbeat.
            stale_kt: True if `kt` room sensor exceeded staleness threshold.
            dt_s: Elapsed step duration in seconds (default `300.0`).

        Returns:
            Dictionary of 8 trajectory values (`t_floor_*`, `d_est_*`, `q_fh_*`,
            `t_hind6h_*`) to persist in the RingBuffer row.
        """
        ts_utc = int(
            raw_row.get("ts_utc", self._last_ts_utc + int(round(dt_s)))
        )
        has_flags = "flags" in raw_row
        default_fl = (
            const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
        )
        fl = int(raw_row["flags"]) if has_flags else default_fl

        eff_stale_wk = stale_wk or (
            has_flags and not (fl & const.FLAG_VALID_WK)
        )
        eff_stale_kt = stale_kt or (
            has_flags and not (fl & const.FLAG_VALID_KT)
        )
        aanwezig = (
            bool(fl & const.FLAG_AANWEZIG)
            if has_flags
            else bool(raw_row.get("aanwezig", True))
        )

        t_dak = _opt_row_temp(raw_row, "t_out_dak_c")
        t_wea = _opt_row_temp(raw_row, "t_out_weather_c")
        t_hm = _opt_row_temp(raw_row, "t_out_hm_c")
        if t_dak is None and t_wea is None and t_hm is None:
            t_dak = _opt_row_temp(raw_row, "t_out_c")
        if t_hm is not None:
            self._hm_ema_c = (
                t_hm
                if self._hm_ema_c is None
                else (t_hm / 6.0) + (5.0 / 6.0) * self._hm_ema_c
            )
        t_out_c = solar_weather.select_outdoor_temp(
            ts_utc=ts_utc,
            t_dak_c=t_dak,
            t_weather_c=t_wea,
            t_hm_ema_c=self._hm_ema_c,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        v_wind = max(0.0, _row_float(raw_row, "v_wind_ms", 3.0))

        pv_fb_hint = bool(fl & const.FLAG_PV_FALLBACK)
        if (
            "i_sol_wk_wm2" in raw_row
            and "i_sol_kt_wm2" in raw_row
            and "p_pv_w" not in raw_row
        ):
            i_sol_wk = max(0.0, _row_float(raw_row, "i_sol_wk_wm2", 0.0))
            i_sol_kt = max(0.0, _row_float(raw_row, "i_sol_kt_wm2", 0.0))
        else:
            if "p_pv_w" in raw_row:
                raw_pv = raw_row["p_pv_w"]
                p_pv = math.nan if raw_pv is None else float(raw_pv)
            else:
                p_pv = 0.0
            p_pv_fc = _row_float(raw_row, "p_pv_forecast_w", 0.0)
            epex = _row_float(raw_row, "epex_eur_kwh", 0.20)
            scr_s = _row_float(raw_row, "screen_small_pos", 100.0)
            scr_l = _row_float(raw_row, "screen_large_pos", 100.0)
            i_wk_raw, i_kt_raw, _, _ = solar_weather.project_solar_irradiance(
                ts_utc=ts_utc,
                p_pv_w=p_pv,
                p_pv_forecast_w=p_pv_fc,
                epex_eur_kwh=epex,
                screen_small_pos=scr_s,
                screen_large_pos=scr_l,
                pv_stale=pv_fb_hint,
                lat_deg=self.latitude_deg,
                lon_deg=self.longitude_deg,
            )
            i_sol_wk = max(0.0, i_wk_raw)
            i_sol_kt = max(0.0, i_kt_raw)

        t_wk = _row_float(
            raw_row,
            "t_air_wk_c",
            (
                self.estimator_wk.state.t_air_c
                if self.estimator_wk.initialized
                else 19.5
            ),
        )
        t_kt = _row_float(
            raw_row,
            "t_air_kantoor_c",
            (
                self.estimator_kt.state.t_air_c
                if self.estimator_kt.initialized
                else 19.0
            ),
            "t_air_kt_c",
        )
        room_temps: dict[str, float | None] = {
            "wk": t_wk,
            "gang": t_wk - 0.8,
            "slaapkamer": _row_float(raw_row, "t_air_slaapkamer_c", 18.0),
            "badkamer": _row_float(raw_row, "t_air_badkamer_c", 18.5),
            "werkkamer_1e": _row_float(raw_row, "t_air_werkkamer_1e_c", 18.0),
            "kamer_1e_midden": _row_float(
                raw_row, "t_air_kamer_1e_midden_c", 18.0
            ),
            "kamer_1e_zuid": _row_float(
                raw_row, "t_air_kamer_1e_zuid_c", 18.0
            ),
            "kantoor": t_kt,
            "badkamer_boven": _row_float(
                raw_row, "t_air_badkamer_boven_c", 18.0
            ),
            "logeerkamer": _row_float(raw_row, "t_air_logeerkamer_c", 17.5),
        }

        u_bg = min(
            1.0, max(0.0, _row_float(raw_row, "u_eff_bg", 0.0, "u_eff_wk"))
        )
        u_kt_v = min(
            1.0,
            max(0.0, _row_float(raw_row, "u_eff_kantoor", 0.0, "u_eff_kt")),
        )
        u_pump = min(1.0, max(0.0, _row_float(raw_row, "u_pump_sec", 0.0)))
        u_valves: dict[str, float] = {
            "wk": u_bg,
            "gang": u_bg,
            "slaapkamer": min(
                1.0, max(0.0, _row_float(raw_row, "u_eff_slaapkamer", 0.0))
            ),
            "badkamer": min(
                1.0, max(0.0, _row_float(raw_row, "u_eff_badkamer", 0.0))
            ),
            "werkkamer_1e": min(
                1.0, max(0.0, _row_float(raw_row, "u_eff_werkkamer_1e", 0.0))
            ),
            "kamer_1e_midden": min(
                1.0,
                max(0.0, _row_float(raw_row, "u_eff_kamer_1e_midden", 0.0)),
            ),
            "kamer_1e_zuid": min(
                1.0, max(0.0, _row_float(raw_row, "u_eff_kamer_1e_zuid", 0.0))
            ),
            "kantoor": u_kt_v,
            "badkamer_boven": min(
                1.0, max(0.0, _row_float(raw_row, "u_eff_badkamer_boven", 0.0))
            ),
            "logeerkamer": min(
                1.0, max(0.0, _row_float(raw_row, "u_eff_logeerkamer", 0.0))
            ),
        }

        threeway_room = (
            bool(raw_row["threeway_room"])
            if "threeway_room" in raw_row
            else ((fl & const.FLAG_DHW_TANK) == 0)
        )
        wp_defrost = bool(
            raw_row.get("wp_defrost", bool(fl & const.FLAG_DEFROST_WP))
        )
        v_prim = max(0.0, _row_float(raw_row, "v_prim_l_min", 0.0))
        p_prim = _row_float(raw_row, "p_prim_th_w", 0.0)
        f_hz = max(0.0, _row_float(raw_row, "f_comp_hz", 0.0))

        t_top_e = _row_float(raw_row, "t_buf_top_end_c", 22.0, "t_buf_top_c")
        t_top_s = _row_float(raw_row, "t_buf_top_start_c", t_top_e)
        t_bot_e = _row_float(
            raw_row, "t_buf_bot_end_c", t_top_e - 1.0, "t_buf_bot_c"
        )
        t_bot_s = _row_float(raw_row, "t_buf_bot_start_c", t_bot_e)
        t_bg_sup = _opt_row_temp(raw_row, "t_bg_sup_c")
        t_bg_ret = _opt_row_temp(raw_row, "t_bg_ret_c")

        p_ac = max(0.0, _row_float(raw_row, "p_el_ac_kt_w", 0.0))
        ac_mode = _row_float(raw_row, "ac_mode_code", 0.0)
        p_desk = _opt_row_temp(raw_row, "p_desk_kt_w")
        pir_kt = _row_float(raw_row, "pir_occ_kt", 0.0)
        p_grow = max(0.0, _row_float(raw_row, "p_growlights_kt_w", 0.0))
        ac_def = bool(fl & const.FLAG_DEFROST_AC)

        q_int_kt, q_ac_kt = solar_weather.compute_internal_and_ac_loads_kt(
            p_desk_w=p_desk,
            pir_occ=pir_kt,
            p_growlights_w=p_grow,
            p_el_ac_w=p_ac,
            ac_mode_code=ac_mode,
            ac_defrost=ac_def,
            t_out_c=t_out_c,
            eta_ac=self._params_kt.eta_ac,
            desk_state=self._desk_state_kt,
            dt_s=min(dt_s, 10800.0),
        )
        t_up_wk, t_dn_kt = solar_weather.compute_interfloor_boundaries(
            room_temps_c=room_temps,
            last_valid_cache=self._interfloor_cache,
            ts_utc=ts_utc,
        )
        t_dn_wk = house_config.T_GROUND_MEAN_C

        # Initialize estimators on very first tick or >3h outage
        if not self.estimator_wk.initialized or dt_s > 10800.0:
            ex_init_wk = contracts.ExogenousStepInput(
                t_out_c=t_out_c,
                v_wind_ms=v_wind,
                i_sol_wm2=i_sol_wk,
                t_down_c=t_dn_wk,
                q_int_known_w=house_config.BASELOAD_WK_W,
                t_up_c=t_up_wk,
            )
            self.estimator_wk.cold_start(
                t_air_c=t_wk,
                exog=ex_init_wk,
                q_fh_w=0.0,
                q_ac_w=0.0,
                u_pump=u_pump,
                t_buf_top_c=t_top_e,
            )
            if dt_s > 10800.0:
                self.hindcast_wk.clear()

        if not self.estimator_kt.initialized or dt_s > 10800.0:
            ex_init_kt = contracts.ExogenousStepInput(
                t_out_c=t_out_c,
                v_wind_ms=v_wind,
                i_sol_wm2=i_sol_kt,
                t_down_c=t_dn_kt,
                q_int_known_w=q_int_kt,
                t_up_c=0.0,
            )
            self.estimator_kt.cold_start(
                t_air_c=t_kt,
                exog=ex_init_kt,
                q_fh_w=0.0,
                q_ac_w=q_ac_kt,
                u_pump=u_pump,
                t_buf_top_c=t_top_e,
            )
            if dt_s > 10800.0:
                self.hindcast_kt.clear()

        hyd_res = self.hydraulics.step(
            dt_s=min(dt_s, 10800.0),
            u_pump_sec=u_pump,
            u_valves=u_valves,
            threeway_room=threeway_room,
            wp_defrost=wp_defrost,
            v_prim_l_min=v_prim,
            p_prim_th_w=p_prim,
            f_comp_hz=f_hz,
            t_buf_top_start_c=t_top_s,
            t_buf_top_end_c=t_top_e,
            t_buf_bot_start_c=t_bot_s,
            t_buf_bot_end_c=t_bot_e,
            t_floor_wk_c=self.estimator_wk.state.t_floor_c,
            t_floor_kt_c=self.estimator_kt.state.t_floor_c,
            room_temps_c=room_temps,
            t_bg_sup_c=t_bg_sup,
            t_bg_ret_c=t_bg_ret,
            bg_ret_offset_c=0.0,
        )
        q_fh_wk = hyd_res.q_fh_wk_w
        q_fh_kt = hyd_res.q_fh_kt_w
        if u_pump <= 1e-4 and v_prim <= 1e-4:
            if _row_float(raw_row, "q_fh_wk_w", 0.0) > 0.0:
                q_fh_wk = _row_float(raw_row, "q_fh_wk_w", 0.0)
            if _row_float(raw_row, "q_fh_kt_w", 0.0) > 0.0:
                q_fh_kt = _row_float(raw_row, "q_fh_kt_w", 0.0)

        q_int_wk = house_config.BASELOAD_WK_W + max(
            0.0, hyd_res.q_leak_1e_to_wk_w
        )
        exog_wk = contracts.ExogenousStepInput(
            t_out_c=t_out_c,
            v_wind_ms=v_wind,
            i_sol_wm2=i_sol_wk,
            t_down_c=t_dn_wk,
            q_int_known_w=q_int_wk,
            t_up_c=t_up_wk,
        )
        exog_kt = contracts.ExogenousStepInput(
            t_out_c=t_out_c,
            v_wind_ms=v_wind,
            i_sol_wm2=i_sol_kt,
            t_down_c=t_dn_kt,
            q_int_known_w=q_int_kt,
            t_up_c=0.0,
        )

        t_surf_meas_wk = _opt_row_temp(raw_row, "t_floor_surf_meas_wk_c")
        meas_wk = None if eff_stale_wk else t_wk
        meas_kt = None if eff_stale_kt else t_kt
        if (
            not temp_changed_wk
            and meas_wk is not None
            and self.estimator_wk._last_meas_temp_c is not None
            and abs(meas_wk - self.estimator_wk._last_meas_temp_c) >= 1e-4
        ):
            temp_changed_wk = True
        if (
            not temp_changed_kt
            and meas_kt is not None
            and self.estimator_kt._last_meas_temp_c is not None
            and abs(meas_kt - self.estimator_kt._last_meas_temp_c) >= 1e-4
        ):
            temp_changed_kt = True

        if 450.0 < dt_s <= 10800.0:
            # Bridge short restart gap before regular tick semantics
            gap_extra = dt_s - float(const.DT_STEP_S)
            self.hindcast_wk.bridge_gap(
                gap_extra, self._params_wk, exog_wk, q_fh_wk, 0.0
            )
            self.hindcast_kt.bridge_gap(
                gap_extra, self._params_kt, exog_kt, q_fh_kt, q_ac_kt
            )
            step_hind_dt = float(const.DT_STEP_S)
        else:
            step_hind_dt = dt_s if dt_s <= 10800.0 else float(const.DT_STEP_S)

        state_wk = self.estimator_wk.step(
            ts_utc=ts_utc,
            dt_s=min(dt_s, 10800.0),
            exog=exog_wk,
            q_fh_w=q_fh_wk,
            q_ac_w=0.0,
            u_pump=u_pump,
            t_air_meas_c=meas_wk,
            temp_changed=temp_changed_wk,
            hum_changed=hum_changed_wk,
            sensor_stale=eff_stale_wk,
            aanwezig=aanwezig,
            t_surf_meas_c=t_surf_meas_wk,
        )
        state_kt = self.estimator_kt.step(
            ts_utc=ts_utc,
            dt_s=min(dt_s, 10800.0),
            exog=exog_kt,
            q_fh_w=q_fh_kt,
            q_ac_w=q_ac_kt,
            u_pump=u_pump,
            t_air_meas_c=meas_kt,
            temp_changed=temp_changed_kt,
            hum_changed=hum_changed_kt,
            sensor_stale=eff_stale_kt,
            aanwezig=aanwezig,
            t_surf_meas_c=None,
        )

        t_hind_wk, _ = self.hindcast_wk.step(
            state_wk, self._params_wk, exog_wk, q_fh_wk, 0.0, step_hind_dt
        )
        t_hind_kt, _ = self.hindcast_kt.step(
            state_kt, self._params_kt, exog_kt, q_fh_kt, q_ac_kt, step_hind_dt
        )

        self._last_ts_utc = ts_utc
        self._last_q_fh_wk_w = q_fh_wk
        self._last_q_fh_kt_w = q_fh_kt
        self._last_t_air_meas_wk_c = meas_wk
        self._last_t_air_meas_kt_c = meas_kt
        self._refresh_cached_snapshots()

        return {
            "t_floor_wk_c": float(state_wk.t_floor_c),
            "d_est_wk_w": float(state_wk.d_est_w),
            "q_fh_wk_w": float(q_fh_wk),
            "t_hind6h_wk_c": (
                math.nan if t_hind_wk is None else float(t_hind_wk)
            ),
            "t_floor_kt_c": float(state_kt.t_floor_c),
            "d_est_kt_w": float(state_kt.d_est_w),
            "q_fh_kt_w": float(q_fh_kt),
            "t_hind6h_kt_c": (
                math.nan if t_hind_kt is None else float(t_hind_kt)
            ),
        }

    def replay_48h_from_columns(
        self,
        columns: Mapping[str, Sequence[float | int]],
    ) -> None:
        """Runs 48h (`576`-step) Kalman + Hindcast replay after parameter updates.

        Args:
            columns: RingBuffer telemetry column dictionary.
        """
        ts_seq = columns.get("ts_utc", ())
        n_total = len(ts_seq)
        if n_total == 0:
            return

        start_idx = max(0, n_total - evaluation.SPINUP_48H_STEPS)
        sub_cols = evaluation.slice_columns(columns, start_idx, n_total)

        self._k_hx_by_group_w_k["wk"] = self._params_wk.k_hx_w_k
        self._k_hx_by_group_w_k["kantoor"] = self._params_kt.k_hx_w_k
        self.hydraulics = hydraulics.HydraulicModel(
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group_w_k=self._k_hx_by_group_w_k,
        )

        hist = evaluation.preprocess_history_columns(
            sub_cols,
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group=self._k_hx_by_group_w_k,
            eta_ac_ref=self._params_kt.eta_ac,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        exog_wk = evaluation.build_zone_exog_list(hist, "wk")
        exog_kt = evaluation.build_zone_exog_list(hist, "kt")

        states_wk, q_fh_wk_arr, _, est_wk = (
            evaluation.reconstruct_kalman_states(
                hist, "wk", self._params_wk, exog_wk
            )
        )
        states_kt, q_fh_kt_arr, q_ac_kt_arr, est_kt = (
            evaluation.reconstruct_kalman_states(
                hist, "kt", self._params_kt, exog_kt
            )
        )

        self.estimator_wk = est_wk
        self.estimator_kt = est_kt

        tail_start = max(
            0, hist.n_rows - (evaluation.HORIZON_6H_STEPS + 1)
        )
        steps_wk = [
            {
                "posterior_state": states_wk[k],
                "exog": exog_wk[k],
                "q_fh_w": q_fh_wk_arr[k],
                "q_ac_w": 0.0,
                "dt_s": hist.dt_s[k],
            }
            for k in range(tail_start, hist.n_rows)
        ]
        steps_kt = [
            {
                "posterior_state": states_kt[k],
                "exog": exog_kt[k],
                "q_fh_w": q_fh_kt_arr[k],
                "q_ac_w": q_ac_kt_arr[k],
                "dt_s": hist.dt_s[k],
            }
            for k in range(tail_start, hist.n_rows)
        ]
        self.hindcast_wk.reseed_from_history(steps_wk, self._params_wk)
        self.hindcast_kt.reseed_from_history(steps_kt, self._params_kt)

        self._last_ts_utc = int(hist.ts_utc[-1])
        self._last_q_fh_wk_w = q_fh_wk_arr[-1] if q_fh_wk_arr else 0.0
        self._last_q_fh_kt_w = q_fh_kt_arr[-1] if q_fh_kt_arr else 0.0
        self._last_t_air_meas_wk_c = (
            hist.t_air_wk_c[-1]
            if (hist.flags[-1] & const.FLAG_VALID_WK)
            else None
        )
        self._last_t_air_meas_kt_c = (
            hist.t_air_kt_c[-1]
            if (hist.flags[-1] & const.FLAG_VALID_KT)
            else None
        )
        self._refresh_cached_snapshots()

    # ========================================================================
    # Calibration, Snapshot Stack (8-deep), Rollback, Reset & Diagnostics
    # ========================================================================
    def _capture_snapshot(
        self,
        reason: str,
        ts_utc: int | None = None,
    ) -> dict[str, Any]:
        """Captures an immutable parameter + seasonal memory snapshot dict."""
        return {
            "ts_utc": int(ts_utc if ts_utc is not None else self._last_ts_utc),
            "reason": reason,
            "params_wk": _zone_params_to_dict(self._params_wk),
            "params_kt": _zone_params_to_dict(self._params_kt),
            "k_hx_by_group_w_k": dict(self._k_hx_by_group_w_k),
            "cop_params": _cop_params_to_dict(self.cop_params),
            "memory_wk": calibration.posterior_memory_to_dict(self.memory_wk),
            "memory_kt": calibration.posterior_memory_to_dict(self.memory_kt),
            "memory_cop": calibration.posterior_memory_to_dict(
                self.memory_cop
            ),
        }

    def _apply_snapshot_dict(self, snap: Mapping[str, Any]) -> None:
        """Restores active parameters and seasonal memory from `snap`."""
        self._params_wk = _zone_params_from_dict(
            snap.get("params_wk", {}), house_config.DAY1_PRIORS_WK
        )
        self._params_kt = _zone_params_from_dict(
            snap.get("params_kt", {}), house_config.DAY1_PRIORS_KT
        )

        khx_raw = snap.get("k_hx_by_group_w_k")
        self._k_hx_by_group_w_k = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
        if isinstance(khx_raw, Mapping):
            self._k_hx_by_group_w_k.update(
                {str(k): float(v) for k, v in khx_raw.items()}
            )
        self._k_hx_by_group_w_k["wk"] = self._params_wk.k_hx_w_k
        self._k_hx_by_group_w_k["kantoor"] = self._params_kt.k_hx_w_k

        self.cop_params = _cop_params_from_dict(snap.get("cop_params"))
        self.memory_wk = calibration.posterior_memory_from_dict(
            snap.get("memory_wk")
        )
        self.memory_kt = calibration.posterior_memory_from_dict(
            snap.get("memory_kt")
        )
        self.memory_cop = calibration.posterior_memory_from_dict(
            snap.get("memory_cop")
        )

        self.estimator_wk.params = self._params_wk
        self.estimator_kt.params = self._params_kt
        self.hydraulics.update_k_hx(self._k_hx_by_group_w_k)

    def run_calibration(
        self,
        columns: Mapping[str, Sequence[float | int]],
        mode: Literal["manual", "auto_weekly"] = "manual",
    ) -> dict[str, Any]:
        """Runs batch calibration (`episode_bank` + `k_hx` + zones + COP + 48h).

        Args:
            columns: Full RingBuffer telemetry column dictionary.
            mode: `"manual"` or `"auto_weekly"`.

        Returns:
            Summary dictionary with `accepted_any`, `accepted_wk`,
            `accepted_kt`, `outcome_wk`, `outcome_kt`, `cop_params`, and
            `snapshot_depth`.
        """
        ts_seq = columns.get("ts_utc", ())
        n_total = len(ts_seq)
        ts_now = int(ts_seq[-1]) if n_total > 0 else self._last_ts_utc

        if mode == "auto_weekly":
            gate_ok, gate_reason = (
                calibration.check_seasonal_calibration_gate(
                    columns,
                    lat_deg=self.latitude_deg,
                    lon_deg=self.longitude_deg,
                )
            )
            if not gate_ok:
                return {
                    "accepted_any": False,
                    "accepted_wk": False,
                    "accepted_kt": False,
                    "blocked_by_seasonal_gate": True,
                    "reason": gate_reason,
                    "snapshot_depth": len(self.snapshot_stack),
                }

        pre_snap = self._capture_snapshot(
            reason=f"{mode}_calibration", ts_utc=ts_now
        )

        # Curate EpisodeBank strictly from training split (no holdout leakage)
        if n_total >= 4 * const.STEPS_PER_DAY:
            train_end = (
                n_total - evaluation.EVAL_14D_STEPS
                if n_total >= 28 * const.STEPS_PER_DAY
                else max(2 * const.STEPS_PER_DAY, int(0.70 * n_total))
            )
            train_slice = evaluation.slice_columns(columns, 0, train_end)
            self.episode_bank.update_from_columns(
                train_slice,
                flow_rates_l_min=self._flow_rates_l_min,
                k_hx_by_group=self._k_hx_by_group_w_k,
                lat_deg=self.latitude_deg,
                lon_deg=self.longitude_deg,
            )
        else:
            train_slice = dict(columns)

        # 1. Calibrate steady-state K_hx on training split
        raw_khx = calibration.calibrate_k_hx_robust(
            train_slice,
            self._flow_rates_l_min,
            self._k_hx_by_group_w_k,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        if mode == "auto_weekly":
            cand_khx = {
                g: 0.80
                * self._k_hx_by_group_w_k.get(
                    g, house_config.DAY1_K_HX_BY_GROUP_W_K[g]
                )
                + 0.20
                * raw_khx.get(g, house_config.DAY1_K_HX_BY_GROUP_W_K[g])
                for g in const.ALL_HYDRAULIC_GROUPS
            }
        else:
            cand_khx = dict(raw_khx)

        # 2. Calibrate `wk` and `kt` (Choice A2 + B2 + D2 + D3 + Holdout Gate)
        out_wk, mem_wk = calibration.calibrate_zone_a2(
            columns,
            "wk",
            self._params_wk,
            mode=mode,
            episode_bank=self.episode_bank,
            memory=self.memory_wk,
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group=cand_khx,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        out_kt, mem_kt = calibration.calibrate_zone_a2(
            columns,
            "kt",
            self._params_kt,
            mode=mode,
            episode_bank=self.episode_bank,
            memory=self.memory_kt,
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group=cand_khx,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )

        # 3. Calibrate heat pump COP model
        cop_cols = (
            train_slice
            if len(train_slice.get("ts_utc", ())) >= const.STEPS_PER_DAY
            else columns
        )
        new_cop, mem_cop, cop_diag = _fit_cop_with_mode(
            cop_cols,
            self.cop_params,
            self.memory_cop,
            mode,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        cop_changed = new_cop != self.cop_params
        accepted_any = out_wk.accepted or out_kt.accepted or cop_changed

        if accepted_any:
            self.snapshot_stack.append(pre_snap)
            if out_wk.accepted:
                self._params_wk = out_wk.params_applied
                self.memory_wk = mem_wk
                self._k_hx_by_group_w_k["wk"] = out_wk.params_applied.k_hx_w_k
                self._k_hx_by_group_w_k["gang"] = cand_khx.get(
                    "gang", self._k_hx_by_group_w_k["gang"]
                )
            if out_kt.accepted:
                self._params_kt = out_kt.params_applied
                self.memory_kt = mem_kt
                self._k_hx_by_group_w_k["kantoor"] = (
                    out_kt.params_applied.k_hx_w_k
                )
            if cop_changed:
                self.cop_params = new_cop
                self.memory_cop = mem_cop
            self.cop_diagnostics = cop_diag
            self.hydraulics.update_k_hx(self._k_hx_by_group_w_k)
            self.replay_48h_from_columns(columns)
        else:
            self.cop_diagnostics = cop_diag

        self._last_posterior_std_wk = dict(out_wk.posterior_std)
        self._last_posterior_std_kt = dict(out_kt.posterior_std)
        self._last_corr_wk = dict(out_wk.correlation_matrix)
        self._last_corr_kt = dict(out_kt.correlation_matrix)
        self._last_collinear_wk = list(out_wk.collinearity_warnings)
        self._last_collinear_kt = list(out_kt.collinearity_warnings)

        self.refresh_14d_evaluation(columns)

        hist_entry = {
            "ts_utc": ts_now,
            "event": f"calibration_{mode}",
            "mode": mode,
            "accepted_wk": out_wk.accepted,
            "reason_wk": out_wk.reason,
            "holdout_j_wk_before": out_wk.holdout_before.overall_6h.j_eval_c,
            "holdout_j_wk_after": out_wk.holdout_after.overall_6h.j_eval_c,
            "accepted_kt": out_kt.accepted,
            "reason_kt": out_kt.reason,
            "holdout_j_kt_before": out_kt.holdout_before.overall_6h.j_eval_c,
            "holdout_j_kt_after": out_kt.holdout_after.overall_6h.j_eval_c,
            "cop_updated": cop_changed,
        }
        self.calibration_history.append(hist_entry)
        if len(self.calibration_history) > MAX_CALIBRATION_HISTORY:
            self.calibration_history = self.calibration_history[
                -MAX_CALIBRATION_HISTORY:
            ]

        return {
            "accepted_any": accepted_any,
            "accepted_wk": out_wk.accepted,
            "accepted_kt": out_kt.accepted,
            "outcome_wk": out_wk,
            "outcome_kt": out_kt,
            "cop_params": self.cop_params,
            "snapshot_depth": len(self.snapshot_stack),
        }

    def rollback_parameters(
        self,
        columns: Mapping[str, Sequence[float | int]] | None = None,
    ) -> bool:
        """Pops most recent parameter snapshot from 8-deep stack + 48h replay.

        Args:
            columns: Optional RingBuffer columns for 48h state replay.

        Returns:
            True if a snapshot was available and restored, else False.
        """
        if not self.snapshot_stack:
            return False

        snap = self.snapshot_stack.pop()
        self._apply_snapshot_dict(snap)

        if columns is not None and len(columns.get("ts_utc", ())) > 0:
            self.replay_48h_from_columns(columns)
            self.refresh_14d_evaluation(columns)
        else:
            self._refresh_cached_snapshots()

        self.calibration_history.append(
            {
                "ts_utc": self._last_ts_utc,
                "event": "rollback",
                "restored_snapshot_ts_utc": snap.get("ts_utc", 0),
                "snapshot_depth": len(self.snapshot_stack),
            }
        )
        return True

    def reset_parameters(
        self,
        columns: Mapping[str, Sequence[float | int]] | None = None,
    ) -> None:
        """Pushes active set onto stack, restores Day-1 priors, + 48h replay.

        Args:
            columns: Optional RingBuffer columns for 48h state replay.
        """
        self.snapshot_stack.append(self._capture_snapshot(reason="pre_reset"))

        self._params_wk = house_config.DAY1_PRIORS_WK
        self._params_kt = house_config.DAY1_PRIORS_KT
        self._k_hx_by_group_w_k = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
        self.cop_params = house_config.DAY1_COP_PARAMS
        self.memory_wk = None
        self.memory_kt = None
        self.memory_cop = None
        self._last_posterior_std_wk = {}
        self._last_posterior_std_kt = {}
        self._last_corr_wk = {}
        self._last_corr_kt = {}
        self._last_collinear_wk = []
        self._last_collinear_kt = []

        self.estimator_wk.params = self._params_wk
        self.estimator_kt.params = self._params_kt
        self.hydraulics.update_k_hx(self._k_hx_by_group_w_k)

        if columns is not None and len(columns.get("ts_utc", ())) > 0:
            self.replay_48h_from_columns(columns)
            self.refresh_14d_evaluation(columns)
        else:
            self._refresh_cached_snapshots()

        self.calibration_history.append(
            {
                "ts_utc": self._last_ts_utc,
                "event": "reset_to_day1_priors",
                "snapshot_depth": len(self.snapshot_stack),
            }
        )

    def refresh_14d_evaluation(
        self,
        columns: Mapping[str, Sequence[float | int]],
    ) -> dict[str, Any]:
        """Re-evaluates 14-day multi-regime open-loop hindcast for `wk` & `kt`.

        Args:
            columns: RingBuffer telemetry column dictionary.

        Returns:
            Dictionary `{"wk": ZoneEvaluationReport, "kt": ZoneEvaluationReport}`.
        """
        self.eval_report_wk = evaluation.evaluate_14d_hindcast(
            columns,
            "wk",
            self._params_wk,
            house_config.DAY1_PRIORS_WK,
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group=self._k_hx_by_group_w_k,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        self.eval_report_kt = evaluation.evaluate_14d_hindcast(
            columns,
            "kt",
            self._params_kt,
            house_config.DAY1_PRIORS_KT,
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group=self._k_hx_by_group_w_k,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        self._refresh_cached_snapshots()
        return {
            "wk": self.eval_report_wk,
            "kt": self.eval_report_kt,
        }

    def export_diagnostics(
        self,
        export_dir: str | pathlib.Path,
        columns: Mapping[str, Sequence[float | int]],
    ) -> tuple[str, str]:
        """Builds `fit_report` and exports `telemetry_5m.csv` + `fit_report.json`.

        Args:
            export_dir: Destination directory path.
            columns: RingBuffer telemetry column dictionary.

        Returns:
            Tuple `(csv_path_str, json_path_str)`.
        """
        ts_seq = columns.get("ts_utc", ())
        n_rows = len(ts_seq)
        if n_rows >= evaluation.SPINUP_48H_STEPS and (
            self.eval_report_wk is None or self.eval_report_kt is None
        ):
            self.refresh_14d_evaluation(columns)

        prev_snap = self.snapshot_stack[-1] if self.snapshot_stack else None

        fit_report: dict[str, Any] = {
            "ringbuffer": {
                "n_rows": n_rows,
                "ts_start_utc": int(ts_seq[0]) if n_rows > 0 else None,
                "ts_end_utc": int(ts_seq[-1]) if n_rows > 0 else None,
                "corrupt_chunks_recovered": 0,
            },
            "flow_weights_l_min": dict(self._flow_rates_l_min),
            "k_hx_by_group_w_k": dict(self._k_hx_by_group_w_k),
            "zones": {
                "wk": {
                    "parameters_active": _zone_params_to_dict(self._params_wk),
                    "posterior_std": dict(self._last_posterior_std_wk),
                    "correlation_matrix": dict(self._last_corr_wk),
                    "collinearity_warnings": list(self._last_collinear_wk),
                    "parameters_previous_snapshot": (
                        prev_snap.get("params_wk") if prev_snap else None
                    ),
                    "parameters_day1_prior": _zone_params_to_dict(
                        house_config.DAY1_PRIORS_WK
                    ),
                    "evaluation_14d": evaluation.zone_eval_report_to_dict(
                        self.eval_report_wk
                    ),
                },
                "kt": {
                    "parameters_active": _zone_params_to_dict(self._params_kt),
                    "posterior_std": dict(self._last_posterior_std_kt),
                    "correlation_matrix": dict(self._last_corr_kt),
                    "collinearity_warnings": list(self._last_collinear_kt),
                    "parameters_previous_snapshot": (
                        prev_snap.get("params_kt") if prev_snap else None
                    ),
                    "parameters_day1_prior": _zone_params_to_dict(
                        house_config.DAY1_PRIORS_KT
                    ),
                    "evaluation_14d": evaluation.zone_eval_report_to_dict(
                        self.eval_report_kt
                    ),
                },
            },
            "cop_model": {
                "parameters_active": _cop_params_to_dict(self.cop_params),
                "parameters_day1_prior": _cop_params_to_dict(
                    house_config.DAY1_COP_PARAMS
                ),
                "diagnostics": dict(self.cop_diagnostics),
            },
            "snapshot_stack_depth": len(self.snapshot_stack),
            "calibration_history": list(self.calibration_history),
        }

        return evaluation.export_csv_and_report(
            export_dir,
            columns,
            fit_report,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )

    # ========================================================================
    # Layer-2 Pure-Python Prediction, Control & Solver API (Component 5 §4.2)
    # ========================================================================
    def get_zone_fit_diagnostics(
        self,
        zone: contracts.ZoneId,
    ) -> dict[str, Any]:
        """Returns latest Fisher `1-sigma` std, correlations, and warnings."""
        if zone == "wk":
            return {
                "posterior_std": dict(self._last_posterior_std_wk),
                "correlation_matrix": dict(self._last_corr_wk),
                "collinearity_warnings": list(self._last_collinear_wk),
            }
        return {
            "posterior_std": dict(self._last_posterior_std_kt),
            "correlation_matrix": dict(self._last_corr_kt),
            "collinearity_warnings": list(self._last_collinear_kt),
        }

    def get_zone_snapshot(
        self,
        zone: contracts.ZoneId,
    ) -> contracts.ZoneSnapshot:
        """Returns O(1) immutable `ZoneSnapshot` for `zone`."""
        cached = self._snapshot_wk if zone == "wk" else self._snapshot_kt
        if cached is not None:
            return cached
        snap = self._build_snapshot(zone)
        if zone == "wk":
            self._snapshot_wk = snap
        else:
            self._snapshot_kt = snap
        return snap

    def simulate_step(
        self,
        zone: contracts.ZoneId,
        state: contracts.ThermalState | None,
        exog: contracts.ExogenousStepInput,
        actuator: contracts.ActuatorStepInput | None = None,
        dt_s: float = 300.0,
        step_index: int = 0,
    ) -> contracts.SimulationStepResult:
        """Executes one side-effect-free forward ZOH step from `state`.

        Args:
            zone: Zone identifier (`"wk"` or `"kt"`).
            state: Starting `ThermalState` (defaults to current zone snapshot).
            exog: Exogenous boundary conditions for the step.
            actuator: Optional floor/split-AC actuator command.
            dt_s: Step duration in seconds (default `300.0`).
            step_index: 1-based rollout index tag for the returned result.

        Returns:
            `SimulationStepResult` with propagated state and heat fluxes.
        """
        st_in = (
            state if state is not None else self.get_zone_snapshot(zone).state
        )
        params = self._params_wk if zone == "wk" else self._params_kt

        q_fh = 0.0
        q_ac = 0.0
        if actuator is not None:
            if actuator.q_fh_w is not None:
                q_fh = float(actuator.q_fh_w)
            elif actuator.t_supply_sec_c is not None:
                q_fh = (
                    max(0.0, float(actuator.u_eff))
                    * params.k_hx_w_k
                    * max(
                        0.0, float(actuator.t_supply_sec_c) - st_in.t_floor_c
                    )
                )
            if zone == "kt" and actuator.p_el_ac_w > 25.0:
                q_ac = evaluation.compute_ac_heating_power_w(
                    p_el_ac_w=float(actuator.p_el_ac_w),
                    ac_mode_code=1.0,
                    ac_defrost=False,
                    t_out_c=exog.t_out_c,
                    eta_ac=params.eta_ac,
                )

        st_next = zoh_kalman.propagate_open_loop_step(
            state=st_in,
            params=params,
            exog=exog,
            q_fh_w=q_fh,
            q_ac_w=q_ac,
            dt_s=dt_s,
        )
        t_surf = evaluation.estimate_zone_surface_temp(
            zone, st_next.t_air_c, st_next.t_floor_c, params
        )
        q_fl_to_air = (st_next.t_floor_c - st_next.t_air_c) * (
            1000.0 / params.r_fh_k_kw
        )
        h_oa = params.h_loss_w_k * (
            1.0 + params.c_wind * max(0.0, exog.v_wind_ms - 3.0)
        )
        q_shell = h_oa * (st_next.t_air_c - exog.t_out_c)

        return contracts.SimulationStepResult(
            step_index=step_index,
            state=st_next,
            t_surface_est_c=t_surf,
            q_fh_applied_w=q_fh,
            q_ac_applied_w=q_ac,
            q_floor_to_air_w=q_fl_to_air,
            q_shell_loss_w=q_shell,
        )

    def simulate_horizon(
        self,
        zone: contracts.ZoneId,
        exog_forecast: Sequence[contracts.ExogenousStepInput],
        actuators: (
            Sequence[contracts.ActuatorStepInput]
            | contracts.ActuatorStepInput
            | None
        ) = None,
        initial_state: contracts.ThermalState | None = None,
        dt_s: float = 300.0,
    ) -> list[contracts.SimulationStepResult]:
        """Runs side-effect-free multi-step ZOH rollout over `exog_forecast`.

        Args:
            zone: Zone identifier (`"wk"` or `"kt"`).
            exog_forecast: Sequence of future exogenous inputs.
            actuators: Per-step sequence, constant `ActuatorStepInput`, or
                `None` (passive coasting).
            initial_state: Optional starting `ThermalState`.
            dt_s: Step duration in seconds (default `300.0`).

        Returns:
            List of `SimulationStepResult` aligned with `exog_forecast`.
        """
        st = (
            initial_state
            if initial_state is not None
            else self.get_zone_snapshot(zone).state
        )
        results: list[contracts.SimulationStepResult] = []
        for idx, exog in enumerate(exog_forecast):
            if isinstance(actuators, Sequence):
                act: contracts.ActuatorStepInput | None = (
                    actuators[idx]
                    if idx < len(actuators)
                    else contracts.ActuatorStepInput()
                )
            else:
                act = actuators
            res = self.simulate_step(
                zone, st, exog, actuator=act, dt_s=dt_s, step_index=idx + 1
            )
            results.append(res)
            st = res.state
        return results

    def estimate_coast_duration(
        self,
        zone: contracts.ZoneId,
        floor_temp_min_air_c: float,
        exog_forecast: Sequence[contracts.ExogenousStepInput],
        initial_state: contracts.ThermalState | None = None,
        dt_s: float = 300.0,
    ) -> contracts.CoastEstimate:
        """Estimates passive coasting time (`Q_fh=0, P_ac=0`) & slab overshoot.

        Args:
            zone: Zone identifier (`"wk"` or `"kt"`).
            floor_temp_min_air_c: Lower comfort threshold for `T_air` [°C].
            exog_forecast: Forecast sequence of exogenous inputs.
            initial_state: Optional starting `ThermalState`.
            dt_s: Step duration in seconds (default `300.0`).

        Returns:
            `CoastEstimate` with coast duration, horizon min, and peak air temp.
        """
        st0 = (
            initial_state
            if initial_state is not None
            else self.get_zone_snapshot(zone).state
        )
        if not exog_forecast:
            return contracts.CoastEstimate(
                coast_duration_min=0,
                min_t_air_horizon_c=st0.t_air_c,
                peak_t_air_overshoot_c=st0.t_air_c,
            )

        step_min = max(1, int(round(dt_s / 60.0)))
        already_below = st0.t_air_c < floor_temp_min_air_c
        peak_air = st0.t_air_c
        min_air = st0.t_air_c
        coast_min: int | None = 0 if already_below else None

        st = st0
        for idx, exog in enumerate(exog_forecast):
            res = self.simulate_step(
                zone, st, exog, actuator=None, dt_s=dt_s, step_index=idx + 1
            )
            st = res.state
            if st.t_air_c > peak_air:
                peak_air = st.t_air_c
            if st.t_air_c < min_air:
                min_air = st.t_air_c
            if coast_min is None and st.t_air_c < floor_temp_min_air_c:
                coast_min = idx * step_min

        if coast_min is None:
            coast_min = len(exog_forecast) * step_min

        return contracts.CoastEstimate(
            coast_duration_min=coast_min,
            min_t_air_horizon_c=min_air,
            peak_t_air_overshoot_c=peak_air,
        )

    def estimate_preheat_lead_time(
        self,
        zone: contracts.ZoneId,
        target_temp_c: float,
        exog_forecast: Sequence[contracts.ExogenousStepInput],
        t_supply_sec_c: float = 30.0,
        q_fh_max_w: float | None = None,
        p_el_ac_w: float = 0.0,
        initial_state: contracts.ThermalState | None = None,
        max_surface_temp_c: float = 27.5,
        dt_s: float = 300.0,
    ) -> contracts.PreheatEstimate:
        """Estimates preheat lead time (multiples of 5 min) to `target_temp_c`.

        Args:
            zone: Zone identifier (`"wk"` or `"kt"`).
            target_temp_c: Target room air temperature [°C].
            exog_forecast: Forecast sequence of exogenous inputs.
            t_supply_sec_c: Assumed secondary supply water temperature [°C].
            q_fh_max_w: Optional explicit floor power override [W].
            p_el_ac_w: Optional split-AC electrical assist power for `kt` [W].
            initial_state: Optional starting `ThermalState`.
            max_surface_temp_c: PVC surface comfort cap [°C] (default `27.5`).
            dt_s: Step duration in seconds (default `300.0`).

        Returns:
            `PreheatEstimate` with reachability, lead time, and thermal kWh.
        """
        st = (
            initial_state
            if initial_state is not None
            else self.get_zone_snapshot(zone).state
        )
        if st.t_air_c >= target_temp_c:
            return contracts.PreheatEstimate(
                reachable=True,
                lead_time_min=0,
                predicted_t_air_end_c=st.t_air_c,
                predicted_t_floor_end_c=st.t_floor_c,
                total_thermal_energy_kwh=0.0,
            )

        if not exog_forecast:
            return contracts.PreheatEstimate(
                reachable=False,
                lead_time_min=0,
                predicted_t_air_end_c=st.t_air_c,
                predicted_t_floor_end_c=st.t_floor_c,
                total_thermal_energy_kwh=0.0,
            )

        params = self._params_wk if zone == "wk" else self._params_kt
        step_min = max(1, int(round(dt_s / 60.0)))
        dt_h = dt_s / 3600.0
        energy_kwh = 0.0

        for idx, exog in enumerate(exog_forecast):
            t_surf_now = evaluation.estimate_zone_surface_temp(
                zone, st.t_air_c, st.t_floor_c, params
            )
            u_eff = 0.0 if t_surf_now >= max_surface_temp_c else 1.0
            act = contracts.ActuatorStepInput(
                q_fh_w=(q_fh_max_w * u_eff) if q_fh_max_w is not None else None,
                t_supply_sec_c=t_supply_sec_c if q_fh_max_w is None else None,
                u_eff=u_eff,
                p_el_ac_w=p_el_ac_w if zone == "kt" else 0.0,
            )
            res = self.simulate_step(
                zone, st, exog, actuator=act, dt_s=dt_s, step_index=idx + 1
            )
            st = res.state
            energy_kwh += (
                (res.q_fh_applied_w + res.q_ac_applied_w) * dt_h / 1000.0
            )

            if st.t_air_c >= target_temp_c:
                return contracts.PreheatEstimate(
                    reachable=True,
                    lead_time_min=(idx + 1) * step_min,
                    predicted_t_air_end_c=st.t_air_c,
                    predicted_t_floor_end_c=st.t_floor_c,
                    total_thermal_energy_kwh=energy_kwh,
                )

        return contracts.PreheatEstimate(
            reachable=False,
            lead_time_min=len(exog_forecast) * step_min,
            predicted_t_air_end_c=st.t_air_c,
            predicted_t_floor_end_c=st.t_floor_c,
            total_thermal_energy_kwh=energy_kwh,
        )

    def estimate_total_sink_capacity_w(
        self,
        t_supply_sec_c: float,
        open_zones: Sequence[str] | Mapping[str, float],
    ) -> float:
        """Estimates total floor heat uptake [W] across `open_zones`.

        Args:
            t_supply_sec_c: Secondary manifold supply temperature [°C].
            open_zones: Open zone names or `{group: u_eff}` mapping.

        Returns:
            Estimated secondary sink power [W] for anti-short-cycling checks.
        """
        return float(
            self.hydraulics.estimate_total_sink_capacity_w(
                t_supply_sec_c,
                open_zones,
                self.estimator_wk.state.t_floor_c,
                self.estimator_kt.state.t_floor_c,
            )
        )

    def predict_cop_cv(
        self,
        t_out_c: float,
        t_sup_c: float,
        rh_out_pct: float = 80.0,
    ) -> float:
        """Predicts calibrated space-heating COP for `(t_out_c, t_sup_c, rh)`."""
        return cop_model.predict_cop_cv(
            self.cop_params, t_out_c, t_sup_c, rh_out_pct
        )

    def predict_cop_dhw(
        self,
        t_out_c: float,
        t_dhw_start_c: float,
        t_dhw_target_c: float,
        rh_out_pct: float = 80.0,
    ) -> float:
        """Predicts calibrated 2-stage DHW COP across `[start_c -> target_c]`."""
        return cop_model.predict_cop_dhw(
            self.cop_params, t_out_c, t_dhw_start_c, t_dhw_target_c, rh_out_pct
        )

    def export_solver_model(
        self,
        zone: contracts.ZoneId,
        dt_s: float = 1800.0,
    ) -> contracts.LinearizedDiscreteModel:
        """Exports discrete 2x2 `(Phi_2x2, Gamma_fh, Gamma_ac)` for MILP/MPC.

        Args:
            zone: Zone identifier (`"wk"` or `"kt"`).
            dt_s: Solver time step in seconds (default `1800.0` = 30 min).

        Returns:
            `LinearizedDiscreteModel` with exact ZOH matrices and `tau_coast`.
        """
        params = self._params_wk if zone == "wk" else self._params_kt
        phi_2x2, gamma_fh, gamma_ac = _discretize_2x2(params, dt_s)

        h_fh = 1000.0 / params.r_fh_k_kw
        h_down = params.h_down_w_k
        series_down = (
            (h_fh * h_down) / (h_fh + h_down) if (h_fh + h_down) > 1e-9 else 0.0
        )
        steady_loss = params.h_loss_w_k + params.h_up_w_k + series_down

        c_air_j = 3600.0 * params.c_air_wh_k
        c_floor_j = 3600.0 * params.c_floor_wh_k
        a00 = -(h_fh + params.h_loss_w_k + params.h_up_w_k) / c_air_j
        a01 = h_fh / c_air_j
        a10 = h_fh / c_floor_j
        a11 = -(h_fh + h_down) / c_floor_j
        tr = a00 + a11
        det = a00 * a11 - a01 * a10
        disc = max(0.0, tr * tr - 4.0 * det)
        lambda_slow = 0.5 * (tr + math.sqrt(disc))
        tau_coast_h = -1.0 / (min(-1e-9, lambda_slow) * 3600.0)

        return contracts.LinearizedDiscreteModel(
            zone=zone,
            dt_s=float(dt_s),
            phi_2x2=phi_2x2,
            gamma_fh=gamma_fh,
            gamma_ac_el_ref=gamma_ac,
            steady_loss_total_w_k=steady_loss,
            effective_tau_coast_h=tau_coast_h,
        )

    def update_flow_rates(self, flow_rates_l_min: Mapping[str, float]) -> None:
        """Updates manifold rotameter flow rates [L/min] from HA OptionsFlow."""
        self._flow_rates_l_min.update(
            {str(k): float(v) for k, v in flow_rates_l_min.items()}
        )
        self.hydraulics.update_flow_rates(self._flow_rates_l_min)

    # ========================================================================
    # Persistence (`betonbatterij.state`)
    # ========================================================================
    def to_state_dict(self) -> dict[str, Any]:
        """Serializes complete engine state for `betonbatterij.state` Store."""
        raw_state: dict[str, Any] = {
            "last_ts_utc": self._last_ts_utc,
            "hm_ema_c": self._hm_ema_c,
            "interfloor_cache": {
                k: [float(v[0]), int(v[1])]
                for k, v in self._interfloor_cache.items()
            },
            "desk_state_kt": dict(self._desk_state_kt),
            "flow_rates_l_min": dict(self._flow_rates_l_min),
            "k_hx_by_group_w_k": dict(self._k_hx_by_group_w_k),
            "params_wk": _zone_params_to_dict(self._params_wk),
            "params_kt": _zone_params_to_dict(self._params_kt),
            "estimator_wk": self.estimator_wk.to_dict(),
            "estimator_kt": self.estimator_kt.to_dict(),
            "hydraulics": self.hydraulics.to_dict(),
            "cop_params": _cop_params_to_dict(self.cop_params),
            "hindcast_wk": self.hindcast_wk.to_dict(),
            "hindcast_kt": self.hindcast_kt.to_dict(),
            "episode_bank": self.episode_bank.to_dict(),
            "memory_wk": calibration.posterior_memory_to_dict(self.memory_wk),
            "memory_kt": calibration.posterior_memory_to_dict(self.memory_kt),
            "memory_cop": calibration.posterior_memory_to_dict(
                self.memory_cop
            ),
            "eval_report_wk": evaluation.zone_eval_report_to_dict(
                self.eval_report_wk
            ),
            "eval_report_kt": evaluation.zone_eval_report_to_dict(
                self.eval_report_kt
            ),
            "cop_diagnostics": dict(self.cop_diagnostics),
            "calibration_history": list(self.calibration_history),
            "snapshot_stack": list(self.snapshot_stack),
            "last_q_fh_wk_w": self._last_q_fh_wk_w,
            "last_q_fh_kt_w": self._last_q_fh_kt_w,
            "last_t_air_meas_wk_c": self._last_t_air_meas_wk_c,
            "last_t_air_meas_kt_c": self._last_t_air_meas_kt_c,
            "posterior_std_wk": dict(self._last_posterior_std_wk),
            "posterior_std_kt": dict(self._last_posterior_std_kt),
            "corr_wk": dict(self._last_corr_wk),
            "corr_kt": dict(self._last_corr_kt),
            "collinear_wk": list(self._last_collinear_wk),
            "collinear_kt": list(self._last_collinear_kt),
        }
        sanitized: dict[str, Any] = evaluation.sanitize_json_payload(raw_state)
        return sanitized

    def load_state_dict(self, data: Mapping[str, Any]) -> None:
        """Restores complete engine state from `betonbatterij.state` dict."""
        if not data:
            return
        if "last_ts_utc" in data:
            self._last_ts_utc = int(data["last_ts_utc"])
        if "hm_ema_c" in data:
            hm_raw = data["hm_ema_c"]
            self._hm_ema_c = float(hm_raw) if hm_raw is not None else None

        if isinstance(data.get("interfloor_cache"), Mapping):
            self._interfloor_cache = {}
            for k, pair in data["interfloor_cache"].items():
                if (
                    isinstance(pair, Sequence)
                    and len(pair) == 2
                    and pair[0] is not None
                ):
                    self._interfloor_cache[str(k)] = (
                        float(pair[0]),
                        int(pair[1]),
                    )
        if isinstance(data.get("desk_state_kt"), Mapping):
            self._desk_state_kt = dict(data["desk_state_kt"])

        if isinstance(data.get("flow_rates_l_min"), Mapping):
            self._flow_rates_l_min.update(
                {
                    str(k): float(v)
                    for k, v in data["flow_rates_l_min"].items()
                    if v is not None
                }
            )
        if isinstance(data.get("k_hx_by_group_w_k"), Mapping):
            self._k_hx_by_group_w_k.update(
                {
                    str(k): float(v)
                    for k, v in data["k_hx_by_group_w_k"].items()
                    if v is not None
                }
            )

        if isinstance(data.get("params_wk"), Mapping):
            self._params_wk = _zone_params_from_dict(
                data["params_wk"], house_config.DAY1_PRIORS_WK
            )
        if isinstance(data.get("params_kt"), Mapping):
            self._params_kt = _zone_params_from_dict(
                data["params_kt"], house_config.DAY1_PRIORS_KT
            )

        if isinstance(data.get("estimator_wk"), Mapping):
            self.estimator_wk = zoh_kalman.ZoneKalmanEstimator.from_dict(
                data["estimator_wk"]
            )
            self.estimator_wk.params = self._params_wk
        if isinstance(data.get("estimator_kt"), Mapping):
            self.estimator_kt = zoh_kalman.ZoneKalmanEstimator.from_dict(
                data["estimator_kt"]
            )
            self.estimator_kt.params = self._params_kt
        if isinstance(data.get("hydraulics"), Mapping):
            self.hydraulics = hydraulics.HydraulicModel.from_dict(
                data["hydraulics"],
                flow_rates_l_min=self._flow_rates_l_min,
                k_hx_by_group_w_k=self._k_hx_by_group_w_k,
            )

        if "cop_params" in data:
            self.cop_params = _cop_params_from_dict(data.get("cop_params"))

        if isinstance(data.get("hindcast_wk"), Mapping):
            self.hindcast_wk = evaluation.LiveHindcastTracker.from_dict(
                data["hindcast_wk"]
            )
        if isinstance(data.get("hindcast_kt"), Mapping):
            self.hindcast_kt = evaluation.LiveHindcastTracker.from_dict(
                data["hindcast_kt"]
            )

        if isinstance(data.get("episode_bank"), Mapping):
            self.episode_bank = calibration.SeasonalEpisodeBank.from_dict(
                data["episode_bank"]
            )

        self.memory_wk = calibration.posterior_memory_from_dict(
            data.get("memory_wk")
        )
        self.memory_kt = calibration.posterior_memory_from_dict(
            data.get("memory_kt")
        )
        self.memory_cop = calibration.posterior_memory_from_dict(
            data.get("memory_cop")
        )

        self.eval_report_wk = evaluation.zone_eval_report_from_dict(
            data.get("eval_report_wk")
        )
        self.eval_report_kt = evaluation.zone_eval_report_from_dict(
            data.get("eval_report_kt")
        )

        if isinstance(data.get("cop_diagnostics"), Mapping):
            self.cop_diagnostics = dict(data["cop_diagnostics"])
        if isinstance(data.get("calibration_history"), Sequence):
            self.calibration_history = list(data["calibration_history"])[
                -MAX_CALIBRATION_HISTORY:
            ]

        self.snapshot_stack.clear()
        if isinstance(data.get("snapshot_stack"), Sequence):
            for snap in data["snapshot_stack"][-SNAPSHOT_STACK_MAX_DEPTH:]:
                if isinstance(snap, Mapping):
                    self.snapshot_stack.append(dict(snap))

        self._last_q_fh_wk_w = float(data.get("last_q_fh_wk_w", 0.0))
        self._last_q_fh_kt_w = float(data.get("last_q_fh_kt_w", 0.0))
        m_wk = data.get("last_t_air_meas_wk_c")
        m_kt = data.get("last_t_air_meas_kt_c")
        self._last_t_air_meas_wk_c = float(m_wk) if m_wk is not None else None
        self._last_t_air_meas_kt_c = float(m_kt) if m_kt is not None else None

        if isinstance(data.get("posterior_std_wk"), Mapping):
            self._last_posterior_std_wk = {
                str(k): float(v) for k, v in data["posterior_std_wk"].items()
            }
        if isinstance(data.get("posterior_std_kt"), Mapping):
            self._last_posterior_std_kt = {
                str(k): float(v) for k, v in data["posterior_std_kt"].items()
            }
        if isinstance(data.get("corr_wk"), Mapping):
            self._last_corr_wk = {
                str(k): {str(m): float(val) for m, val in row.items()}
                for k, row in data["corr_wk"].items()
                if isinstance(row, Mapping)
            }
        if isinstance(data.get("corr_kt"), Mapping):
            self._last_corr_kt = {
                str(k): {str(m): float(val) for m, val in row.items()}
                for k, row in data["corr_kt"].items()
                if isinstance(row, Mapping)
            }
        if isinstance(data.get("collinear_wk"), Sequence):
            self._last_collinear_wk = [
                str(x) for x in data["collinear_wk"]
            ]
        if isinstance(data.get("collinear_kt"), Sequence):
            self._last_collinear_kt = [
                str(x) for x in data["collinear_kt"]
            ]

        self._refresh_cached_snapshots()
