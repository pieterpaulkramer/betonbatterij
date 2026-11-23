"""DigitalTwinEngine facade for Home Assistant (`ha_bridge`) and Layer 2.

Owns the stateful pure-Python Layer-1 Digital Twin orchestration:
- Real-time 5-minute online estimation (`process_5m_window`) and 48-hour
  post-calibration replay (`replay_48h_from_columns`),
- Lock-free 3-phase batch calibration (`capture_calibration_context`, static
  `compute_calibration_plan`, `apply_calibration_plan`), 8-deep rollback stack,
  and Day-1 reset,
- Read-only `ThermalTwinProtocol` query, rollout, preheat, coasting, COP, and
  solver-export API for Layer-2 MPC/MILP consumers, and
- Complete JSON-safe state serialization (`to_state_dict`, `load_state_dict`).

Does not own Home Assistant entities, recorder/backfill queries, or disk
storage (`ha_bridge/`). Zero `homeassistant.*`, `numpy`, or `scipy` imports.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import math
import pathlib
from typing import Any, Literal

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import calibration
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import cop_model
from custom_components.betonbatterij.domain import evaluation
from custom_components.betonbatterij.domain import hydraulics
from custom_components.betonbatterij.domain import kalman_tape
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import reporting
from custom_components.betonbatterij.domain import solar_weather
from custom_components.betonbatterij.domain import zoh_kalman

SNAPSHOT_STACK_MAX_DEPTH: int = 8
MAX_CALIBRATION_HISTORY: int = 32


def _clone_memory(
    mem: contracts.ParameterPosteriorMemory | None,
) -> contracts.ParameterPosteriorMemory | None:
    """Returns a deep-copied `ParameterPosteriorMemory` or `None`."""
    if mem is None:
        return None
    return contracts.ParameterPosteriorMemory(
        mean=dict(mem.mean),
        precision=dict(mem.precision),
    )


def _evaluate_both_zones_14d(
    columns: Mapping[str, Sequence[float | int]],
    params_wk: contracts.ZoneParameters,
    params_kt: contracts.ZoneParameters,
    flow_rates_l_min: Mapping[str, float],
    k_hx_by_group: Mapping[str, float],
    lat_deg: float,
    lon_deg: float,
    *,
    precomputed_hist_eval: preprocessing.PreprocessedHistory | None = None,
    precomputed_hist_prior: preprocessing.PreprocessedHistory | None = None,
) -> tuple[contracts.ZoneEvaluationReport, contracts.ZoneEvaluationReport]:
    """Computes 14d `ZoneEvaluationReport`s for `wk` and `kt` purely."""
    ts_all = columns.get("ts_utc", ())
    n_total = len(ts_all)
    hist_eval_shared: preprocessing.PreprocessedHistory | None = None
    hist_prior_shared: preprocessing.PreprocessedHistory | None = None
    if n_total >= evaluation.SPINUP_48H_STEPS:
        start_slice = max(0, n_total - evaluation.MAX_EVAL_SLICE_STEPS)
        n_eval = n_total - start_slice
        sub_cols: Mapping[str, Sequence[float | int]] | None = None
        khx_eval = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
        for g_name, val in k_hx_by_group.items():
            khx_eval[preprocessing.zone_to_hydraulic_group(g_name)] = float(val)
        khx_eval["wk"] = params_wk.k_hx_w_k
        khx_eval["kantoor"] = params_kt.k_hx_w_k

        if (
            precomputed_hist_eval is not None
            and precomputed_hist_eval.n_rows == n_eval
            and abs(precomputed_hist_eval.hyd_k_hx_wk - khx_eval["wk"]) <= 1e-9
            and abs(precomputed_hist_eval.hyd_k_hx_kt - khx_eval["kantoor"])
            <= 1e-9
        ):
            hist_eval_shared = precomputed_hist_eval
        elif (
            precomputed_hist_eval is not None
            and precomputed_hist_eval.n_rows == n_eval
        ):
            sub_cols = preprocessing.slice_columns(
                columns, start_slice, n_total
            )
            hist_eval_shared = preprocessing.recompute_history_for_khx(
                precomputed_hist_eval,
                sub_cols,
                flow_rates_l_min=flow_rates_l_min,
                k_hx_by_group=khx_eval,
            )
        else:
            sub_cols = preprocessing.slice_columns(
                columns, start_slice, n_total
            )
            hist_eval_shared = preprocessing.preprocess_history_columns(
                sub_cols,
                flow_rates_l_min=flow_rates_l_min,
                k_hx_by_group=khx_eval,
                eta_ac_ref=params_kt.eta_ac,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
            )

        if (
            precomputed_hist_prior is not None
            and precomputed_hist_prior.n_rows == n_eval
        ):
            hist_prior_shared = precomputed_hist_prior
        else:
            if sub_cols is None:
                sub_cols = preprocessing.slice_columns(
                    columns, start_slice, n_total
                )
            hist_prior_shared = preprocessing.recompute_history_for_khx(
                hist_eval_shared,
                sub_cols,
                flow_rates_l_min=flow_rates_l_min,
                k_hx_by_group=priors.DAY1_K_HX_BY_GROUP_W_K,
            )

    eval_wk = evaluation.evaluate_14d_hindcast(
        columns,
        "wk",
        params_wk,
        priors.DAY1_PRIORS_WK,
        flow_rates_l_min=flow_rates_l_min,
        k_hx_by_group=k_hx_by_group,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        precomputed_hist_eval=hist_eval_shared,
        precomputed_hist_prior=hist_prior_shared,
    )
    eval_kt = evaluation.evaluate_14d_hindcast(
        columns,
        "kt",
        params_kt,
        priors.DAY1_PRIORS_KT,
        flow_rates_l_min=flow_rates_l_min,
        k_hx_by_group=k_hx_by_group,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        precomputed_hist_eval=hist_eval_shared,
        precomputed_hist_prior=hist_prior_shared,
    )
    return eval_wk, eval_kt


def _fit_cop_with_mode(
    columns: Mapping[str, Sequence[float | int]],
    current_cop: contracts.CopParameters,
    memory_cop: contracts.ParameterPosteriorMemory | None,
    mode: Literal["manual", "auto_weekly"],
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
    eval_columns: Mapping[str, Sequence[float | int]] | None = None,
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
        eval_columns=eval_columns,
    )
    diag = dict(diag_raw)

    if mode == "auto_weekly" and fitted_cop != current_cop:
        bnds = priors.COP_PARAM_BOUNDS
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
        episode_bank: Stratified 365-day `SeasonalEpisodeBank`.
        memory_wk: Seasonal posterior memory for `wk`.
        memory_kt: Seasonal posterior memory for `kt`.
        memory_cop: Seasonal posterior memory for COP parameters.
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
                `priors.LATITUDE_DEG`).
            longitude_deg: Optional site longitude [deg E] (non-zero overrides
                `priors.LONGITUDE_DEG`).
        """
        self.latitude_deg: float = (
            float(latitude_deg)
            if latitude_deg is not None
            and math.isfinite(latitude_deg)
            and float(latitude_deg) != 0.0
            else priors.LATITUDE_DEG
        )
        self.longitude_deg: float = (
            float(longitude_deg)
            if longitude_deg is not None
            and math.isfinite(longitude_deg)
            and float(longitude_deg) != 0.0
            else priors.LONGITUDE_DEG
        )
        self._flow_rates_l_min: dict[str, float] = dict(
            const.DEFAULT_FLOW_RATES_L_MIN
        )
        if flow_rates_l_min:
            self._flow_rates_l_min.update(
                {k: float(v) for k, v in flow_rates_l_min.items()}
            )

        self._k_hx_by_group_w_k: dict[str, float] = dict(
            priors.DAY1_K_HX_BY_GROUP_W_K
        )
        if k_hx_by_group_w_k:
            for g, v in k_hx_by_group_w_k.items():
                self._k_hx_by_group_w_k[
                    preprocessing.zone_to_hydraulic_group(g)
                ] = float(v)

        p_wk = params_wk or dataclasses.replace(
            priors.DAY1_PRIORS_WK,
            k_hx_w_k=self._k_hx_by_group_w_k["wk"],
        )
        p_kt = params_kt or dataclasses.replace(
            priors.DAY1_PRIORS_KT,
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
            cop_params or priors.DAY1_COP_PARAMS
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
        self._last_bound_warnings_wk: list[str] = []
        self._last_bound_warnings_kt: list[str] = []

        self._param_epoch: int = 0
        self._snapshot_wk: contracts.ZoneSnapshot | None = None
        self._snapshot_kt: contracts.ZoneSnapshot | None = None

    @property
    def param_epoch(self) -> int:
        """Monotonic counter incremented on every parameter mutation."""
        return self._param_epoch

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
        t_surf = zoh_kalman.estimate_zone_surface_temp(
            zone, st.t_air_c, st.t_floor_c, params
        )
        stored_wh = params.c_floor_wh_k * (st.t_floor_c - st.t_air_c)
        rmse_6h = ev.overall_6h.rmse_end_c if ev is not None else None
        warmup_rem = (
            int(hind.warmup_remaining_min)
            if (est.initialized or len(hind.queue) > 0)
            else 0
        )

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
            warmup_remaining_min=warmup_rem,
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
        parsed = preprocessing.parse_raw_step(
            raw_row,
            default_ts_utc=self._last_ts_utc + int(round(dt_s)),
            default_t_wk=(
                self.estimator_wk.state.t_air_c
                if self.estimator_wk.initialized
                else 19.5
            ),
            default_t_kt=(
                self.estimator_kt.state.t_air_c
                if self.estimator_kt.initialized
                else 19.0
            ),
        )
        ts_utc = parsed.ts_utc
        fl = parsed.flags
        eff_stale_wk = stale_wk or (
            parsed.has_flags and not (fl & const.FLAG_VALID_WK)
        )
        eff_stale_kt = stale_kt or (
            parsed.has_flags and not (fl & const.FLAG_VALID_KT)
        )
        aanwezig = parsed.aanwezig

        if parsed.t_hm_c is not None:
            self._hm_ema_c = (
                parsed.t_hm_c
                if self._hm_ema_c is None
                else (parsed.t_hm_c / 6.0) + (5.0 / 6.0) * self._hm_ema_c
            )
        t_out_c = solar_weather.select_outdoor_temp(
            ts_utc=ts_utc,
            t_dak_c=parsed.t_dak_c,
            t_weather_c=parsed.t_weather_c,
            t_hm_ema_c=self._hm_ema_c,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        v_wind = parsed.v_wind_ms

        pv_fb_hint = bool(fl & const.FLAG_PV_FALLBACK)
        if parsed.has_preproj_solar:
            i_sol_wk = parsed.i_sol_wk_preproj_wm2
            i_sol_kt = parsed.i_sol_kt_preproj_wm2
        else:
            i_wk_raw, i_kt_raw, _, _ = solar_weather.project_solar_irradiance(
                ts_utc=ts_utc,
                p_pv_w=parsed.p_pv_w,
                p_pv_forecast_w=parsed.p_pv_forecast_w,
                epex_eur_kwh=parsed.epex_eur_kwh,
                screen_small_pos=parsed.screen_small_pos,
                screen_large_pos=parsed.screen_large_pos,
                pv_stale=pv_fb_hint,
                lat_deg=self.latitude_deg,
                lon_deg=self.longitude_deg,
            )
            i_sol_wk = max(0.0, i_wk_raw)
            i_sol_kt = max(0.0, i_kt_raw)

        t_wk = parsed.t_air_wk_c
        t_kt = parsed.t_air_kt_c
        room_temps = parsed.room_temps
        u_pump = parsed.u_pump_sec
        u_valves = parsed.u_valves
        v_prim = parsed.v_prim_l_min
        t_top_e = parsed.t_buf_top_end_c
        max_gap = const.MAX_RESTART_BRIDGE_GAP_S

        q_int_kt, q_ac_kt = solar_weather.compute_internal_and_ac_loads_kt(
            p_desk_w=parsed.p_desk_kt_w,
            pir_occ=parsed.pir_occ_kt,
            p_growlights_w=parsed.p_growlights_kt_w,
            p_el_ac_w=parsed.p_el_ac_kt_w,
            ac_mode_code=parsed.ac_mode_code,
            ac_defrost=parsed.ac_defrost,
            t_out_c=t_out_c,
            eta_ac=self._params_kt.eta_ac,
            desk_state=self._desk_state_kt,
            dt_s=min(dt_s, max_gap),
        )
        t_up_wk, t_dn_kt = solar_weather.compute_interfloor_boundaries(
            room_temps_c=room_temps,
            last_valid_cache=self._interfloor_cache,
            ts_utc=ts_utc,
        )
        t_dn_wk = priors.T_GROUND_MEAN_C

        # Initialize estimators on very first tick or >3h outage
        if not self.estimator_wk.initialized or dt_s > max_gap:
            ex_init_wk = contracts.ExogenousStepInput(
                t_out_c=t_out_c,
                v_wind_ms=v_wind,
                i_sol_wm2=i_sol_wk,
                t_down_c=t_dn_wk,
                q_int_known_w=priors.BASELOAD_WK_W,
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
            if dt_s > max_gap:
                self.hindcast_wk.clear()

        if not self.estimator_kt.initialized or dt_s > max_gap:
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
            if dt_s > max_gap:
                self.hindcast_kt.clear()

        hyd_res = self.hydraulics.step(
            dt_s=min(dt_s, max_gap),
            u_pump_sec=u_pump,
            u_valves=u_valves,
            threeway_room=parsed.threeway_room,
            wp_defrost=parsed.wp_defrost,
            v_prim_l_min=v_prim,
            p_prim_th_w=parsed.p_prim_th_w,
            f_comp_hz=parsed.f_comp_hz,
            t_buf_top_start_c=parsed.t_buf_top_start_c,
            t_buf_top_end_c=t_top_e,
            t_buf_bot_start_c=parsed.t_buf_bot_start_c,
            t_buf_bot_end_c=parsed.t_buf_bot_end_c,
            t_floor_wk_c=self.estimator_wk.state.t_floor_c,
            t_floor_kt_c=self.estimator_kt.state.t_floor_c,
            room_temps_c=room_temps,
            t_bg_sup_c=parsed.t_bg_sup_c,
            t_bg_ret_c=parsed.t_bg_ret_c,
            bg_ret_offset_c=0.0,
        )
        q_fh_wk = hyd_res.q_fh_wk_w
        q_fh_kt = hyd_res.q_fh_kt_w
        if u_pump <= 1e-4 and v_prim <= 1e-4:
            if parsed.q_fh_wk_override_w > 0.0:
                q_fh_wk = parsed.q_fh_wk_override_w
            if parsed.q_fh_kt_override_w > 0.0:
                q_fh_kt = parsed.q_fh_kt_override_w

        q_int_wk = priors.BASELOAD_WK_W + max(0.0, hyd_res.q_leak_1e_to_wk_w)
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

        t_surf_meas_wk = parsed.t_surf_meas_wk_c
        meas_wk = None if eff_stale_wk else t_wk
        meas_kt = None if eff_stale_kt else t_kt
        if (
            not temp_changed_wk
            and meas_wk is not None
            and self.estimator_wk.last_meas_temp_c is not None
            and abs(meas_wk - self.estimator_wk.last_meas_temp_c) >= 1e-4
        ):
            temp_changed_wk = True
        if (
            not temp_changed_kt
            and meas_kt is not None
            and self.estimator_kt.last_meas_temp_c is not None
            and abs(meas_kt - self.estimator_kt.last_meas_temp_c) >= 1e-4
        ):
            temp_changed_kt = True

        if 450.0 < dt_s <= max_gap:
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
            step_hind_dt = dt_s if dt_s <= max_gap else float(const.DT_STEP_S)

        state_wk = self.estimator_wk.step(
            ts_utc=ts_utc,
            dt_s=min(dt_s, max_gap),
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
            dt_s=min(dt_s, max_gap),
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
        sub_cols = preprocessing.slice_columns(columns, start_idx, n_total)

        self._k_hx_by_group_w_k["wk"] = self._params_wk.k_hx_w_k
        self._k_hx_by_group_w_k["kantoor"] = self._params_kt.k_hx_w_k
        self.hydraulics = hydraulics.HydraulicModel(
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group_w_k=self._k_hx_by_group_w_k,
        )

        hist = preprocessing.preprocess_history_columns(
            sub_cols,
            flow_rates_l_min=self._flow_rates_l_min,
            k_hx_by_group=self._k_hx_by_group_w_k,
            eta_ac_ref=self._params_kt.eta_ac,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )
        exog_wk = preprocessing.build_zone_exog_list(hist, "wk")
        exog_kt = preprocessing.build_zone_exog_list(hist, "kt")

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
            "params_wk": self._params_wk.to_dict(),
            "params_kt": self._params_kt.to_dict(),
            "k_hx_by_group_w_k": dict(self._k_hx_by_group_w_k),
            "cop_params": self.cop_params.to_dict(),
            "memory_wk": calibration.posterior_memory_to_dict(self.memory_wk),
            "memory_kt": calibration.posterior_memory_to_dict(self.memory_kt),
            "memory_cop": calibration.posterior_memory_to_dict(
                self.memory_cop
            ),
            "bound_warnings_wk": list(self._last_bound_warnings_wk),
            "bound_warnings_kt": list(self._last_bound_warnings_kt),
        }

    def _apply_snapshot_dict(self, snap: Mapping[str, Any]) -> None:
        """Restores active parameters and seasonal memory from `snap`."""
        self._params_wk = contracts.ZoneParameters.from_dict(
            snap.get("params_wk"), defaults=priors.DAY1_PRIORS_WK
        )
        self._params_kt = contracts.ZoneParameters.from_dict(
            snap.get("params_kt"), defaults=priors.DAY1_PRIORS_KT
        )

        khx_raw = snap.get("k_hx_by_group_w_k")
        self._k_hx_by_group_w_k = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
        if isinstance(khx_raw, Mapping):
            self._k_hx_by_group_w_k.update(
                {
                    str(k): float(v)
                    for k, v in khx_raw.items()
                    if preprocessing.is_finite_number(v)
                }
            )
        self._k_hx_by_group_w_k["wk"] = self._params_wk.k_hx_w_k
        self._k_hx_by_group_w_k["kantoor"] = self._params_kt.k_hx_w_k

        self.cop_params = contracts.CopParameters.from_dict(
            snap.get("cop_params"), defaults=priors.DAY1_COP_PARAMS
        )
        self.memory_wk = calibration.posterior_memory_from_dict(
            snap.get("memory_wk")
        )
        self.memory_kt = calibration.posterior_memory_from_dict(
            snap.get("memory_kt")
        )
        self.memory_cop = calibration.posterior_memory_from_dict(
            snap.get("memory_cop")
        )
        if isinstance(snap.get("bound_warnings_wk"), Sequence):
            self._last_bound_warnings_wk = [
                str(x) for x in snap["bound_warnings_wk"]
            ]
        else:
            self._last_bound_warnings_wk = []
        if isinstance(snap.get("bound_warnings_kt"), Sequence):
            self._last_bound_warnings_kt = [
                str(x) for x in snap["bound_warnings_kt"]
            ]
        else:
            self._last_bound_warnings_kt = []

        self.estimator_wk.params = self._params_wk
        self.estimator_kt.params = self._params_kt
        self.hydraulics.update_k_hx(self._k_hx_by_group_w_k)

    def capture_calibration_context(self) -> dict[str, Any]:
        """Captures an isolated `< 1 ms` calibration context under lock."""
        return {
            "param_epoch": int(self._param_epoch),
            "params_wk": self._params_wk,
            "params_kt": self._params_kt,
            "flow_rates_l_min": dict(self._flow_rates_l_min),
            "k_hx_by_group_w_k": dict(self._k_hx_by_group_w_k),
            "cop_params": self.cop_params,
            "memory_wk": _clone_memory(self.memory_wk),
            "memory_kt": _clone_memory(self.memory_kt),
            "memory_cop": _clone_memory(self.memory_cop),
            "episode_bank": self.episode_bank.clone(),
            "bound_warnings_wk": list(self._last_bound_warnings_wk),
            "bound_warnings_kt": list(self._last_bound_warnings_kt),
            "snapshot_depth": len(self.snapshot_stack),
            "latitude_deg": float(self.latitude_deg),
            "longitude_deg": float(self.longitude_deg),
            "last_ts_utc": int(self._last_ts_utc),
        }

    @staticmethod
    def compute_calibration_plan(
        columns: Mapping[str, Sequence[float | int]],
        context: Mapping[str, Any],
        mode: Literal["manual", "auto_weekly"] = "manual",
    ) -> dict[str, Any]:
        """Computes a lock-free calibration plan with zero state mutation.

        Implemented as a `@staticmethod` to structurally guarantee that
        background executor calibration cannot read or mutate engine state.

        Args:
            columns: Snapshot of full RingBuffer telemetry columns.
            context: Decoupled dictionary captured by
                `capture_calibration_context()`.
            mode: `"manual"` or `"auto_weekly"`.

        Returns:
            Pure calibration plan dictionary ready for `apply_calibration_plan`.
        """
        param_epoch = int(context["param_epoch"])
        params_wk: contracts.ZoneParameters = context["params_wk"]
        params_kt: contracts.ZoneParameters = context["params_kt"]
        flow_rates: dict[str, float] = dict(context["flow_rates_l_min"])
        k_hx_by_group: dict[str, float] = dict(context["k_hx_by_group_w_k"])
        cop_params: contracts.CopParameters = context["cop_params"]
        mem_wk_in = _clone_memory(context.get("memory_wk"))
        mem_kt_in = _clone_memory(context.get("memory_kt"))
        mem_cop_in = _clone_memory(context.get("memory_cop"))
        snapshot_depth = int(context.get("snapshot_depth", 0))
        lat_deg = float(context.get("latitude_deg", priors.LATITUDE_DEG))
        lon_deg = float(context.get("longitude_deg", priors.LONGITUDE_DEG))
        last_ts_utc = int(context.get("last_ts_utc", 0))

        ts_seq = columns.get("ts_utc", ())
        n_total = len(ts_seq)
        if n_total == 0:
            return {
                "empty": True,
                "param_epoch": param_epoch,
                "accepted_any": False,
                "accepted_wk": False,
                "accepted_kt": False,
                "reason": "Empty telemetry history.",
                "cop_params": cop_params,
                "snapshot_depth": snapshot_depth,
            }

        ts_now = int(ts_seq[-1])

        if mode == "auto_weekly":
            gate_ok, gate_reason = calibration.check_seasonal_calibration_gate(
                columns,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
            )
            if not gate_ok:
                eval_wk_blk, eval_kt_blk = _evaluate_both_zones_14d(
                    columns,
                    params_wk,
                    params_kt,
                    flow_rates,
                    k_hx_by_group,
                    lat_deg,
                    lon_deg,
                )
                return {
                    "param_epoch": param_epoch,
                    "blocked_by_seasonal_gate": True,
                    "accepted_any": False,
                    "accepted_wk": False,
                    "accepted_kt": False,
                    "reason": gate_reason,
                    "eval_wk": eval_wk_blk,
                    "eval_kt": eval_kt_blk,
                    "snapshot_depth": snapshot_depth,
                }

        pre_snap = {
            "ts_utc": ts_now if n_total > 0 else last_ts_utc,
            "reason": f"{mode}_calibration",
            "params_wk": params_wk.to_dict(),
            "params_kt": params_kt.to_dict(),
            "k_hx_by_group_w_k": dict(k_hx_by_group),
            "cop_params": cop_params.to_dict(),
            "memory_wk": calibration.posterior_memory_to_dict(mem_wk_in),
            "memory_kt": calibration.posterior_memory_to_dict(mem_kt_in),
            "memory_cop": calibration.posterior_memory_to_dict(mem_cop_in),
            "bound_warnings_wk": list(context.get("bound_warnings_wk", ())),
            "bound_warnings_kt": list(context.get("bound_warnings_kt", ())),
        }

        bank_ctx = context.get("episode_bank")
        if isinstance(bank_ctx, calibration.SeasonalEpisodeBank):
            local_bank = bank_ctx.clone()
        elif isinstance(context.get("episode_bank_dict"), Mapping):
            local_bank = calibration.SeasonalEpisodeBank.from_dict(
                context["episode_bank_dict"]
            )
        else:
            local_bank = calibration.SeasonalEpisodeBank()

        # Shared raw-aware chronological Train / Holdout split
        is_lts_all, _ = preprocessing.detect_lts_and_missing_prim_rows(columns)
        train_end = preprocessing.compute_train_holdout_split(
            columns, is_lts_row=is_lts_all
        )

        # Curate EpisodeBank strictly from training split (no holdout leakage)
        hist_train_init: preprocessing.PreprocessedHistory | None = None
        if n_total >= 4 * const.STEPS_PER_DAY and train_end < n_total:
            train_slice: Mapping[str, Sequence[float | int]] = (
                preprocessing.slice_columns(columns, 0, train_end)
            )
            hist_train_init = preprocessing.preprocess_history_columns(
                train_slice,
                flow_rates_l_min=flow_rates,
                k_hx_by_group=k_hx_by_group,
                eta_ac_ref=params_kt.eta_ac,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
            )
            local_bank.update_from_columns(
                train_slice,
                flow_rates_l_min=flow_rates,
                k_hx_by_group=k_hx_by_group,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                precomputed_hist=hist_train_init,
            )
        else:
            train_slice = dict(columns)

        # 1. Calibrate steady-state K_hx on training split (damped once in auto_weekly)
        raw_khx = calibration.calibrate_k_hx_robust(
            train_slice,
            flow_rates,
            k_hx_by_group,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            precomputed_hist=hist_train_init,
        )
        khx_damped_here = mode == "auto_weekly"
        if khx_damped_here:
            cand_khx = {
                g: 0.80
                * k_hx_by_group.get(g, priors.DAY1_K_HX_BY_GROUP_W_K[g])
                + 0.20 * raw_khx.get(g, priors.DAY1_K_HX_BY_GROUP_W_K[g])
                for g in const.ALL_HYDRAULIC_GROUPS
            }
        else:
            cand_khx = dict(raw_khx)

        # Share precomputed train/holdout history across `wk` and `kt`
        hist_train_cand: preprocessing.PreprocessedHistory | None = None
        hist_hold_cand: preprocessing.PreprocessedHistory | None = None
        hist_hold_prior: preprocessing.PreprocessedHistory | None = None
        if n_total >= 4 * const.STEPS_PER_DAY and train_end < n_total:
            if (
                hist_train_init is not None
                and cand_khx == k_hx_by_group
            ):
                hist_train_cand = hist_train_init
            elif hist_train_init is not None:
                hist_train_cand = preprocessing.recompute_history_for_khx(
                    hist_train_init,
                    train_slice,
                    flow_rates_l_min=flow_rates,
                    k_hx_by_group=cand_khx,
                )
            else:
                hist_train_cand = preprocessing.preprocess_history_columns(
                    train_slice,
                    flow_rates_l_min=flow_rates,
                    k_hx_by_group=cand_khx,
                    eta_ac_ref=params_kt.eta_ac,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                )
            holdout_slice_start = max(
                0, train_end - evaluation.SPINUP_48H_STEPS
            )
            holdout_cols = preprocessing.slice_columns(
                columns, holdout_slice_start, n_total
            )
            hist_hold_cand = preprocessing.preprocess_history_columns(
                holdout_cols,
                flow_rates_l_min=flow_rates,
                k_hx_by_group=cand_khx,
                eta_ac_ref=params_kt.eta_ac,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
            )
            kalman_tape.get_zone_tape(hist_hold_cand, "wk")
            kalman_tape.get_zone_tape(hist_hold_cand, "kt")
            hist_hold_prior = preprocessing.recompute_history_for_khx(
                hist_hold_cand,
                holdout_cols,
                flow_rates_l_min=flow_rates,
                k_hx_by_group=priors.DAY1_K_HX_BY_GROUP_W_K,
            )

        # 2. Calibrate `wk` and `kt` (Joint Trajectory MAP + Holdout Gate)
        out_wk, mem_wk = calibration.calibrate_zone_a2(
            columns,
            "wk",
            params_wk,
            mode=mode,
            episode_bank=local_bank,
            memory=mem_wk_in,
            flow_rates_l_min=flow_rates,
            k_hx_by_group=cand_khx,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            precomputed_train_hist=hist_train_cand,
            precomputed_holdout_hist=hist_hold_cand,
            precomputed_holdout_prior_hist=hist_hold_prior,
            is_lts_row=is_lts_all,
            khx_already_damped=khx_damped_here,
        )
        out_kt, mem_kt = calibration.calibrate_zone_a2(
            columns,
            "kt",
            params_kt,
            mode=mode,
            episode_bank=local_bank,
            memory=mem_kt_in,
            flow_rates_l_min=flow_rates,
            k_hx_by_group=cand_khx,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            precomputed_train_hist=hist_train_cand,
            precomputed_holdout_hist=hist_hold_cand,
            precomputed_holdout_prior_hist=hist_hold_prior,
            is_lts_row=is_lts_all,
            khx_already_damped=khx_damped_here,
        )

        # 3. Calibrate heat pump COP model on `train_slice` once raw rows exist
        n_non_lts_train = sum(
            1
            for idx in range(min(train_end, len(is_lts_all)))
            if not is_lts_all[idx]
        )
        cop_cols = train_slice if n_non_lts_train > 0 else columns
        new_cop, mem_cop, cop_diag = _fit_cop_with_mode(
            cop_cols,
            cop_params,
            mem_cop_in,
            mode,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            eval_columns=columns,
        )
        cop_changed = new_cop != cop_params
        accepted_any = out_wk.accepted or out_kt.accepted or cop_changed

        next_params_wk = out_wk.params_applied if out_wk.accepted else params_wk
        next_params_kt = out_kt.params_applied if out_kt.accepted else params_kt
        next_khx = dict(k_hx_by_group)
        if out_wk.accepted:
            next_khx["wk"] = next_params_wk.k_hx_w_k
            next_khx["gang"] = cand_khx.get("gang", next_khx["gang"])
        if out_kt.accepted:
            next_khx["kantoor"] = next_params_kt.k_hx_w_k

        can_reuse_14d = (
            n_total >= evaluation.SPINUP_48H_STEPS
            and (n_total - train_end) == evaluation.EVAL_14D_STEPS
            and next_khx == cand_khx
        )
        reuse_wk = (
            can_reuse_14d
            and out_wk.accepted
            and out_wk.params_applied == out_wk.params_map_raw
        )
        reuse_kt = (
            can_reuse_14d
            and out_kt.accepted
            and out_kt.params_applied == out_kt.params_map_raw
        )
        if reuse_wk and reuse_kt:
            eval_wk = out_wk.holdout_after
            eval_kt = out_kt.holdout_after
        elif reuse_wk:
            eval_wk = out_wk.holdout_after
            eval_kt = evaluation.evaluate_14d_hindcast(
                columns,
                "kt",
                next_params_kt,
                priors.DAY1_PRIORS_KT,
                flow_rates_l_min=flow_rates,
                k_hx_by_group=next_khx,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                precomputed_hist_eval=hist_hold_cand,
                precomputed_hist_prior=hist_hold_prior,
            )
        elif reuse_kt:
            eval_wk = evaluation.evaluate_14d_hindcast(
                columns,
                "wk",
                next_params_wk,
                priors.DAY1_PRIORS_WK,
                flow_rates_l_min=flow_rates,
                k_hx_by_group=next_khx,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                precomputed_hist_eval=hist_hold_cand,
                precomputed_hist_prior=hist_hold_prior,
            )
            eval_kt = out_kt.holdout_after
        else:
            eval_wk, eval_kt = _evaluate_both_zones_14d(
                columns,
                next_params_wk,
                next_params_kt,
                flow_rates,
                next_khx,
                lat_deg,
                lon_deg,
                precomputed_hist_eval=hist_hold_cand,
                precomputed_hist_prior=hist_hold_prior,
            )

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

        return {
            "empty": False,
            "param_epoch": param_epoch,
            "mode": mode,
            "accepted_any": accepted_any,
            "accepted_wk": out_wk.accepted,
            "accepted_kt": out_kt.accepted,
            "cop_changed": cop_changed,
            "pre_snap": pre_snap,
            "episode_bank": local_bank,
            "outcome_wk": out_wk,
            "outcome_kt": out_kt,
            "memory_wk": mem_wk,
            "memory_kt": mem_kt,
            "new_cop": new_cop,
            "memory_cop": mem_cop,
            "cop_diagnostics": cop_diag,
            "next_k_hx": next_khx,
            "eval_wk": eval_wk,
            "eval_kt": eval_kt,
            "history_entry": hist_entry,
        }

    def apply_calibration_plan(
        self,
        plan: Mapping[str, Any],
        recent_columns: Mapping[str, Sequence[float | int]] | None = None,
    ) -> dict[str, Any]:
        """Applies a computed calibration plan in `< 20 ms` under lock."""
        if plan.get("empty"):
            return {
                "empty": True,
                "accepted_any": False,
                "accepted_wk": False,
                "accepted_kt": False,
                "reason": str(plan.get("reason", "Empty telemetry history.")),
                "cop_params": self.cop_params,
                "snapshot_depth": len(self.snapshot_stack),
            }

        if plan.get("param_epoch") != self._param_epoch:
            return {
                "accepted_any": False,
                "accepted_wk": False,
                "accepted_kt": False,
                "aborted_stale_epoch": True,
                "reason": (
                    "Aborted: parameters changed concurrently during "
                    "calibration."
                ),
                "snapshot_depth": len(self.snapshot_stack),
            }

        if plan.get("blocked_by_seasonal_gate"):
            if plan.get("eval_wk") is not None:
                self.eval_report_wk = plan["eval_wk"]
            if plan.get("eval_kt") is not None:
                self.eval_report_kt = plan["eval_kt"]
            self._refresh_cached_snapshots()
            return {
                "accepted_any": False,
                "accepted_wk": False,
                "accepted_kt": False,
                "blocked_by_seasonal_gate": True,
                "reason": str(plan.get("reason", "")),
                "snapshot_depth": len(self.snapshot_stack),
            }

        bank_obj = plan.get("episode_bank")
        if isinstance(bank_obj, calibration.SeasonalEpisodeBank):
            self.episode_bank = bank_obj
        elif isinstance(plan.get("episode_bank_dict"), Mapping):
            self.episode_bank = calibration.SeasonalEpisodeBank.from_dict(
                plan["episode_bank_dict"]
            )

        out_wk: contracts.CalibrationZoneOutcome = plan["outcome_wk"]
        out_kt: contracts.CalibrationZoneOutcome = plan["outcome_kt"]
        accepted_any = bool(plan["accepted_any"])
        cop_changed = bool(plan["cop_changed"])

        self.cop_diagnostics = dict(plan["cop_diagnostics"])
        self.eval_report_wk = plan["eval_wk"]
        self.eval_report_kt = plan["eval_kt"]

        if accepted_any:
            self.snapshot_stack.append(dict(plan["pre_snap"]))
            if out_wk.accepted:
                self._params_wk = out_wk.params_applied
                self.memory_wk = _clone_memory(plan["memory_wk"])
                self._k_hx_by_group_w_k["wk"] = out_wk.params_applied.k_hx_w_k
                self._k_hx_by_group_w_k["gang"] = float(
                    plan["next_k_hx"]["gang"]
                )
            if out_kt.accepted:
                self._params_kt = out_kt.params_applied
                self.memory_kt = _clone_memory(plan["memory_kt"])
                self._k_hx_by_group_w_k["kantoor"] = (
                    out_kt.params_applied.k_hx_w_k
                )
            if cop_changed:
                self.cop_params = plan["new_cop"]
                self.memory_cop = _clone_memory(plan["memory_cop"])

            self.estimator_wk.params = self._params_wk
            self.estimator_kt.params = self._params_kt
            self.hydraulics.update_k_hx(self._k_hx_by_group_w_k)
            self._param_epoch += 1

        self._last_posterior_std_wk = dict(out_wk.posterior_std)
        self._last_posterior_std_kt = dict(out_kt.posterior_std)
        self._last_corr_wk = dict(out_wk.correlation_matrix)
        self._last_corr_kt = dict(out_kt.correlation_matrix)
        self._last_collinear_wk = list(out_wk.collinearity_warnings)
        self._last_collinear_kt = list(out_kt.collinearity_warnings)
        self._last_bound_warnings_wk = list(out_wk.bound_warnings)
        self._last_bound_warnings_kt = list(out_kt.bound_warnings)

        if (
            accepted_any
            and recent_columns is not None
            and len(recent_columns.get("ts_utc", ())) > 0
        ):
            self.replay_48h_from_columns(recent_columns)
        else:
            self._refresh_cached_snapshots()

        self.calibration_history.append(dict(plan["history_entry"]))
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

    def run_calibration(
        self,
        columns: Mapping[str, Sequence[float | int]],
        mode: Literal["manual", "auto_weekly"] = "manual",
    ) -> dict[str, Any]:
        """Runs synchronous 3-phase batch calibration (`ctx -> plan -> apply`).

        Args:
            columns: Full RingBuffer telemetry column dictionary.
            mode: `"manual"` or `"auto_weekly"`.

        Returns:
            Summary dictionary with `accepted_any`, `accepted_wk`,
            `accepted_kt`, `outcome_wk`, `outcome_kt`, `cop_params`, and
            `snapshot_depth`.
        """
        ctx = self.capture_calibration_context()
        plan = self.compute_calibration_plan(columns, ctx, mode=mode)
        return self.apply_calibration_plan(plan, columns)

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
        self._param_epoch += 1

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

        self._params_wk = priors.DAY1_PRIORS_WK
        self._params_kt = priors.DAY1_PRIORS_KT
        self._k_hx_by_group_w_k = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
        self.cop_params = priors.DAY1_COP_PARAMS
        self.memory_wk = None
        self.memory_kt = None
        self.memory_cop = None
        self._last_posterior_std_wk = {}
        self._last_posterior_std_kt = {}
        self._last_corr_wk = {}
        self._last_corr_kt = {}
        self._last_collinear_wk = []
        self._last_collinear_kt = []
        self._last_bound_warnings_wk = []
        self._last_bound_warnings_kt = []

        self.estimator_wk.params = self._params_wk
        self.estimator_kt.params = self._params_kt
        self.hydraulics.update_k_hx(self._k_hx_by_group_w_k)
        self._param_epoch += 1

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

    def compute_14d_evaluation_reports(
        self,
        columns: Mapping[str, Sequence[float | int]],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[contracts.ZoneEvaluationReport, contracts.ZoneEvaluationReport]:
        """Computes 14d `ZoneEvaluationReport`s for `wk` & `kt` purely.

        Args:
            columns: RingBuffer telemetry column dictionary.
            context: Optional decoupled snapshot dictionary from
                `capture_calibration_context()`. When omitted, uses active
                engine parameters and the full `k_hx_by_group_w_k` mapping.

        Returns:
            Tuple `(eval_report_wk, eval_report_kt)`.
        """
        if context is not None:
            p_wk: contracts.ZoneParameters = context["params_wk"]
            p_kt: contracts.ZoneParameters = context["params_kt"]
            flows: Mapping[str, float] = context["flow_rates_l_min"]
            k_hx_map: Mapping[str, float] = context["k_hx_by_group_w_k"]
            lat = float(context.get("latitude_deg", self.latitude_deg))
            lon = float(context.get("longitude_deg", self.longitude_deg))
        else:
            p_wk = self._params_wk
            p_kt = self._params_kt
            flows = self._flow_rates_l_min
            k_hx_map = self._k_hx_by_group_w_k
            lat = self.latitude_deg
            lon = self.longitude_deg

        return _evaluate_both_zones_14d(
            columns,
            p_wk,
            p_kt,
            flows,
            k_hx_map,
            lat,
            lon,
        )

    def apply_14d_evaluation_reports(
        self,
        rep_wk: contracts.ZoneEvaluationReport,
        rep_kt: contracts.ZoneEvaluationReport,
    ) -> None:
        """Applies newly computed 14-day evaluation reports and refreshes cache.

        Args:
            rep_wk: Evaluated 14-day hindcast report for `wk`.
            rep_kt: Evaluated 14-day hindcast report for `kt`.
        """
        self.eval_report_wk = rep_wk
        self.eval_report_kt = rep_kt
        self._refresh_cached_snapshots()

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
        rep_wk, rep_kt = self.compute_14d_evaluation_reports(columns)
        self.apply_14d_evaluation_reports(rep_wk, rep_kt)
        return {
            "wk": rep_wk,
            "kt": rep_kt,
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
                    "parameters_active": self._params_wk.to_dict(),
                    "posterior_std": dict(self._last_posterior_std_wk),
                    "correlation_matrix": dict(self._last_corr_wk),
                    "collinearity_warnings": list(self._last_collinear_wk),
                    "bound_warnings": list(self._last_bound_warnings_wk),
                    "parameters_previous_snapshot": (
                        prev_snap.get("params_wk") if prev_snap else None
                    ),
                    "parameters_day1_prior": priors.DAY1_PRIORS_WK.to_dict(),
                    "evaluation_14d": evaluation.zone_eval_report_to_dict(
                        self.eval_report_wk
                    ),
                },
                "kt": {
                    "parameters_active": self._params_kt.to_dict(),
                    "posterior_std": dict(self._last_posterior_std_kt),
                    "correlation_matrix": dict(self._last_corr_kt),
                    "collinearity_warnings": list(self._last_collinear_kt),
                    "bound_warnings": list(self._last_bound_warnings_kt),
                    "parameters_previous_snapshot": (
                        prev_snap.get("params_kt") if prev_snap else None
                    ),
                    "parameters_day1_prior": priors.DAY1_PRIORS_KT.to_dict(),
                    "evaluation_14d": evaluation.zone_eval_report_to_dict(
                        self.eval_report_kt
                    ),
                },
            },
            "cop_model": {
                "parameters_active": self.cop_params.to_dict(),
                "parameters_day1_prior": priors.DAY1_COP_PARAMS.to_dict(),
                "diagnostics": dict(self.cop_diagnostics),
            },
            "snapshot_stack_depth": len(self.snapshot_stack),
            "calibration_history": list(self.calibration_history),
        }

        return reporting.export_csv_and_report(
            export_dir,
            columns,
            fit_report,
            lat_deg=self.latitude_deg,
            lon_deg=self.longitude_deg,
        )

    # ========================================================================
    # Layer-2 Pure-Python Prediction, Control & Solver API
    # ========================================================================
    def get_cop_parameters(self) -> contracts.CopParameters:
        """Returns the active calibrated heat-pump `CopParameters`."""
        return self.cop_params

    def predict_live_expected_cop(self, raw_row: Mapping[str, Any]) -> float:
        """Predicts authoritative live expected space-heating COP from `raw_row`.

        Args:
            raw_row: Latest raw 5-minute telemetry row mapping.

        Returns:
            Expected space-heating COP (`>= 1.0`) with DHW/restwarmte guard.
        """
        return cop_model.predict_live_expected_cop_from_row(
            self.cop_params, raw_row, self.cop_diagnostics
        )

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
                "bound_warnings": list(self._last_bound_warnings_wk),
            }
        return {
            "posterior_std": dict(self._last_posterior_std_kt),
            "correlation_matrix": dict(self._last_corr_kt),
            "collinearity_warnings": list(self._last_collinear_kt),
            "bound_warnings": list(self._last_bound_warnings_kt),
        }

    def get_zone_fit_summary(
        self,
        zone: contracts.ZoneId,
    ) -> dict[str, Any]:
        """Returns a complete JSON-safe diagnostic fit summary for `zone`.

        Packages active/Day-1 parameters, 14-day hindcast metrics, Fisher
        uncertainties, warnings, and the latest calibration record so diagnostic
        sensors do not poke private engine attributes directly.

        Args:
            zone: Target zone identifier (`"wk"` or `"kt"`).

        Returns:
            Dictionary containing `rmse_6h_c`, `parameters_active`,
            `parameters_day1_prior`, `params`, `eval_report`, `overall_6h`,
            `regime_rmses`, `fisher_1sigma_std`, `posterior_std`,
            `correlation_matrix`, `collinearity_warnings`, `bound_warnings`,
            `last_calibration`, and `recent_calibration_history`.
        """
        snap = self.get_zone_snapshot(zone)
        report = self.eval_report_wk if zone == "wk" else self.eval_report_kt
        day1_priors = (
            priors.DAY1_PRIORS_WK if zone == "wk" else priors.DAY1_PRIORS_KT
        )

        rmse_val: float | None = None
        overall_dict: dict[str, Any] = {}
        regime_dict: dict[str, Any] = {}
        if report is not None:
            rep_d = reporting.sanitize_json_payload(
                evaluation.zone_eval_report_to_dict(report)
            )
            if isinstance(rep_d, Mapping):
                overall_dict = dict(rep_d.get("overall_6h") or {})
                rmse_raw = overall_dict.get("rmse_end_c")
                if preprocessing.is_finite_number(rmse_raw):
                    rmse_val = float(rmse_raw)
                by_reg = rep_d.get("by_regime") or {}
                if isinstance(by_reg, Mapping):
                    regime_dict = {
                        str(k): dict(v)
                        for k, v in by_reg.items()
                        if isinstance(v, Mapping)
                    }
        if rmse_val is None and preprocessing.is_finite_number(snap.rmse_6h_c):
            rmse_val = float(snap.rmse_6h_c)

        fit_diag = reporting.sanitize_json_payload(
            self.get_zone_fit_diagnostics(zone)
        )
        fisher_std = dict(fit_diag.get("posterior_std") or {})
        corr_matrix = dict(fit_diag.get("correlation_matrix") or {})
        collinear_warns = list(fit_diag.get("collinearity_warnings") or [])
        bound_warns = list(fit_diag.get("bound_warnings") or [])

        last_cal_summary: dict[str, Any] = {}
        acc_key = f"accepted_{zone}"
        reason_key = f"reason_{zone}"
        jb_key = f"holdout_j_{zone}_before"
        ja_key = f"holdout_j_{zone}_after"
        for item in reversed(self.calibration_history):
            d_item = reporting.sanitize_json_payload(item)
            if not isinstance(d_item, Mapping):
                continue
            if acc_key in d_item:
                ev_str = str(d_item.get("event") or "")
                mode_val = d_item.get("mode") or (
                    ev_str.removeprefix("calibration_")
                    if ev_str.startswith("calibration_")
                    else ev_str
                )
                last_cal_summary = {
                    "accepted": d_item.get(acc_key),
                    "reason": d_item.get(reason_key),
                    "mode": mode_val,
                    "ts_utc": d_item.get("ts_utc"),
                    "holdout_j_before": d_item.get(jb_key),
                    "holdout_j_after": d_item.get(ja_key),
                }
                break
            if d_item.get("zone") == zone:
                if not fisher_std:
                    fisher_std = dict(d_item.get("posterior_std") or {})
                if not corr_matrix:
                    corr_matrix = dict(d_item.get("correlation_matrix") or {})
                if not collinear_warns:
                    collinear_warns = list(
                        d_item.get("collinearity_warnings") or []
                    )
                if not bound_warns:
                    bound_warns = list(d_item.get("bound_warnings") or [])
                last_cal_summary = {
                    "accepted": d_item.get("accepted"),
                    "reason": d_item.get("reason"),
                    "mode": d_item.get("mode"),
                    "ts_utc": d_item.get("ts_utc"),
                }
                break

        return {
            "rmse_6h_c": rmse_val,
            "params": snap.params,
            "parameters_active": reporting.sanitize_json_payload(
                snap.params.to_dict()
            ),
            "parameters_day1_prior": reporting.sanitize_json_payload(
                day1_priors.to_dict()
            ),
            "eval_report": report,
            "overall_6h": overall_dict,
            "regime_rmses": regime_dict,
            "fisher_1sigma_std": fisher_std,
            "posterior_std": fisher_std,
            "correlation_matrix": corr_matrix,
            "collinearity_warnings": collinear_warns,
            "bound_warnings": bound_warns,
            "last_calibration": last_cal_summary,
            "recent_calibration_history": reporting.sanitize_json_payload(
                list(self.calibration_history[-8:])
            ),
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
                    * self.hydraulics.gamma_cal
                    * max(
                        0.0, float(actuator.t_supply_sec_c) - st_in.t_floor_c
                    )
                )
            if zone == "kt" and actuator.p_el_ac_w > 25.0:
                q_ac = solar_weather.compute_ac_heating_power_w(
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
        t_surf = zoh_kalman.estimate_zone_surface_temp(
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
            t_surf_now = zoh_kalman.estimate_zone_surface_temp(
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
        return zoh_kalman.export_2r2c_solver_model(zone, params, dt_s=dt_s)

    def update_flow_rates(self, flow_rates_l_min: Mapping[str, float]) -> None:
        """Updates manifold rotameter flow rates [L/min] from HA OptionsFlow."""
        self._flow_rates_l_min.update(
            {str(k): float(v) for k, v in flow_rates_l_min.items()}
        )
        self.hydraulics.update_flow_rates(self._flow_rates_l_min)
        self._k_hx_by_group_w_k = dict(self.hydraulics.k_hx_by_group_w_k)
        self._params_wk = dataclasses.replace(
            self._params_wk, k_hx_w_k=self._k_hx_by_group_w_k["wk"]
        )
        self._params_kt = dataclasses.replace(
            self._params_kt, k_hx_w_k=self._k_hx_by_group_w_k["kantoor"]
        )
        self.estimator_wk.params = self._params_wk
        self.estimator_kt.params = self._params_kt
        self._param_epoch += 1
        self._refresh_cached_snapshots()

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
            "params_wk": self._params_wk.to_dict(),
            "params_kt": self._params_kt.to_dict(),
            "estimator_wk": self.estimator_wk.to_dict(),
            "estimator_kt": self.estimator_kt.to_dict(),
            "hydraulics": self.hydraulics.to_dict(),
            "cop_params": self.cop_params.to_dict(),
            "hindcast_wk": self.hindcast_wk.to_dict(),
            "hindcast_kt": self.hindcast_kt.to_dict(),
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
            "bound_warnings_wk": list(self._last_bound_warnings_wk),
            "bound_warnings_kt": list(self._last_bound_warnings_kt),
        }
        sanitized: dict[str, Any] = reporting.sanitize_json_payload(raw_state)
        sanitized["episode_bank"] = self.episode_bank.to_dict()
        return sanitized

    def load_state_dict(self, data: Mapping[str, Any]) -> None:
        """Restores complete engine state from `betonbatterij.state` dict."""
        if not data:
            return
        ts_raw = data.get("last_ts_utc")
        if preprocessing.is_finite_number(ts_raw):
            self._last_ts_utc = int(ts_raw)
        if "hm_ema_c" in data:
            hm_raw = data["hm_ema_c"]
            self._hm_ema_c = (
                float(hm_raw)
                if preprocessing.is_finite_number(hm_raw)
                else None
            )

        if isinstance(data.get("interfloor_cache"), Mapping):
            self._interfloor_cache = {}
            for k, pair in data["interfloor_cache"].items():
                if (
                    isinstance(pair, Sequence)
                    and len(pair) == 2
                    and preprocessing.is_finite_number(pair[0])
                    and preprocessing.is_finite_number(pair[1])
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
                    if preprocessing.is_finite_number(v)
                }
            )
        if isinstance(data.get("k_hx_by_group_w_k"), Mapping):
            self._k_hx_by_group_w_k.update(
                {
                    str(k): float(v)
                    for k, v in data["k_hx_by_group_w_k"].items()
                    if preprocessing.is_finite_number(v)
                }
            )

        if isinstance(data.get("params_wk"), Mapping):
            self._params_wk = contracts.ZoneParameters.from_dict(
                data["params_wk"], defaults=priors.DAY1_PRIORS_WK
            )
        if isinstance(data.get("params_kt"), Mapping):
            self._params_kt = contracts.ZoneParameters.from_dict(
                data["params_kt"], defaults=priors.DAY1_PRIORS_KT
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
            self.cop_params = contracts.CopParameters.from_dict(
                data.get("cop_params"), defaults=priors.DAY1_COP_PARAMS
            )

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

        self._last_q_fh_wk_w = preprocessing.safe_float(
            data.get("last_q_fh_wk_w"), 0.0
        )
        self._last_q_fh_kt_w = preprocessing.safe_float(
            data.get("last_q_fh_kt_w"), 0.0
        )
        m_wk = data.get("last_t_air_meas_wk_c")
        m_kt = data.get("last_t_air_meas_kt_c")
        self._last_t_air_meas_wk_c = (
            float(m_wk) if preprocessing.is_finite_number(m_wk) else None
        )
        self._last_t_air_meas_kt_c = (
            float(m_kt) if preprocessing.is_finite_number(m_kt) else None
        )

        if isinstance(data.get("posterior_std_wk"), Mapping):
            self._last_posterior_std_wk = {
                str(k): float(v)
                for k, v in data["posterior_std_wk"].items()
                if preprocessing.is_finite_number(v)
            }
        if isinstance(data.get("posterior_std_kt"), Mapping):
            self._last_posterior_std_kt = {
                str(k): float(v)
                for k, v in data["posterior_std_kt"].items()
                if preprocessing.is_finite_number(v)
            }
        if isinstance(data.get("corr_wk"), Mapping):
            self._last_corr_wk = {
                str(k): {
                    str(m): float(val)
                    for m, val in row.items()
                    if preprocessing.is_finite_number(val)
                }
                for k, row in data["corr_wk"].items()
                if isinstance(row, Mapping)
            }
        if isinstance(data.get("corr_kt"), Mapping):
            self._last_corr_kt = {
                str(k): {
                    str(m): float(val)
                    for m, val in row.items()
                    if preprocessing.is_finite_number(val)
                }
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
        if isinstance(data.get("bound_warnings_wk"), Sequence):
            self._last_bound_warnings_wk = [
                str(x) for x in data["bound_warnings_wk"]
            ]
        if isinstance(data.get("bound_warnings_kt"), Sequence):
            self._last_bound_warnings_kt = [
                str(x) for x in data["bound_warnings_kt"]
            ]

        self._param_epoch += 1
        self._refresh_cached_snapshots()
