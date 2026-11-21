"""Open-loop 6h/1h hindcast evaluation, live FIFO tracker, and CSV/JSON export.

Implements Component 4 (`04-evaluatie-hindcast-en-telemetrie.md`) in pure Python
3.13 standard library with zero external or `homeassistant.*` dependencies.
"""
from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import csv
import dataclasses
import datetime
import json
import math
import os
import pathlib
from typing import Any

from custom_components.betonbatterij import const
from custom_components.betonbatterij import house_config
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import hydraulics
from custom_components.betonbatterij.domain import solar_weather
from custom_components.betonbatterij.domain import zoh_kalman

# Evaluation constants (Component 4 §2–§4)
SPINUP_48H_STEPS: int = 576  # 48h * 12 steps/h
EVAL_14D_STEPS: int = 14 * 288  # 4032 steps
MAX_EVAL_SLICE_STEPS: int = SPINUP_48H_STEPS + EVAL_14D_STEPS  # 4608 steps (16d)
HORIZON_6H_STEPS: int = 72  # 6.0 hours
HORIZON_AC_1H_STEPS: int = 12  # 1.0 hour
MIN_CLEAN_6H_STEPS: int = 48  # >= 4.0 hours clean before mid-window truncation
STRIDE_6H_STEPS: int = 6  # 30 minutes
STRIDE_AC_1H_STEPS: int = 2  # 10 minutes
COOLING_TAIL_STEPS: int = 24  # 2.0 hours post-cooling exclusion

CSV_EXPORT_COLUMNS: tuple[str, ...] = (
    "ts_utc",
    "iso_utc",
    "flags",
    "t_out_c",
    "v_wind_ms",
    "i_sol_wk_wm2",
    "i_sol_kt_wm2",
    "tau_screen_wk",
    "t_buf_top_c",
    "t_buf_bot_c",
    "q_sec_tot_w",
    "u_eff_wk",
    "f_flow_wk",
    "t_air_wk_c",
    "t_floor_wk_c",
    "d_est_wk_w",
    "q_fh_wk_w",
    "t_hind6h_wk_c",
    "u_eff_kt",
    "f_flow_kt",
    "t_air_kt_c",
    "t_down_kt_c",
    "q_int_kt_w",
    "p_el_ac_kt_w",
    "t_floor_kt_c",
    "d_est_kt_w",
    "t_hind6h_kt_c",
    "t_surf_wk_c",
    "t_surf_kt_c",
    "p_wp_raw_el_w",
    "f_comp_hz",
)


def _opt_float(val: float | int | None, min_valid: float = -100.0) -> float | None:
    """Returns `float(val)` if finite and `>= min_valid`, else `None`."""
    if val is None:
        return None
    fval = float(val)
    if not math.isfinite(fval) or fval < min_valid:
        return None
    return fval


def _safe_float(val: float | int | None, default: float = 0.0) -> float:
    """Returns a finite float for `val`, falling back to `default`."""
    if val is None:
        return default
    fval = float(val)
    if not math.isfinite(fval):
        return default
    return fval


def estimate_zone_surface_temp(
    zone: contracts.ZoneId,
    t_air_c: float,
    t_floor_c: float,
    params: contracts.ZoneParameters,
) -> float:
    """Estimates top-of-PVC floor surface temperature for `zone`.

    Args:
        zone: Zone identifier (`"wk"` or `"kt"`).
        t_air_c: Room air temperature [°C].
        t_floor_c: Concrete slab core temperature [°C].
        params: Physical zone parameters providing `r_fh_k_kw`.

    Returns:
        Estimated PVC surface temperature [°C].
    """
    area_m2 = house_config.AREA_WK_M2 if zone == "wk" else house_config.AREA_KT_M2
    return solar_weather.estimate_surface_temp(
        t_air_c=t_air_c,
        t_floor_c=t_floor_c,
        r_fh_k_kw=params.r_fh_k_kw,
        area_m2=area_m2,
        h_surf_w_m2k=house_config.H_SURF_UPWARD_W_M2K,
    )


def compute_ac_heating_power_w(
    p_el_ac_w: float,
    ac_mode_code: float,
    ac_defrost: bool,
    t_out_c: float,
    eta_ac: float,
    base_q_ac_w: float = 0.0,
) -> float:
    """Computes candidate-parameter split-AC thermal power `Q_ac,kt` [W].

    Args:
        p_el_ac_w: Electrical split-AC power [W].
        ac_mode_code: Numeric AC mode (`1.0` = heat, `2.0` = cool, `3.0` = dry).
        ac_defrost: True when AC defrost extraction is active.
        t_out_c: Outdoor temperature [°C].
        eta_ac: Candidate net split-AC thermal efficiency at 7 °C [-].
        base_q_ac_w: Baseline AC load for non-heating modes [W].

    Returns:
        Net convective thermal power delivered to `kt` air node [W].
    """
    if ac_defrost:
        if base_q_ac_w < 0.0:
            return base_q_ac_w
        return -min(200.0, 50.0 + 0.15 * max(0.0, p_el_ac_w - 3.0))
    if (
        math.isfinite(ac_mode_code)
        and round(ac_mode_code) == int(const.AC_MODE_HEAT)
        and p_el_ac_w > 25.0
    ):
        temp_factor = min(1.35, max(0.60, 1.0 + 0.025 * (t_out_c - 7.0)))
        return max(0.0, p_el_ac_w - 3.0) * eta_ac * temp_factor
    return base_q_ac_w


@dataclasses.dataclass(slots=True)
class PreprocessedHistory:
    """Precomputed exogenous, hydraulic, and flag series over a column slice.

    Attributes:
        n_rows: Number of 5-minute rows in the slice.
        ts_utc: Unix UTC timestamps at window ends [s].
        flags: Bitmask status flags per step.
        dt_s: Step duration [s].
        t_out_c: Selected outdoor temperature [°C].
        v_wind_ms: Wind speed [m/s].
        i_sol_wk_wm2: Projected solar irradiance on `wk` glazing + screens [W/m²].
        i_sol_kt_wm2: Projected solar irradiance on `kt` roof [W/m²].
        tau_screen_wk: Effective solar screen transmittance for `wk` [-].
        pv_fallback: Per-step flag indicating Forecast.Solar fallback usage.
        t_up_wk_c: Soft 1st-floor ceiling boundary temperature above `wk` [°C].
        t_down_wk_c: Ground temperature boundary below `wk` [°C].
        t_down_kt_c: Soft 1st-floor boundary temperature below `kt` [°C].
        q_int_wk_w: Known internal heat gain in `wk` (baseload + 1e leak) [W].
        q_int_kt_w: Known internal heat gain in `kt` (growlights + desk/PIR) [W].
        q_ac_base_kt_w: Baseline AC power for non-heating modes in `kt` [W].
        p_el_ac_kt_w: Electrical split-AC power in `kt` [W].
        ac_mode_code: Numeric AC mode code per step.
        p_desk_kt_w: Measured desk plug power in `kt` [W].
        pir_occ_kt: Occupancy radar duty in `kt` [0..1].
        u_pump_sec: Secondary circulation pump duty [0..1].
        u_eff_wk: Effective BG valve opening [0..1].
        u_eff_kt: Effective `kantoor` valve opening [0..1].
        t_buf_top_c: Top 200 L buffer temperature at window end [°C].
        t_buf_bot_cal_c: Calibrated bottom 200 L buffer temperature [°C].
        t_sup_sec_c: Effective secondary manifold supply temperature [°C].
        f_comp_hz: Heat pump compressor frequency [Hz].
        p_prim_th_w: Primary heat pump thermal power [W].
        q_fh_wk_hyd_w: Hydraulic floor heating power in `wk` [W].
        q_fh_kt_hyd_w: Hydraulic floor heating power in `kt` [W].
        q_sec_tot_w: Total secondary floor heating power across all groups [W].
        f_flow_wk: Flow share of `wk` [-].
        f_flow_kt: Flow share of `kt` [-].
        t_air_wk_c: Measured air temperature in `wk` [°C].
        t_air_kt_c: Measured air temperature in `kt` [°C].
        t_surf_meas_wk_c: Optional PVC surface sensor reading in `wk` [°C].
        d_est_col_wk_w: Stored online disturbance column for `wk` [W].
        d_est_col_kt_w: Stored online disturbance column for `kt` [W].
        cooling_or_tail: Mask covering active cooling plus 2.0h post-cooling tail.
        hyd_k_hx_wk: Reference `K_hx` used for `wk` in hydraulic pass [W/K].
        hyd_k_hx_kt: Reference `K_hx` used for `kt` in hydraulic pass [W/K].
    """

    n_rows: int
    ts_utc: list[int]
    flags: list[int]
    dt_s: list[float]
    t_out_c: list[float]
    v_wind_ms: list[float]
    i_sol_wk_wm2: list[float]
    i_sol_kt_wm2: list[float]
    tau_screen_wk: list[float]
    pv_fallback: list[bool]
    t_up_wk_c: list[float]
    t_down_wk_c: list[float]
    t_down_kt_c: list[float]
    q_int_wk_w: list[float]
    q_int_kt_w: list[float]
    q_ac_base_kt_w: list[float]
    p_el_ac_kt_w: list[float]
    ac_mode_code: list[float]
    p_desk_kt_w: list[float]
    pir_occ_kt: list[float]
    u_pump_sec: list[float]
    u_eff_wk: list[float]
    u_eff_kt: list[float]
    t_buf_top_c: list[float]
    t_buf_bot_cal_c: list[float]
    t_sup_sec_c: list[float]
    f_comp_hz: list[float]
    p_prim_th_w: list[float]
    q_fh_wk_hyd_w: list[float]
    q_fh_kt_hyd_w: list[float]
    q_sec_tot_w: list[float]
    f_flow_wk: list[float]
    f_flow_kt: list[float]
    t_air_wk_c: list[float]
    t_air_kt_c: list[float]
    t_surf_meas_wk_c: list[float | None]
    d_est_col_wk_w: list[float]
    d_est_col_kt_w: list[float]
    cooling_or_tail: list[bool]
    hyd_k_hx_wk: float
    hyd_k_hx_kt: float


def slice_columns(
    columns: Mapping[str, Sequence[float | int]],
    start: int,
    end: int,
) -> dict[str, list[float | int]]:
    """Returns a `[start:end]` list slice across all columns in `columns`."""
    return {key: list(seq[start:end]) for key, seq in columns.items()}


def _col_at(
    columns: Mapping[str, Sequence[float | int]],
    key: str,
    idx: int,
    default: float = 0.0,
    alt_key: str | None = None,
) -> float:
    """Reads a finite float at `idx` from `columns[key]` (or `alt_key`)."""
    seq = columns.get(key)
    if seq is None and alt_key is not None:
        seq = columns.get(alt_key)
    if seq is None or idx >= len(seq):
        return default
    return _safe_float(seq[idx], default)


def preprocess_history_columns(
    columns: Mapping[str, Sequence[float | int]],
    flow_rates_l_min: Mapping[str, float] | None = None,
    k_hx_by_group: Mapping[str, float] | None = None,
    eta_ac_ref: float | None = None,
    lat_deg: float = house_config.LATITUDE_DEG,
    lon_deg: float = house_config.LONGITUDE_DEG,
) -> PreprocessedHistory:
    """Preprocesses raw RingBuffer columns into step-aligned physical series.

    Args:
        columns: Mapping of RingBuffer column names to equal-length sequences.
        flow_rates_l_min: Optional manifold rotameter flow rates [L/min].
        k_hx_by_group: Optional `K_hx` [W/K] overrides per hydraulic group.
        eta_ac_ref: Optional reference split-AC efficiency for `kt`.

    Returns:
        Populated `PreprocessedHistory` over all rows in `columns`.
    """
    ts_seq = columns.get("ts_utc", ())
    n = len(ts_seq)
    flows = dict(const.DEFAULT_FLOW_RATES_L_MIN)
    if flow_rates_l_min is not None:
        flows.update({k: float(v) for k, v in flow_rates_l_min.items()})

    k_hx_map = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
    if k_hx_by_group is not None:
        for group_name, val in k_hx_by_group.items():
            target_key = "kantoor" if group_name == "kt" else group_name
            k_hx_map[target_key] = float(val)

    hyd = hydraulics.HydraulicModel(
        flow_rates_l_min=flows, k_hx_by_group_w_k=k_hx_map
    )
    eta_ref = (
        house_config.DAY1_PRIORS_KT.eta_ac if eta_ac_ref is None else eta_ac_ref
    )

    ts_utc: list[int] = [0] * n
    flags: list[int] = [0] * n
    dt_s: list[float] = [float(const.DT_STEP_S)] * n
    t_out_c: list[float] = [0.0] * n
    v_wind_ms: list[float] = [3.0] * n
    i_sol_wk_wm2: list[float] = [0.0] * n
    i_sol_kt_wm2: list[float] = [0.0] * n
    tau_screen_wk: list[float] = [1.0] * n
    pv_fallback: list[bool] = [False] * n
    t_up_wk_c: list[float] = [18.0] * n
    t_down_wk_c: list[float] = [house_config.T_GROUND_MEAN_C] * n
    t_down_kt_c: list[float] = [18.0] * n
    q_int_wk_w: list[float] = [house_config.BASELOAD_WK_W] * n
    q_int_kt_w: list[float] = [0.0] * n
    q_ac_base_kt_w: list[float] = [0.0] * n
    p_el_ac_kt_w: list[float] = [0.0] * n
    ac_mode_code: list[float] = [0.0] * n
    p_desk_kt_w: list[float] = [0.0] * n
    pir_occ_kt: list[float] = [0.0] * n
    u_pump_sec: list[float] = [0.0] * n
    u_eff_wk: list[float] = [0.0] * n
    u_eff_kt: list[float] = [0.0] * n
    t_buf_top_c: list[float] = [20.0] * n
    t_buf_bot_cal_c: list[float] = [20.0] * n
    t_sup_sec_c: list[float] = [20.0] * n
    f_comp_hz: list[float] = [0.0] * n
    p_prim_th_w: list[float] = [0.0] * n
    q_fh_wk_hyd_w: list[float] = [0.0] * n
    q_fh_kt_hyd_w: list[float] = [0.0] * n
    q_sec_tot_w: list[float] = [0.0] * n
    f_flow_wk: list[float] = [0.0] * n
    f_flow_kt: list[float] = [0.0] * n
    t_air_wk_c: list[float] = [19.5] * n
    t_air_kt_c: list[float] = [19.0] * n
    t_surf_meas_wk_c: list[float | None] = [None] * n
    d_est_col_wk_w: list[float] = [0.0] * n
    d_est_col_kt_w: list[float] = [0.0] * n
    cooling_raw: list[bool] = [False] * n

    hm_ema_c: float | None = None
    interfloor_cache: dict[str, tuple[float, int]] = {}
    desk_state: dict[str, Any] = {}
    t_floor_wk_running: float = _col_at(columns, "t_air_wk_c", 0, 19.5) + 0.8
    t_floor_kt_running: float = (
        _col_at(columns, "t_air_kantoor_c", 0, 19.0, "t_air_kt_c") + 0.5
    )

    flags_seq = columns.get("flags")
    surf_seq = columns.get("t_floor_surf_meas_wk_c")
    bg_sup_seq = columns.get("t_bg_sup_c")
    bg_ret_seq = columns.get("t_bg_ret_c")
    desk_seq = columns.get("p_desk_kt_w")
    default_flags = (
        const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
    )

    for k in range(n):
        ts_k = int(ts_seq[k])
        ts_utc[k] = ts_k
        fl_k = (
            int(flags_seq[k])
            if flags_seq is not None and k < len(flags_seq)
            else default_flags
        )
        flags[k] = fl_k

        if k > 0:
            diff = float(ts_k - ts_utc[k - 1])
            dt_s[k] = diff if 1.0 <= diff <= 86400.0 else float(const.DT_STEP_S)

        t_dak = (
            _opt_float(columns["t_out_dak_c"][k])
            if "t_out_dak_c" in columns and k < len(columns["t_out_dak_c"])
            else None
        )
        t_wea = (
            _opt_float(columns["t_out_weather_c"][k])
            if "t_out_weather_c" in columns and k < len(columns["t_out_weather_c"])
            else None
        )
        t_hm = (
            _opt_float(columns["t_out_hm_c"][k])
            if "t_out_hm_c" in columns and k < len(columns["t_out_hm_c"])
            else None
        )
        if t_dak is None and t_wea is None and t_hm is None and "t_out_c" in columns:
            t_dak = _opt_float(columns["t_out_c"][k])
        if t_hm is not None:
            hm_ema_c = (
                t_hm if hm_ema_c is None else (t_hm / 6.0) + (5.0 / 6.0) * hm_ema_c
            )
        t_out_k = solar_weather.select_outdoor_temp(
            ts_utc=ts_k,
            t_dak_c=t_dak,
            t_weather_c=t_wea,
            t_hm_ema_c=hm_ema_c,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )
        t_out_c[k] = t_out_k
        v_wind_ms[k] = max(0.0, _col_at(columns, "v_wind_ms", k, 3.0))

        pv_fb_flag = bool(fl_k & const.FLAG_PV_FALLBACK)
        if "p_pv_w" in columns and k < len(columns["p_pv_w"]):
            raw_pv = columns["p_pv_w"][k]
            p_pv_k = math.nan if raw_pv is None else float(raw_pv)
        else:
            p_pv_k = 0.0
        p_pv_fc_k = _col_at(columns, "p_pv_forecast_w", k, 0.0)
        epex_k = _col_at(columns, "epex_eur_kwh", k, 0.20)
        scr_s = _col_at(columns, "screen_small_pos", k, 100.0)
        scr_l = _col_at(columns, "screen_large_pos", k, 100.0)

        if (
            "i_sol_wk_wm2" in columns
            and "i_sol_kt_wm2" in columns
            and "p_pv_w" not in columns
        ):
            i_wk = _col_at(columns, "i_sol_wk_wm2", k, 0.0)
            i_kt = _col_at(columns, "i_sol_kt_wm2", k, 0.0)
            tau_scr = _col_at(columns, "tau_screen_wk", k, 1.0)
            pv_fb = pv_fb_flag
        else:
            i_wk, i_kt, tau_scr, pv_fb = solar_weather.project_solar_irradiance(
                ts_utc=ts_k,
                p_pv_w=p_pv_k,
                p_pv_forecast_w=p_pv_fc_k,
                epex_eur_kwh=epex_k,
                screen_small_pos=scr_s,
                screen_large_pos=scr_l,
                pv_stale=pv_fb_flag,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
            )
        i_sol_wk_wm2[k] = max(0.0, i_wk)
        i_sol_kt_wm2[k] = max(0.0, i_kt)
        tau_screen_wk[k] = tau_scr
        pv_fallback[k] = pv_fb or pv_fb_flag
        if pv_fallback[k]:
            flags[k] |= const.FLAG_PV_FALLBACK

        t_wk = _col_at(columns, "t_air_wk_c", k, 19.5)
        t_kt = _col_at(columns, "t_air_kantoor_c", k, 19.0, "t_air_kt_c")
        t_air_wk_c[k] = t_wk
        t_air_kt_c[k] = t_kt
        room_temps: dict[str, float | None] = {
            "wk": t_wk,
            "gang": t_wk - 0.8,
            "slaapkamer": _col_at(columns, "t_air_slaapkamer_c", k, 18.0),
            "badkamer": _col_at(columns, "t_air_badkamer_c", k, 18.5),
            "werkkamer_1e": _col_at(columns, "t_air_werkkamer_1e_c", k, 18.0),
            "kamer_1e_midden": _col_at(
                columns, "t_air_kamer_1e_midden_c", k, 18.0
            ),
            "kamer_1e_zuid": _col_at(
                columns, "t_air_kamer_1e_zuid_c", k, 18.0
            ),
            "kantoor": t_kt,
            "badkamer_boven": _col_at(columns, "t_air_badkamer_boven_c", k, 18.0),
            "logeerkamer": _col_at(columns, "t_air_logeerkamer_c", k, 17.5),
        }

        u_bg = min(1.0, max(0.0, _col_at(columns, "u_eff_bg", k, 0.0, "u_eff_wk")))
        u_kt_v = min(
            1.0, max(0.0, _col_at(columns, "u_eff_kantoor", k, 0.0, "u_eff_kt"))
        )
        u_pump = min(1.0, max(0.0, _col_at(columns, "u_pump_sec", k, 0.0)))
        u_pump_sec[k] = u_pump
        u_eff_wk[k] = u_bg
        u_eff_kt[k] = u_kt_v

        u_valves: dict[str, float] = {
            "wk": u_bg,
            "gang": u_bg,
            "slaapkamer": min(
                1.0, max(0.0, _col_at(columns, "u_eff_slaapkamer", k, 0.0))
            ),
            "badkamer": min(
                1.0, max(0.0, _col_at(columns, "u_eff_badkamer", k, 0.0))
            ),
            "werkkamer_1e": min(
                1.0, max(0.0, _col_at(columns, "u_eff_werkkamer_1e", k, 0.0))
            ),
            "kamer_1e_midden": min(
                1.0, max(0.0, _col_at(columns, "u_eff_kamer_1e_midden", k, 0.0))
            ),
            "kamer_1e_zuid": min(
                1.0, max(0.0, _col_at(columns, "u_eff_kamer_1e_zuid", k, 0.0))
            ),
            "kantoor": u_kt_v,
            "badkamer_boven": min(
                1.0, max(0.0, _col_at(columns, "u_eff_badkamer_boven", k, 0.0))
            ),
            "logeerkamer": min(
                1.0, max(0.0, _col_at(columns, "u_eff_logeerkamer", k, 0.0))
            ),
        }

        threeway_room = (fl_k & const.FLAG_DHW_TANK) == 0
        wp_defrost = bool(fl_k & const.FLAG_DEFROST_WP)
        v_prim = max(0.0, _col_at(columns, "v_prim_l_min", k, 0.0))
        p_prim = _col_at(columns, "p_prim_th_w", k, 0.0)
        f_hz = max(0.0, _col_at(columns, "f_comp_hz", k, 0.0))
        f_comp_hz[k] = f_hz
        p_prim_th_w[k] = p_prim

        t_top_e = _col_at(columns, "t_buf_top_end_c", k, 22.0, "t_buf_top_c")
        t_top_s = _col_at(columns, "t_buf_top_start_c", k, t_top_e)
        t_bot_e = _col_at(
            columns, "t_buf_bot_end_c", k, t_top_e - 1.0, "t_buf_bot_c"
        )
        t_bot_s = _col_at(columns, "t_buf_bot_start_c", k, t_bot_e)
        t_buf_top_c[k] = t_top_e
        t_buf_bot_cal_c[k] = t_bot_e + house_config.BUF_BOT_SENSOR_OFFSET_C

        tf_wk_in = t_floor_wk_running
        tf_kt_in = t_floor_kt_running

        t_bg_sup = (
            _opt_float(bg_sup_seq[k], 5.0)
            if bg_sup_seq is not None and k < len(bg_sup_seq)
            else None
        )
        t_bg_ret = (
            _opt_float(bg_ret_seq[k], 5.0)
            if bg_ret_seq is not None and k < len(bg_ret_seq)
            else None
        )

        v_sec_tot = u_pump * sum(
            u_valves[g] * flows[g] for g in const.ALL_HYDRAULIC_GROUPS
        )
        t_sup_sec_c[k] = hyd.compute_effective_supply_temp(
            threeway_room=threeway_room,
            v_prim_l_min=v_prim,
            v_sec_total_l_min=v_sec_tot,
            t_buf_top_c=t_top_e,
            t_buf_bot_raw_c=t_bot_e,
        )

        hyd_res = hyd.step(
            dt_s=dt_s[k],
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
            t_floor_wk_c=tf_wk_in,
            t_floor_kt_c=tf_kt_in,
            room_temps_c=room_temps,
            t_bg_sup_c=t_bg_sup,
            t_bg_ret_c=t_bg_ret,
            bg_ret_offset_c=0.0,
        )
        q_wk = hyd_res.q_fh_wk_w
        q_kt = hyd_res.q_fh_kt_w
        q_tot = sum(hyd_res.q_fh_by_group_w.values())

        if (
            "q_fh_wk_w" in columns
            and v_prim == 0.0
            and u_pump == 0.0
            and _col_at(columns, "q_fh_wk_w", k, 0.0) > 0.0
        ):
            q_wk = _col_at(columns, "q_fh_wk_w", k, 0.0)
        if (
            "q_fh_kt_w" in columns
            and v_prim == 0.0
            and u_pump == 0.0
            and _col_at(columns, "q_fh_kt_w", k, 0.0) > 0.0
        ):
            q_kt = _col_at(columns, "q_fh_kt_w", k, 0.0)

        q_fh_wk_hyd_w[k] = q_wk
        q_fh_kt_hyd_w[k] = q_kt
        q_sec_tot_w[k] = q_tot
        f_flow_wk[k] = hyd_res.f_flow_wk
        f_flow_kt[k] = hyd_res.f_flow_kt

        c_wk_j = 3600.0 * house_config.DAY1_PRIORS_WK.c_floor_wh_k
        h_fh_wk = 1000.0 / house_config.DAY1_PRIORS_WK.r_fh_k_kw
        t_floor_wk_running = tf_wk_in + (dt_s[k] / c_wk_j) * (
            q_wk - (tf_wk_in - t_wk) * h_fh_wk
        )
        c_kt_j = 3600.0 * house_config.DAY1_PRIORS_KT.c_floor_wh_k
        h_fh_kt = 1000.0 / house_config.DAY1_PRIORS_KT.r_fh_k_kw
        t_floor_kt_running = tf_kt_in + (dt_s[k] / c_kt_j) * (
            q_kt - (tf_kt_in - t_kt) * h_fh_kt
        )

        t_up_w, t_dn_k = solar_weather.compute_interfloor_boundaries(
            room_temps_c=room_temps,
            last_valid_cache=interfloor_cache,
            ts_utc=ts_k,
        )
        t_up_wk_c[k] = t_up_w
        t_down_wk_c[k] = house_config.T_GROUND_MEAN_C
        t_down_kt_c[k] = t_dn_k
        q_int_wk_w[k] = house_config.BASELOAD_WK_W + max(
            0.0, hyd_res.q_leak_1e_to_wk_w
        )

        p_ac = max(0.0, _col_at(columns, "p_el_ac_kt_w", k, 0.0))
        ac_mode = _col_at(columns, "ac_mode_code", k, 0.0)
        p_desk_opt = (
            _opt_float(desk_seq[k], 0.0)
            if desk_seq is not None and k < len(desk_seq)
            else None
        )
        pir_k = _col_at(columns, "pir_occ_kt", k, 0.0)
        p_grow = max(0.0, _col_at(columns, "p_growlights_kt_w", k, 0.0))
        ac_def = bool(fl_k & const.FLAG_DEFROST_AC)

        p_el_ac_kt_w[k] = p_ac
        ac_mode_code[k] = ac_mode
        p_desk_kt_w[k] = p_desk_opt if p_desk_opt is not None else 0.0
        pir_occ_kt[k] = pir_k

        q_int_k, q_ac_k = solar_weather.compute_internal_and_ac_loads_kt(
            p_desk_w=p_desk_opt,
            pir_occ=pir_k,
            p_growlights_w=p_grow,
            p_el_ac_w=p_ac,
            ac_mode_code=ac_mode,
            ac_defrost=ac_def,
            t_out_c=t_out_k,
            eta_ac=eta_ref,
            desk_state=desk_state,
            dt_s=dt_s[k],
        )
        q_int_kt_w[k] = q_int_k
        q_ac_base_kt_w[k] = q_ac_k

        t_surf_meas_wk_c[k] = (
            _opt_float(surf_seq[k], 5.0)
            if surf_seq is not None and k < len(surf_seq)
            else None
        )
        d_est_col_wk_w[k] = _col_at(columns, "d_est_wk_w", k, 0.0)
        d_est_col_kt_w[k] = _col_at(columns, "d_est_kt_w", k, 0.0)

        cooling_raw[k] = bool(
            fl_k & (const.FLAG_FLOOR_COOL | const.FLAG_AC_COOL_KT)
        ) or round(ac_mode) in (int(const.AC_MODE_COOL), int(const.AC_MODE_DRY))

    cooling_or_tail: list[bool] = [False] * n
    countdown = 0
    for k in range(n):
        if cooling_raw[k]:
            countdown = COOLING_TAIL_STEPS + 1
        if countdown > 0:
            cooling_or_tail[k] = True
            countdown -= 1

    return PreprocessedHistory(
        n_rows=n,
        ts_utc=ts_utc,
        flags=flags,
        dt_s=dt_s,
        t_out_c=t_out_c,
        v_wind_ms=v_wind_ms,
        i_sol_wk_wm2=i_sol_wk_wm2,
        i_sol_kt_wm2=i_sol_kt_wm2,
        tau_screen_wk=tau_screen_wk,
        pv_fallback=pv_fallback,
        t_up_wk_c=t_up_wk_c,
        t_down_wk_c=t_down_wk_c,
        t_down_kt_c=t_down_kt_c,
        q_int_wk_w=q_int_wk_w,
        q_int_kt_w=q_int_kt_w,
        q_ac_base_kt_w=q_ac_base_kt_w,
        p_el_ac_kt_w=p_el_ac_kt_w,
        ac_mode_code=ac_mode_code,
        p_desk_kt_w=p_desk_kt_w,
        pir_occ_kt=pir_occ_kt,
        u_pump_sec=u_pump_sec,
        u_eff_wk=u_eff_wk,
        u_eff_kt=u_eff_kt,
        t_buf_top_c=t_buf_top_c,
        t_buf_bot_cal_c=t_buf_bot_cal_c,
        t_sup_sec_c=t_sup_sec_c,
        f_comp_hz=f_comp_hz,
        p_prim_th_w=p_prim_th_w,
        q_fh_wk_hyd_w=q_fh_wk_hyd_w,
        q_fh_kt_hyd_w=q_fh_kt_hyd_w,
        q_sec_tot_w=q_sec_tot_w,
        f_flow_wk=f_flow_wk,
        f_flow_kt=f_flow_kt,
        t_air_wk_c=t_air_wk_c,
        t_air_kt_c=t_air_kt_c,
        t_surf_meas_wk_c=t_surf_meas_wk_c,
        d_est_col_wk_w=d_est_col_wk_w,
        d_est_col_kt_w=d_est_col_kt_w,
        cooling_or_tail=cooling_or_tail,
        hyd_k_hx_wk=float(k_hx_map["wk"]),
        hyd_k_hx_kt=float(k_hx_map["kantoor"]),
    )


def build_zone_exog_list(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
) -> list[contracts.ExogenousStepInput]:
    """Constructs per-step `ExogenousStepInput` instances for `zone`."""
    n = hist.n_rows
    if zone == "wk":
        return [
            contracts.ExogenousStepInput(
                t_out_c=hist.t_out_c[k],
                v_wind_ms=hist.v_wind_ms[k],
                i_sol_wm2=hist.i_sol_wk_wm2[k],
                t_down_c=hist.t_down_wk_c[k],
                q_int_known_w=hist.q_int_wk_w[k],
                t_up_c=hist.t_up_wk_c[k],
            )
            for k in range(n)
        ]
    return [
        contracts.ExogenousStepInput(
            t_out_c=hist.t_out_c[k],
            v_wind_ms=hist.v_wind_ms[k],
            i_sol_wm2=hist.i_sol_kt_wm2[k],
            t_down_c=hist.t_down_kt_c[k],
            q_int_known_w=hist.q_int_kt_w[k],
            t_up_c=0.0,
        )
        for k in range(n)
    ]


def step_q_fh_for_params(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
    k: int,
    t_floor_prev_c: float,
    params: contracts.ZoneParameters,
) -> float:
    """Computes step `k` floor power `Q_fh` [W] consistent with `params`."""
    fl_k = hist.flags[k]
    u_pump = hist.u_pump_sec[k]
    u_eff = hist.u_eff_wk[k] if zone == "wk" else hist.u_eff_kt[k]
    q_hyd = hist.q_fh_wk_hyd_w[k] if zone == "wk" else hist.q_fh_kt_hyd_w[k]
    base_khx = hist.hyd_k_hx_wk if zone == "wk" else hist.hyd_k_hx_kt

    if u_pump <= 1e-4 or u_eff <= 1e-4:
        if q_hyd > 0.0:
            scale = (params.k_hx_w_k / base_khx) if base_khx > 1e-6 else 1.0
            return q_hyd * scale
        return 0.0

    is_harvest = bool(
        fl_k & (const.FLAG_DHW_TANK | const.FLAG_Z1_OFF_HARVEST)
    ) or (hist.f_comp_hz[k] <= 0.5 and hist.p_prim_th_w[k] <= 50.0)
    if is_harvest:
        return max(
            0.0,
            u_pump * u_eff * params.k_hx_w_k * (hist.t_buf_top_c[k] - t_floor_prev_c),
        )

    if base_khx > 1e-6 and abs(params.k_hx_w_k - base_khx) > 1e-6:
        return q_hyd * (params.k_hx_w_k / base_khx)
    return q_hyd


def step_q_ac_for_params(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
    k: int,
    params: contracts.ZoneParameters,
) -> float:
    """Computes step `k` split-AC thermal power `Q_ac` [W] for `params`."""
    if zone != "kt":
        return 0.0
    ac_def = bool(hist.flags[k] & const.FLAG_DEFROST_AC)
    return compute_ac_heating_power_w(
        p_el_ac_w=hist.p_el_ac_kt_w[k],
        ac_mode_code=hist.ac_mode_code[k],
        ac_defrost=ac_def,
        t_out_c=hist.t_out_c[k],
        eta_ac=params.eta_ac,
        base_q_ac_w=hist.q_ac_base_kt_w[k],
    )


def reconstruct_kalman_states(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
    params: contracts.ZoneParameters,
    exog_list: Sequence[contracts.ExogenousStepInput] | None = None,
) -> tuple[
    list[contracts.ThermalState],
    list[float],
    list[float],
    zoh_kalman.ZoneKalmanEstimator,
]:
    """Runs a deterministic forward Kalman reconstruction pass under `params`.

    Args:
        hist: Preprocessed telemetry slice.
        zone: Zone identifier (`"wk"` or `"kt"`).
        params: Candidate `ZoneParameters` to evaluate.
        exog_list: Optional pre-built exogenous step inputs for `zone`.

    Returns:
        A tuple `(states, q_fh_series, q_ac_series, estimator)` containing the
        posterior `ThermalState` per step, realized `Q_fh` and `Q_ac` series,
        and the final `ZoneKalmanEstimator` instance.
    """
    n = hist.n_rows
    exogs = (
        exog_list if exog_list is not None else build_zone_exog_list(hist, zone)
    )
    est = zoh_kalman.ZoneKalmanEstimator(zone, params)
    if n == 0:
        return [], [], [], est

    valid_flag = const.FLAG_VALID_WK if zone == "wk" else const.FLAG_VALID_KT
    t_meas_arr = hist.t_air_wk_c if zone == "wk" else hist.t_air_kt_c

    states: list[contracts.ThermalState] = []
    q_fh_series: list[float] = [0.0] * n
    q_ac_series: list[float] = [0.0] * n

    q_ac_0 = step_q_ac_for_params(hist, zone, 0, params)
    q_fh_0 = hist.q_fh_wk_hyd_w[0] if zone == "wk" else hist.q_fh_kt_hyd_w[0]
    est.cold_start(
        t_air_c=t_meas_arr[0],
        exog=exogs[0],
        q_fh_w=q_fh_0,
        q_ac_w=q_ac_0,
        u_pump=hist.u_pump_sec[0],
        t_buf_top_c=hist.t_buf_top_c[0],
    )
    states.append(est.state)
    q_fh_series[0] = q_fh_0
    q_ac_series[0] = q_ac_0

    for k in range(1, n):
        fl_k = hist.flags[k]
        valid_k = bool(fl_k & valid_flag)
        aanwezig_k = bool(fl_k & const.FLAG_AANWEZIG)
        dt_k = hist.dt_s[k]

        q_fh_k = step_q_fh_for_params(hist, zone, k, est.state.t_floor_c, params)
        q_ac_k = step_q_ac_for_params(hist, zone, k, params)
        q_fh_series[k] = q_fh_k
        q_ac_series[k] = q_ac_k

        if dt_k > 10800.0:
            est.cold_start(
                t_air_c=t_meas_arr[k],
                exog=exogs[k],
                q_fh_w=q_fh_k,
                q_ac_w=q_ac_k,
                u_pump=hist.u_pump_sec[k],
                t_buf_top_c=hist.t_buf_top_c[k],
            )
            states.append(est.state)
            continue

        temp_changed = abs(t_meas_arr[k] - t_meas_arr[k - 1]) >= 1e-4
        hum_changed = valid_k and (not temp_changed) and (k % 6 == 0)
        t_surf = hist.t_surf_meas_wk_c[k] if zone == "wk" else None

        st = est.step(
            ts_utc=hist.ts_utc[k],
            dt_s=dt_k,
            exog=exogs[k],
            q_fh_w=q_fh_k,
            q_ac_w=q_ac_k,
            u_pump=hist.u_pump_sec[k],
            t_air_meas_c=t_meas_arr[k] if valid_k else None,
            temp_changed=temp_changed,
            hum_changed=hum_changed,
            sensor_stale=not valid_k,
            aanwezig=aanwezig_k,
            t_surf_meas_c=t_surf,
        )
        states.append(st)

    return states, q_fh_series, q_ac_series, est


class LiveHindcastTracker:
    """FIFO tracker of in-flight 72-step (6-hour) open-loop predictions."""

    def __init__(self, horizon_steps: int = HORIZON_6H_STEPS) -> None:
        """Initializes the tracker with a target rollout horizon.

        Args:
            horizon_steps: Number of 5-minute ZOH steps to track (`72` = 6h).
        """
        self._horizon_steps: int = int(horizon_steps)
        self._queue: collections.deque[list[float | int]] = collections.deque()
        self._last_published_c: float | None = None
        self._last_published_exact_c: float | None = None
        self._warmup_remaining_min: int = self._horizon_steps * 5

    @property
    def horizon_steps(self) -> int:
        """Number of 5-minute steps in the hindcast horizon."""
        return self._horizon_steps

    @property
    def queue(self) -> collections.deque[list[float | int]]:
        """FIFO queue of `[age_steps, t_air_ol, t_floor_ol, d_ol]` entries."""
        return self._queue

    @property
    def last_published_c(self) -> float | None:
        """Most recently published 6h hindcast temperature rounded to 0.01 °C."""
        return self._last_published_c

    @property
    def last_published_exact_c(self) -> float | None:
        """Unrounded float of the most recently popped 72-step prediction."""
        return self._last_published_exact_c

    @property
    def warmup_remaining_min(self) -> int:
        """Minutes remaining before the oldest seed reaches `horizon_steps`."""
        return self._warmup_remaining_min

    def clear(self) -> None:
        """Clears the FIFO queue on cold start or outages exceeding 3 hours."""
        self._queue.clear()
        self._last_published_c = None
        self._last_published_exact_c = None
        self._warmup_remaining_min = self._horizon_steps * 5

    def step(
        self,
        posterior_state: contracts.ThermalState,
        params: contracts.ZoneParameters,
        exog_realized: contracts.ExogenousStepInput,
        q_fh_realized_w: float,
        q_ac_realized_w: float = 0.0,
        dt_s: float = 300.0,
    ) -> tuple[float | None, int]:
        """Advances queued states 1 ZOH step, pops at age 72, and seeds posterior.

        Args:
            posterior_state: Fresh closed-loop Kalman posterior at window end.
            params: Active `ZoneParameters` for open-loop propagation.
            exog_realized: Realized exogenous boundary inputs over the step.
            q_fh_realized_w: Realized floor heating power over the step [W].
            q_ac_realized_w: Realized split-AC thermal power over the step [W].
            dt_s: Step duration [s].

        Returns:
            A tuple `(published_hindcast_c, warmup_remaining_min)`, where
            `published_hindcast_c` is `round(t_air_ol, 2)` once warmed up (or
            `None` during the initial 6-hour spin-up).
        """
        for entry in self._queue:
            st_prev = contracts.ThermalState(
                t_air_c=float(entry[1]),
                t_floor_c=float(entry[2]),
                d_est_w=float(entry[3]),
            )
            st_next = zoh_kalman.propagate_open_loop_step(
                state=st_prev,
                params=params,
                exog=exog_realized,
                q_fh_w=q_fh_realized_w,
                q_ac_w=q_ac_realized_w,
                dt_s=dt_s,
            )
            entry[0] = int(entry[0]) + 1
            entry[1] = st_next.t_air_c
            entry[2] = st_next.t_floor_c
            entry[3] = st_next.d_est_w

        published_c: float | None = None
        while self._queue and int(self._queue[0][0]) >= self._horizon_steps:
            oldest = self._queue.popleft()
            if int(oldest[0]) == self._horizon_steps or published_c is None:
                exact_val = float(oldest[1])
                self._last_published_exact_c = exact_val
                published_c = round(exact_val, 2)

        self._queue.append(
            [
                0,
                posterior_state.t_air_c,
                posterior_state.t_floor_c,
                posterior_state.d_est_w,
            ]
        )

        if published_c is not None:
            self._last_published_c = published_c
            self._warmup_remaining_min = 0
            return published_c, 0

        oldest_age = int(self._queue[0][0]) if self._queue else 0
        rem_min = max(5, (self._horizon_steps - oldest_age) * 5)
        self._warmup_remaining_min = rem_min
        return None, rem_min

    def bridge_gap(
        self,
        dt_gap_s: float,
        params: contracts.ZoneParameters,
        exog_avg: contracts.ExogenousStepInput,
        q_fh_avg_w: float,
        q_ac_avg_w: float = 0.0,
    ) -> None:
        """Bridges a Home Assistant restart gap (`<= 3h`) via variable-`dt` ZOH.

        Args:
            dt_gap_s: Duration of the downtime gap [s]. Gaps `> 10800 s` clear
                the queue.
            params: Active `ZoneParameters`.
            exog_avg: Average exogenous inputs across the restart gap.
            q_fh_avg_w: Average floor heating power across the gap [W].
            q_ac_avg_w: Average split-AC thermal power across the gap [W].
        """
        if dt_gap_s <= 0.0:
            return
        if dt_gap_s > 10800.0:
            self.clear()
            return

        gap_steps = max(1, int(round(dt_gap_s / float(const.DT_STEP_S))))
        for entry in self._queue:
            st_prev = contracts.ThermalState(
                t_air_c=float(entry[1]),
                t_floor_c=float(entry[2]),
                d_est_w=float(entry[3]),
            )
            st_next = zoh_kalman.propagate_open_loop_step(
                state=st_prev,
                params=params,
                exog=exog_avg,
                q_fh_w=q_fh_avg_w,
                q_ac_w=q_ac_avg_w,
                dt_s=dt_gap_s,
            )
            entry[0] = int(entry[0]) + gap_steps
            entry[1] = st_next.t_air_c
            entry[2] = st_next.t_floor_c
            entry[3] = st_next.d_est_w

        while self._queue and int(self._queue[0][0]) >= self._horizon_steps:
            popped = self._queue.popleft()
            if int(popped[0]) == self._horizon_steps:
                self._last_published_exact_c = float(popped[1])
                self._last_published_c = round(float(popped[1]), 2)
                self._warmup_remaining_min = 0

        if self._queue and self._last_published_c is None:
            oldest_age = int(self._queue[0][0])
            self._warmup_remaining_min = max(
                5, (self._horizon_steps - oldest_age) * 5
            )

    def reseed_from_history(
        self,
        steps_data: Sequence[dict[str, Any]],
        params: contracts.ZoneParameters,
    ) -> float | None:
        """Rebuilds the 72-step FIFO queue after a parameter update or rollback.

        Args:
            steps_data: Chronological sequence of step dictionaries containing
                `posterior_state`, `exog`, `q_fh_w`, `q_ac_w`, and `dt_s`.
            params: Newly applied `ZoneParameters`.

        Returns:
            Immediate 6h hindcast temperature [°C] if `>= 73` history steps
            were replayed, otherwise `None`.
        """
        self.clear()
        if not steps_data:
            return None

        tail = steps_data[-(self._horizon_steps + 1) :]

        first = tail[0]
        init_st = first.get("initial_state") or first.get("prior_state")
        if len(tail) == self._horizon_steps and isinstance(
            init_st, contracts.ThermalState
        ):
            self._queue.append(
                [0, init_st.t_air_c, init_st.t_floor_c, init_st.d_est_w]
            )

        last_pub: float | None = None
        for item in tail:
            post = item.get("posterior_state") or item.get("state")
            if not isinstance(post, contracts.ThermalState):
                post = contracts.ThermalState(
                    t_air_c=float(item["t_air_c"]),
                    t_floor_c=float(item["t_floor_c"]),
                    d_est_w=float(item.get("d_est_w", 0.0)),
                )
            exog = item.get("exog") or item.get("exog_realized")
            if not isinstance(exog, contracts.ExogenousStepInput):
                exog = contracts.ExogenousStepInput(
                    t_out_c=float(item.get("t_out_c", 10.0)),
                    v_wind_ms=float(item.get("v_wind_ms", 3.0)),
                    i_sol_wm2=float(item.get("i_sol_wm2", 0.0)),
                    t_down_c=float(
                        item.get("t_down_c", house_config.T_GROUND_MEAN_C)
                    ),
                    q_int_known_w=float(
                        item.get("q_int_known_w", house_config.BASELOAD_WK_W)
                    ),
                    t_up_c=float(item.get("t_up_c", 18.0)),
                )
            q_fh = float(item.get("q_fh_w", item.get("q_fh_realized_w", 0.0)))
            q_ac = float(item.get("q_ac_w", item.get("q_ac_realized_w", 0.0)))
            dt_step = float(item.get("dt_s", const.DT_STEP_S))
            last_pub, _ = self.step(post, params, exog, q_fh, q_ac, dt_step)

        return last_pub

    def to_dict(self) -> dict[str, Any]:
        """Serializes tracker state for Home Assistant Store persistence."""
        return {
            "horizon_steps": self._horizon_steps,
            "queue": [
                [int(it[0]), float(it[1]), float(it[2]), float(it[3])]
                for it in self._queue
            ],
            "last_published_c": self._last_published_c,
            "warmup_remaining_min": self._warmup_remaining_min,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LiveHindcastTracker:
        """Restores a `LiveHindcastTracker` from a persisted dictionary."""
        horizon = int(data.get("horizon_steps", HORIZON_6H_STEPS))
        tracker = cls(horizon_steps=horizon)
        for row in data.get("queue", ()):
            if len(row) >= 4:
                tracker._queue.append(
                    [int(row[0]), float(row[1]), float(row[2]), float(row[3])]
                )
        last_pub = data.get("last_published_c")
        tracker._last_published_c = (
            float(last_pub) if last_pub is not None else None
        )
        tracker._last_published_exact_c = tracker._last_published_c
        tracker._warmup_remaining_min = int(
            data.get(
                "warmup_remaining_min",
                horizon * 5 if not tracker._queue else 0,
            )
        )
        return tracker


def _empty_regime_metrics(
    regime: str,
    horizon_steps: int,
) -> contracts.RegimeHindcastMetrics:
    """Returns an empty `RegimeHindcastMetrics` record when `n_windows == 0`."""
    return contracts.RegimeHindcastMetrics(
        regime=regime,
        horizon_steps=horizon_steps,
        n_windows=0,
        rmse_end_c=None,
        rmse_path_c=None,
        mae_end_c=None,
        bias_end_c=None,
        j_eval_c=None,
        prior_rmse_end_c=None,
        skill_score_pct=None,
    )


def _compute_regime_metrics(
    regime: str,
    horizon_steps: int,
    eval_err_paths: Sequence[Sequence[float]],
    prior_err_paths: Sequence[Sequence[float]],
) -> contracts.RegimeHindcastMetrics:
    """Computes `RMSE_end`, `RMSE_path`, `MAE`, `Bias`, `J_eval`, and `Skill Score`."""
    w_cnt = len(eval_err_paths)
    if w_cnt == 0:
        return _empty_regime_metrics(regime, horizon_steps)

    sum_sq_end = 0.0
    sum_abs_end = 0.0
    sum_bias_end = 0.0
    sum_path_mse = 0.0
    sum_sq_prior_end = 0.0

    for e_eval, e_prior in zip(eval_err_paths, prior_err_paths):
        lw = len(e_eval)
        end_err = e_eval[-1]
        prior_end_err = e_prior[-1]
        sum_sq_end += end_err * end_err
        sum_abs_end += abs(end_err)
        sum_bias_end += end_err
        sum_path_mse += sum(x * x for x in e_eval) / float(lw)
        sum_sq_prior_end += prior_end_err * prior_end_err

    rmse_end = math.sqrt(sum_sq_end / float(w_cnt))
    rmse_path = math.sqrt(sum_path_mse / float(w_cnt))
    mae_end = sum_abs_end / float(w_cnt)
    bias_end = sum_bias_end / float(w_cnt)
    j_eval = 0.65 * rmse_end + 0.35 * rmse_path
    prior_rmse_end = math.sqrt(sum_sq_prior_end / float(w_cnt))
    skill_score_pct = 100.0 * (1.0 - rmse_end / max(prior_rmse_end, 0.02))

    return contracts.RegimeHindcastMetrics(
        regime=regime,
        horizon_steps=horizon_steps,
        n_windows=w_cnt,
        rmse_end_c=rmse_end,
        rmse_path_c=rmse_path,
        mae_end_c=mae_end,
        bias_end_c=bias_end,
        j_eval_c=j_eval,
        prior_rmse_end_c=prior_rmse_end,
        skill_score_pct=skill_score_pct,
    )


def _rollout_open_loop_window(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
    k0: int,
    length_steps: int,
    state_init: contracts.ThermalState,
    params: contracts.ZoneParameters,
    exog_list: Sequence[contracts.ExogenousStepInput],
    t_meas_arr: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Rolls out open-loop ZOH from `k0` for `length_steps`."""
    st = state_init
    errs: list[float] = []
    q_fhs: list[float] = []
    for s in range(1, length_steps + 1):
        idx = k0 + s
        q_fh_s = step_q_fh_for_params(hist, zone, idx, st.t_floor_c, params)
        q_ac_s = step_q_ac_for_params(hist, zone, idx, params)
        st = zoh_kalman.propagate_open_loop_step(
            state=st,
            params=params,
            exog=exog_list[idx],
            q_fh_w=q_fh_s,
            q_ac_w=q_ac_s,
            dt_s=hist.dt_s[idx],
        )
        errs.append(st.t_air_c - t_meas_arr[idx])
        q_fhs.append(q_fh_s)
    return errs, q_fhs


def _is_valid_eval_start(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
    k0: int,
    t_meas_arr: Sequence[float],
    d_act: Sequence[float],
) -> bool:
    """Checks Component 4 §4.1 start-point validity at step `k0`."""
    valid_flag = const.FLAG_VALID_WK if zone == "wk" else const.FLAG_VALID_KT
    open_win_flag = (
        const.FLAG_OPEN_WIN_WK if zone == "wk" else const.FLAG_OPEN_WIN_KT
    )
    fl0 = hist.flags[k0]
    if not (fl0 & valid_flag) or hist.cooling_or_tail[k0]:
        return False
    if fl0 & open_win_flag:
        return False
    if k0 >= 3 and (t_meas_arr[k0] - t_meas_arr[k0 - 3]) / 0.25 < -0.8:
        return False
    if abs(d_act[k0]) > 200.0:
        return False
    return True


def _truncate_on_unmodeled_event(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
    k0: int,
    horizon_steps: int,
    d_act: Sequence[float],
    tau_d_s: float,
    t_meas_arr: Sequence[float],
    is_ac_window: bool = False,
) -> int:
    """Returns clean causal window length `L_w <= horizon_steps` (Component 4 §4.2)."""
    valid_flag = const.FLAG_VALID_WK if zone == "wk" else const.FLAG_VALID_KT
    open_win_flag = (
        const.FLAG_OPEN_WIN_WK if zone == "wk" else const.FLAG_OPEN_WIN_KT
    )
    d0 = d_act[k0]

    for s in range(1, horizon_steps + 1):
        idx = k0 + s
        fl = hist.flags[idx]
        if (
            not (fl & valid_flag)
            or (fl & const.FLAG_GAP_BRIDGED)
            or hist.dt_s[idx] > 600.0
        ):
            return s - 1
        if hist.cooling_or_tail[idx]:
            return s - 1
        if fl & open_win_flag:
            return s - 1
        if idx >= 3 and (t_meas_arr[idx] - t_meas_arr[idx - 3]) / 0.25 < -0.8:
            return s - 1

        active_ac_step = (
            is_ac_window
            and round(hist.ac_mode_code[idx]) == int(const.AC_MODE_HEAT)
            and hist.p_el_ac_kt_w[idx] > 80.0
        )
        if not active_ac_step:
            expected_d = d0 * math.exp(-(s * float(const.DT_STEP_S)) / tau_d_s)
            if abs(d_act[idx] - expected_d) > 250.0:
                return s - 1

    return horizon_steps


def evaluate_14d_hindcast(
    columns: Mapping[str, Sequence[float | int]],
    zone: contracts.ZoneId,
    params_eval: contracts.ZoneParameters,
    params_prior: contracts.ZoneParameters,
    flow_rates_l_min: Mapping[str, float] | None = None,
    k_hx_by_group: Mapping[str, float] | None = None,
    *,
    lat_deg: float = house_config.LATITUDE_DEG,
    lon_deg: float = house_config.LONGITUDE_DEG,
    _ref_d_act: Sequence[float] | None = None,
) -> contracts.ZoneEvaluationReport:
    """Evaluates 14-day 6h (and `kt` 1h AC) open-loop hindcast performance.

    Runs dual parameter-consistent forward Kalman reconstructions under
    `params_eval` and `params_prior` over up to 16 days (`48h` spin-up + `14d`
    evaluation) so comparisons against Day-1 priors suffer zero initial-state
    enthalpy jump.

    Args:
        columns: RingBuffer column sequences.
        zone: Target zone (`"wk"` or `"kt"`).
        params_eval: Active or candidate `ZoneParameters` to score.
        params_prior: Reference Day-1 prior `ZoneParameters` for Skill Score.
        flow_rates_l_min: Optional manifold flow rates [L/min].
        k_hx_by_group: Optional `K_hx` [W/K] mapping per group.
        _ref_d_act: Optional shared disturbance series used by the holdout gate
            to guarantee identical window selection between two parameter sets.

    Returns:
        Complete `ZoneEvaluationReport` across `overall_6h` and all physical
        regimes.
    """
    ts_all = columns.get("ts_utc", ())
    n_total = len(ts_all)

    regime_names = (
        ("night_cooling", "heating_and_coast", "sunny_afternoon")
        if zone == "wk"
        else (
            "night_cooling",
            "heating_and_coast",
            "sunny_afternoon",
            "airco_heating_1h",
        )
    )

    if n_total < SPINUP_48H_STEPS:
        ts_eval = int(ts_all[-1]) if n_total > 0 else 0
        by_reg = {
            r: _empty_regime_metrics(
                r,
                HORIZON_AC_1H_STEPS
                if r == "airco_heating_1h"
                else HORIZON_6H_STEPS,
            )
            for r in regime_names
        }
        return contracts.ZoneEvaluationReport(
            zone=zone,
            ts_evaluated_utc=ts_eval,
            overall_6h=_empty_regime_metrics("overall_6h", HORIZON_6H_STEPS),
            by_regime=by_reg,
        )

    start_slice = max(0, n_total - MAX_EVAL_SLICE_STEPS)
    sub_cols = slice_columns(columns, start_slice, n_total)
    n_slice = n_total - start_slice

    if n_slice >= SPINUP_48H_STEPS + HORIZON_6H_STEPS:
        spinup_steps = SPINUP_48H_STEPS
    else:
        spinup_steps = max(288, n_slice - 2 * HORIZON_6H_STEPS)

    k_hx_eval_map = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
    if k_hx_by_group is not None:
        for group_name, val in k_hx_by_group.items():
            target_key = "kantoor" if group_name == "kt" else group_name
            k_hx_eval_map[target_key] = float(val)
    k_hx_eval_map["wk" if zone == "wk" else "kantoor"] = params_eval.k_hx_w_k

    k_hx_prior_map = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
    k_hx_prior_map["wk" if zone == "wk" else "kantoor"] = params_prior.k_hx_w_k

    hist_eval = preprocess_history_columns(
        sub_cols,
        flow_rates_l_min=flow_rates_l_min,
        k_hx_by_group=k_hx_eval_map,
        eta_ac_ref=params_eval.eta_ac,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )
    hist_prior = preprocess_history_columns(
        sub_cols,
        flow_rates_l_min=flow_rates_l_min,
        k_hx_by_group=k_hx_prior_map,
        eta_ac_ref=params_prior.eta_ac,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )

    exog_eval = build_zone_exog_list(hist_eval, zone)
    exog_prior = build_zone_exog_list(hist_prior, zone)

    states_eval, _, _, _ = reconstruct_kalman_states(
        hist_eval, zone, params_eval, exog_eval
    )
    states_prior, _, _, _ = reconstruct_kalman_states(
        hist_prior, zone, params_prior, exog_prior
    )

    if _ref_d_act is not None and len(_ref_d_act) == n_slice:
        d_act = list(_ref_d_act)
    else:
        d_act = [st.d_est_w for st in states_eval]

    t_meas = hist_eval.t_air_wk_c if zone == "wk" else hist_eval.t_air_kt_c
    i_sol_arr = (
        hist_eval.i_sol_wk_wm2 if zone == "wk" else hist_eval.i_sol_kt_wm2
    )
    tau_d_s = max(300.0, params_eval.tau_d_h * 3600.0)

    overall_eval: list[list[float]] = []
    overall_prior: list[list[float]] = []
    reg_eval: dict[str, list[list[float]]] = {r: [] for r in regime_names}
    reg_prior: dict[str, list[list[float]]] = {r: [] for r in regime_names}

    max_k0_6h = n_slice - 1 - HORIZON_6H_STEPS
    for k0 in range(spinup_steps, max_k0_6h + 1, STRIDE_6H_STEPS):
        if not _is_valid_eval_start(hist_eval, zone, k0, t_meas, d_act):
            continue

        lw = _truncate_on_unmodeled_event(
            hist_eval,
            zone,
            k0,
            HORIZON_6H_STEPS,
            d_act,
            tau_d_s,
            t_meas,
            is_ac_window=False,
        )
        if lw < MIN_CLEAN_6H_STEPS:
            continue

        e_eval, q_fh_ol = _rollout_open_loop_window(
            hist_eval,
            zone,
            k0,
            lw,
            states_eval[k0],
            params_eval,
            exog_eval,
            t_meas,
        )
        e_prior, _ = _rollout_open_loop_window(
            hist_prior,
            zone,
            k0,
            lw,
            states_prior[k0],
            params_prior,
            exog_prior,
            t_meas,
        )

        overall_eval.append(e_eval)
        overall_prior.append(e_prior)

        hr0, min0 = solar_weather.local_hour_minute(hist_eval.ts_utc[k0])
        mins_of_day = hr0 * 60 + min0

        e_fh_wh = sum(
            q_fh_ol[s - 1] * (hist_eval.dt_s[k0 + s] / 3600.0)
            for s in range(1, lw + 1)
        )
        sol_slice = [i_sol_arr[k0 + s] for s in range(1, lw + 1)]
        max_sol = max(sol_slice)
        mean_sol = sum(sol_slice) / float(lw)
        max_ac = max(hist_eval.p_el_ac_kt_w[k0 + s] for s in range(1, lw + 1))
        max_desk = max(hist_eval.p_desk_kt_w[k0 + s] for s in range(1, lw + 1))

        if (
            30 <= mins_of_day <= 120
            and e_fh_wh < 100.0
            and max_sol < 10.0
            and (zone != "kt" or (max_ac <= 25.0 and max_desk <= 30.0))
        ):
            reg_eval["night_cooling"].append(e_eval)
            reg_prior["night_cooling"].append(e_prior)

        min_heat_wh = 3000.0 if zone == "wk" else 800.0
        if (
            e_fh_wh >= min_heat_wh
            and mean_sol < 150.0
            and (zone != "kt" or max_ac <= 25.0)
        ):
            reg_eval["heating_and_coast"].append(e_eval)
            reg_prior["heating_and_coast"].append(e_prior)

        if 570 <= mins_of_day <= 810 and (
            mean_sol >= 120.0 or max_sol >= 250.0
        ):
            reg_eval["sunny_afternoon"].append(e_eval)
            reg_prior["sunny_afternoon"].append(e_prior)

    if zone == "kt":
        max_k0_ac = n_slice - 1 - HORIZON_AC_1H_STEPS
        cand_k0_set: set[int] = set(
            range(spinup_steps, max_k0_ac + 1, STRIDE_AC_1H_STEPS)
        )
        for k in range(spinup_steps, max_k0_ac + 1):
            if (
                hist_eval.p_el_ac_kt_w[k] <= 80.0
                and hist_eval.p_el_ac_kt_w[k + 1] > 80.0
            ):
                cand_k0_set.add(k)

        for k0 in sorted(cand_k0_set):
            if not _is_valid_eval_start(hist_eval, zone, k0, t_meas, d_act):
                continue

            first_9_high = sum(
                1
                for s in range(1, 10)
                if round(hist_eval.ac_mode_code[k0 + s])
                == int(const.AC_MODE_HEAT)
                and hist_eval.p_el_ac_kt_w[k0 + s] > 80.0
            )
            if first_9_high < 3:
                continue

            ac_wh_9 = sum(
                hist_eval.p_el_ac_kt_w[k0 + s]
                * (hist_eval.dt_s[k0 + s] / 3600.0)
                for s in range(1, 10)
                if round(hist_eval.ac_mode_code[k0 + s])
                == int(const.AC_MODE_HEAT)
            )
            if ac_wh_9 < 40.0:
                continue

            lw_ac = _truncate_on_unmodeled_event(
                hist_eval,
                zone,
                k0,
                HORIZON_AC_1H_STEPS,
                d_act,
                tau_d_s,
                t_meas,
                is_ac_window=True,
            )
            if lw_ac < HORIZON_AC_1H_STEPS:
                continue

            e_eval_ac, q_fh_ac = _rollout_open_loop_window(
                hist_eval,
                zone,
                k0,
                HORIZON_AC_1H_STEPS,
                states_eval[k0],
                params_eval,
                exog_eval,
                t_meas,
            )
            if max(q_fh_ac) >= 300.0:
                continue

            e_prior_ac, _ = _rollout_open_loop_window(
                hist_prior,
                zone,
                k0,
                HORIZON_AC_1H_STEPS,
                states_prior[k0],
                params_prior,
                exog_prior,
                t_meas,
            )
            reg_eval["airco_heating_1h"].append(e_eval_ac)
            reg_prior["airco_heating_1h"].append(e_prior_ac)

    overall_metrics = _compute_regime_metrics(
        "overall_6h", HORIZON_6H_STEPS, overall_eval, overall_prior
    )
    by_regime_metrics = {
        r: _compute_regime_metrics(
            r,
            HORIZON_AC_1H_STEPS
            if r == "airco_heating_1h"
            else HORIZON_6H_STEPS,
            reg_eval[r],
            reg_prior[r],
        )
        for r in regime_names
    }

    return contracts.ZoneEvaluationReport(
        zone=zone,
        ts_evaluated_utc=int(hist_eval.ts_utc[-1]),
        overall_6h=overall_metrics,
        by_regime=by_regime_metrics,
    )


def zone_eval_report_to_dict(
    report: contracts.ZoneEvaluationReport | None,
) -> dict[str, Any] | None:
    """Converts a `ZoneEvaluationReport` into a JSON-serializable dictionary."""
    if report is None:
        return None
    return {
        "zone": report.zone,
        "ts_evaluated_utc": report.ts_evaluated_utc,
        "overall_6h": dataclasses.asdict(report.overall_6h),
        "by_regime": {
            k: dataclasses.asdict(v) for k, v in report.by_regime.items()
        },
    }


def zone_eval_report_from_dict(
    data: Mapping[str, Any] | None,
) -> contracts.ZoneEvaluationReport | None:
    """Reconstructs a `ZoneEvaluationReport` from a dictionary."""
    if not data:
        return None
    overall_raw = data["overall_6h"]
    overall = contracts.RegimeHindcastMetrics(**overall_raw)
    by_reg = {
        k: contracts.RegimeHindcastMetrics(**v)
        for k, v in data.get("by_regime", {}).items()
    }
    return contracts.ZoneEvaluationReport(
        zone="wk" if data["zone"] == "wk" else "kt",
        ts_evaluated_utc=int(data["ts_evaluated_utc"]),
        overall_6h=overall,
        by_regime=by_reg,
    )


def sanitize_json_payload(obj: Any) -> Any:
    """Recursively replaces non-finite floats (`NaN`, `Inf`) with `None`."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sanitize_json_payload(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): sanitize_json_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json_payload(item) for item in obj]
    return obj


def _fmt_csv_float(val: float | None, decimals: int = 3) -> str:
    """Formats a finite float to `decimals` places, or `""` when missing."""
    if val is None or not math.isfinite(val):
        return ""
    return f"{val:.{decimals}f}"


def _zone_params_from_mapping(
    raw: Any,
    fallback: contracts.ZoneParameters,
) -> contracts.ZoneParameters:
    """Parses `ZoneParameters` from `fit_report`, falling back to `fallback`."""
    if not isinstance(raw, Mapping) or "r_fh_k_kw" not in raw:
        return fallback
    merged = dataclasses.asdict(fallback)
    for key in merged:
        if key in raw and raw[key] is not None:
            merged[key] = float(raw[key])
    return contracts.ZoneParameters(**merged)


def export_csv_and_report(
    export_dir: str | pathlib.Path,
    columns: Mapping[str, Sequence[float | int]],
    fit_report: Mapping[str, Any],
    lat_deg: float = house_config.LATITUDE_DEG,
    lon_deg: float = house_config.LONGITUDE_DEG,
) -> tuple[str, str]:
    """Atomically writes 31-column `telemetry_5m.csv` and `fit_report.json`.

    Uses Syncthing-safe `.tmp` -> `flush()` -> `os.fsync()` -> `os.replace()`.

    Args:
        export_dir: Target directory path.
        columns: RingBuffer telemetry column sequences.
        fit_report: Diagnostic fit report dictionary.

    Returns:
        A tuple `(csv_path, json_path)` of written absolute/resolved file paths.
    """
    out_dir = pathlib.Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_final = out_dir / "telemetry_5m.csv"
    csv_tmp = out_dir / "telemetry_5m.csv.tmp"
    json_final = out_dir / "fit_report.json"
    json_tmp = out_dir / "fit_report.json.tmp"

    flows = fit_report.get("flow_weights_l_min")
    flow_map = (
        flows if isinstance(flows, Mapping) else const.DEFAULT_FLOW_RATES_L_MIN
    )

    zones_info = fit_report.get("zones", {})
    wk_raw = (
        zones_info.get("wk", {}).get("parameters_active")
        if isinstance(zones_info, Mapping)
        else None
    )
    kt_raw = (
        zones_info.get("kt", {}).get("parameters_active")
        if isinstance(zones_info, Mapping)
        else None
    )
    params_wk = _zone_params_from_mapping(wk_raw, house_config.DAY1_PRIORS_WK)
    params_kt = _zone_params_from_mapping(kt_raw, house_config.DAY1_PRIORS_KT)

    hist = preprocess_history_columns(
        columns,
        flow_rates_l_min=flow_map,
        k_hx_by_group={"wk": params_wk.k_hx_w_k, "kantoor": params_kt.k_hx_w_k},
        eta_ac_ref=params_kt.eta_ac,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )

    n = hist.n_rows
    t_floor_wk_seq = columns.get("t_floor_wk_c", ())
    t_floor_kt_seq = columns.get("t_floor_kt_c", ())
    q_fh_wk_seq = columns.get("q_fh_wk_w", ())
    t_hind_wk_seq = columns.get("t_hind6h_wk_c", ())
    t_hind_kt_seq = columns.get("t_hind6h_kt_c", ())
    p_wp_raw_seq = columns.get("p_wp_raw_el_w", ())

    with open(csv_tmp, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(CSV_EXPORT_COLUMNS)
        for k in range(n):
            ts_k = hist.ts_utc[k]
            iso_k = datetime.datetime.fromtimestamp(
                ts_k, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            tf_wk = (
                float(t_floor_wk_seq[k])
                if k < len(t_floor_wk_seq)
                else hist.t_air_wk_c[k]
            )
            tf_kt = (
                float(t_floor_kt_seq[k])
                if k < len(t_floor_kt_seq)
                else hist.t_air_kt_c[k]
            )
            q_wk = (
                float(q_fh_wk_seq[k])
                if k < len(q_fh_wk_seq)
                else hist.q_fh_wk_hyd_w[k]
            )
            t_hind_wk = (
                float(t_hind_wk_seq[k])
                if k < len(t_hind_wk_seq)
                else math.nan
            )
            t_hind_kt = (
                float(t_hind_kt_seq[k])
                if k < len(t_hind_kt_seq)
                else math.nan
            )
            p_wp_raw = (
                _safe_float(p_wp_raw_seq[k], 0.0)
                if k < len(p_wp_raw_seq)
                else 0.0
            )

            t_surf_wk = estimate_zone_surface_temp(
                "wk", hist.t_air_wk_c[k], tf_wk, params_wk
            )
            t_surf_kt = estimate_zone_surface_temp(
                "kt", hist.t_air_kt_c[k], tf_kt, params_kt
            )

            writer.writerow(
                [
                    ts_k,
                    iso_k,
                    hist.flags[k],
                    _fmt_csv_float(hist.t_out_c[k], 2),
                    _fmt_csv_float(hist.v_wind_ms[k], 2),
                    _fmt_csv_float(hist.i_sol_wk_wm2[k], 1),
                    _fmt_csv_float(hist.i_sol_kt_wm2[k], 1),
                    _fmt_csv_float(hist.tau_screen_wk[k], 3),
                    _fmt_csv_float(hist.t_buf_top_c[k], 2),
                    _fmt_csv_float(hist.t_buf_bot_cal_c[k], 2),
                    _fmt_csv_float(hist.q_sec_tot_w[k], 1),
                    _fmt_csv_float(hist.u_eff_wk[k], 3),
                    _fmt_csv_float(hist.f_flow_wk[k], 3),
                    _fmt_csv_float(hist.t_air_wk_c[k], 2),
                    _fmt_csv_float(tf_wk, 3),
                    _fmt_csv_float(hist.d_est_col_wk_w[k], 1),
                    _fmt_csv_float(q_wk, 1),
                    _fmt_csv_float(t_hind_wk, 2),
                    _fmt_csv_float(hist.u_eff_kt[k], 3),
                    _fmt_csv_float(hist.f_flow_kt[k], 3),
                    _fmt_csv_float(hist.t_air_kt_c[k], 2),
                    _fmt_csv_float(hist.t_down_kt_c[k], 2),
                    _fmt_csv_float(hist.q_int_kt_w[k], 1),
                    _fmt_csv_float(hist.p_el_ac_kt_w[k], 1),
                    _fmt_csv_float(tf_kt, 3),
                    _fmt_csv_float(hist.d_est_col_kt_w[k], 1),
                    _fmt_csv_float(t_hind_kt, 2),
                    _fmt_csv_float(t_surf_wk, 2),
                    _fmt_csv_float(t_surf_kt, 2),
                    _fmt_csv_float(p_wp_raw, 1),
                    _fmt_csv_float(hist.f_comp_hz[k], 1),
                ]
            )
        f_csv.flush()
        os.fsync(f_csv.fileno())
    os.replace(csv_tmp, csv_final)

    with open(json_tmp, "w", encoding="utf-8") as f_json:
        json.dump(
            sanitize_json_payload(dict(fit_report)),
            f_json,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        f_json.flush()
        os.fsync(f_json.fileno())
    os.replace(json_tmp, json_final)

    return str(csv_final), str(json_final)
