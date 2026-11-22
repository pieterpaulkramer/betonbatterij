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

_LEGACY_LTS_ACCUM_FLAGS: int = (
    const.FLAG_DHW_TANK
    | const.FLAG_Z1_OFF_HARVEST
    | const.FLAG_PV_FALLBACK
    | const.FLAG_DEFROST_WP
    | const.FLAG_DEFROST_AC
    | const.FLAG_INTERNAL_HEATER
)
_LEGACY_LTS_UPSTAIRS_VALVE_COLS: tuple[str, ...] = (
    "u_eff_slaapkamer",
    "u_eff_badkamer",
    "u_eff_werkkamer_1e",
    "u_eff_kamer_1e_midden",
    "u_eff_kamer_1e_zuid",
    "u_eff_badkamer_boven",
    "u_eff_logeerkamer",
)
_LEGACY_LTS_INDEP_CHANNELS: tuple[str, ...] = (
    "t_out_dak_c",
    "t_out_hm_c",
    "t_air_wk_c",
    "t_air_kantoor_c",
    "p_p1_net_w",
)

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
        p_wp_raw_el_w: Raw heat-pump electrical power [W].
        is_lts_row: Per-step mask for `v0.1.1+` or legacy `v0.1.0` LTS rows.
        missing_prim_th: Per-step mask where primary thermal power is missing.
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
    p_wp_raw_el_w: list[float] = dataclasses.field(default_factory=list)
    is_lts_row: list[bool] = dataclasses.field(default_factory=list)
    missing_prim_th: list[bool] = dataclasses.field(default_factory=list)
    _tape_wk: PrecomputedZoneTape | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    _tape_kt: PrecomputedZoneTape | None = dataclasses.field(
        default=None, repr=False, compare=False
    )


def detect_lts_and_missing_prim_rows(
    columns: Mapping[str, Sequence[float | int]],
) -> tuple[list[bool], list[bool]]:
    """Detects `v0.1.1+` / legacy `v0.1.0` LTS rows and missing `p_prim_th_w`.

    Args:
        columns: RingBuffer column sequences.

    Returns:
        A tuple `(is_lts_row, missing_prim_th)` of equal-length boolean lists.
    """
    if "ts_utc" in columns:
        n = len(columns["ts_utc"])
    elif columns:
        n = min(len(v) for v in columns.values())
    else:
        return [], []
    if n == 0:
        return [], []

    is_lts_row: list[bool] = [False] * n
    flags_seq = columns.get("flags")
    if flags_seq is not None:
        lim_f = min(n, len(flags_seq))
        for k in range(lim_f):
            fv = flags_seq[k]
            if fv is not None and (int(fv) & const.FLAG_LTS_BACKFILL) != 0:
                is_lts_row[k] = True

    ts_seq = columns.get("ts_utc")
    if ts_seq is not None and n >= 13:
        wea_seq = columns.get("t_out_weather_c")
        valve_seqs = [
            columns[c]
            for c in _LEGACY_LTS_UPSTAIRS_VALVE_COLS
            if c in columns
        ]
        indep_seqs: list[Sequence[float | int]] = []
        for ch_name in _LEGACY_LTS_INDEP_CHANNELS:
            if ch_name in columns:
                indep_seqs.append(columns[ch_name])
            elif ch_name == "t_air_kantoor_c" and "t_air_kt_c" in columns:
                indep_seqs.append(columns["t_air_kt_c"])

        detected_k0: set[int] = set()
        flat_candidate_k0: set[int] = set()

        for k0 in range(0, n - 12):
            ts0 = int(ts_seq[k0])
            if ts0 % 3600 != 0:
                continue
            grid_ok = True
            for j in range(1, 13):
                if int(ts_seq[k0 + j]) - int(ts_seq[k0 + j - 1]) != 300:
                    grid_ok = False
                    break
            if not grid_ok:
                continue

            abc_ok = True
            if flags_seq is not None:
                for m in range(k0, k0 + 13):
                    if m < len(flags_seq) and flags_seq[m] is not None:
                        if (int(flags_seq[m]) & _LEGACY_LTS_ACCUM_FLAGS) != 0:
                            abc_ok = False
                            break
            if abc_ok and wea_seq is not None:
                for m in range(k0, k0 + 13):
                    if m < len(wea_seq) and wea_seq[m] is not None:
                        if math.isfinite(float(wea_seq[m])):
                            abc_ok = False
                            break
            if abc_ok:
                for vseq in valve_seqs:
                    for m in range(k0, k0 + 13):
                        if m >= len(vseq) or vseq[m] is None:
                            abc_ok = False
                            break
                        vf = float(vseq[m])
                        if not math.isfinite(vf) or abs(vf) > 1e-9:
                            abc_ok = False
                            break
                    if not abc_ok:
                        break
            if not abc_ok:
                continue

            n_changing = 0
            all_changing_linear = True
            for yseq in indep_seqs:
                if k0 + 12 >= len(yseq):
                    continue
                y0_raw = yseq[k0]
                y12_raw = yseq[k0 + 12]
                if y0_raw is None or y12_raw is None:
                    continue
                y0 = float(y0_raw)
                y12 = float(y12_raw)
                if not (math.isfinite(y0) and math.isfinite(y12)):
                    continue
                if abs(y12 - y0) > 1e-3:
                    n_changing += 1
                    vals_13 = [0.0] * 13
                    valid_13 = True
                    for j in range(13):
                        vj_raw = yseq[k0 + j]
                        if vj_raw is None:
                            valid_13 = False
                            break
                        vj = float(vj_raw)
                        if not math.isfinite(vj):
                            valid_13 = False
                            break
                        vals_13[j] = vj
                    if not valid_13:
                        all_changing_linear = False
                        break
                    tol = max(
                        1e-5,
                        4e-6 * max(1.0, abs(vals_13[0]), abs(vals_13[12])),
                    )
                    for j in range(1, 12):
                        sec_diff = abs(
                            vals_13[j + 1] - 2.0 * vals_13[j] + vals_13[j - 1]
                        )
                        if sec_diff >= tol:
                            all_changing_linear = False
                            break
                    if not all_changing_linear:
                        break

            if n_changing >= 2 and all_changing_linear:
                detected_k0.add(k0)
            elif all_changing_linear:
                flat_candidate_k0.add(k0)

        bridged_k0: set[int] = set()
        visited_flat: set[int] = set()
        for k0 in sorted(flat_candidate_k0):
            if k0 in visited_flat:
                continue
            run: list[int] = []
            cur = k0
            while cur in flat_candidate_k0:
                visited_flat.add(cur)
                run.append(cur)
                cur += 12
            if (k0 - 12) in detected_k0 and cur in detected_k0:
                bridged_k0.update(run)

        for k0 in detected_k0 | bridged_k0:
            for m in range(k0, k0 + 13):
                is_lts_row[m] = True

    p_prim_seq = columns.get("p_prim_th_w")
    has_both_wp_temps = ("t_wp_out_c" in columns) and ("t_wp_in_c" in columns)
    wp_out_seq = columns.get("t_wp_out_c") if has_both_wp_temps else None
    wp_in_seq = columns.get("t_wp_in_c") if has_both_wp_temps else None

    missing_prim_th: list[bool] = [False] * n
    for k in range(n):
        if is_lts_row[k]:
            missing_prim_th[k] = True
            continue
        if p_prim_seq is None or k >= len(p_prim_seq):
            missing_prim_th[k] = True
            continue
        pv = p_prim_seq[k]
        if pv is None or not math.isfinite(float(pv)):
            missing_prim_th[k] = True
            continue
        if wp_out_seq is not None and wp_in_seq is not None:
            if k >= len(wp_out_seq) or k >= len(wp_in_seq):
                missing_prim_th[k] = True
                continue
            vo = wp_out_seq[k]
            vi = wp_in_seq[k]
            if (
                vo is None
                or vi is None
                or not math.isfinite(float(vo))
                or not math.isfinite(float(vi))
            ):
                missing_prim_th[k] = True

    return is_lts_row, missing_prim_th


def compute_train_holdout_split(
    columns: Mapping[str, Sequence[float | int]],
    *,
    is_lts_row: Sequence[bool] | None = None,
) -> int:
    """Computes raw-aware chronological `train_end` split index.

    Ensures both the training slice `[0, train_end)` and the out-of-sample
    holdout slice `[train_end, n_total)` contain genuine non-LTS Recorder rows
    whenever `>= 4` days of non-LTS telemetry are available.

    Args:
        columns: Full RingBuffer column sequences.
        is_lts_row: Optional precomputed boolean mask of LTS rows matching
            `len(columns["ts_utc"])`.

    Returns:
        Exclusive end index `train_end` for the training slice.
    """
    ts_seq = columns.get("ts_utc", ())
    n_total = len(ts_seq)
    if n_total == 0:
        return 0

    if is_lts_row is not None and len(is_lts_row) == n_total:
        is_lts = is_lts_row
    else:
        is_lts = detect_lts_and_missing_prim_rows(columns)[0]

    n_raw = 0
    first_raw_idx = n_total
    for i, lts_flag in enumerate(is_lts):
        if not lts_flag:
            if first_raw_idx == n_total:
                first_raw_idx = i
            n_raw += 1

    # Case A: >= 28 days of raw Recorder data -> standard 14-day holdout.
    if n_raw >= 28 * const.STEPS_PER_DAY:
        return n_total - EVAL_14D_STEPS

    # Case B: 4..28 days of raw data -> 65/35 split on the non-LTS tail.
    if n_raw >= 4 * const.STEPS_PER_DAY:
        holdout_raw = max(2 * const.STEPS_PER_DAY, int(round(0.35 * n_raw)))
        holdout_cap = min(
            EVAL_14D_STEPS,
            max(const.STEPS_PER_DAY, n_raw - 2 * const.STEPS_PER_DAY),
        )
        holdout_raw = min(holdout_raw, holdout_cap)
        return max(first_raw_idx + 1, n_total - holdout_raw)

    # Case C: < 4 days of raw data -> total-length fallback.
    if n_total >= 28 * const.STEPS_PER_DAY:
        return n_total - EVAL_14D_STEPS
    if n_total >= 4 * const.STEPS_PER_DAY:
        train_end = max(2 * const.STEPS_PER_DAY, int(0.70 * n_total))
        if n_total - train_end < HORIZON_6H_STEPS + 24:
            train_end = n_total - (HORIZON_6H_STEPS + 24)
        return train_end
    return n_total


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
    p_wp_raw_el_w: list[float] = [0.0] * n
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
    def _prep_col(
        key: str,
        default: float,
        alt_key: str | None = None,
    ) -> list[float]:
        seq = columns.get(key)
        if seq is None and alt_key is not None:
            seq = columns.get(alt_key)
        if seq is None:
            return [default] * n
        m_len = len(seq)
        pos_inf = math.inf
        neg_inf = -math.inf
        out = [default] * n
        lim = min(n, m_len)
        for i in range(lim):
            v = seq[i]
            if v is not None:
                fv = float(v)
                if fv == fv and fv != pos_inf and fv != neg_inf:
                    out[i] = fv
        return out

    flags_seq = columns.get("flags")
    surf_seq = columns.get("t_floor_surf_meas_wk_c")
    bg_sup_seq = columns.get("t_bg_sup_c")
    bg_ret_seq = columns.get("t_bg_ret_c")
    desk_seq = columns.get("p_desk_kt_w")
    t_dak_seq = columns.get("t_out_dak_c")
    t_wea_seq = columns.get("t_out_weather_c")
    t_hm_seq = columns.get("t_out_hm_c")
    t_out_direct_seq = columns.get("t_out_c")
    pv_seq = columns.get("p_pv_w")
    has_preproj_sol = (
        "i_sol_wk_wm2" in columns
        and "i_sol_kt_wm2" in columns
        and pv_seq is None
    )

    v_wind_raw = _prep_col("v_wind_ms", 3.0)
    pv_fc_raw = _prep_col("p_pv_forecast_w", 0.0)
    epex_raw = _prep_col("epex_eur_kwh", 0.20)
    scr_s_raw = _prep_col("screen_small_pos", 100.0)
    scr_l_raw = _prep_col("screen_large_pos", 100.0)
    i_wk_in = _prep_col("i_sol_wk_wm2", 0.0) if has_preproj_sol else []
    i_kt_in = _prep_col("i_sol_kt_wm2", 0.0) if has_preproj_sol else []
    tau_scr_in = _prep_col("tau_screen_wk", 1.0) if has_preproj_sol else []

    t_air_wk_c = _prep_col("t_air_wk_c", 19.5)
    t_air_kt_c = _prep_col("t_air_kantoor_c", 19.0, "t_air_kt_c")
    t_slaap_col = _prep_col("t_air_slaapkamer_c", 18.0)
    t_bad_col = _prep_col("t_air_badkamer_c", 18.5)
    t_werk1e_col = _prep_col("t_air_werkkamer_1e_c", 18.0)
    t_mid1e_col = _prep_col("t_air_kamer_1e_midden_c", 18.0)
    t_zuid1e_col = _prep_col("t_air_kamer_1e_zuid_c", 18.0)
    t_badbov_col = _prep_col("t_air_badkamer_boven_c", 18.0)
    t_logeer_col = _prep_col("t_air_logeerkamer_c", 17.5)

    u_bg_raw = _prep_col("u_eff_bg", 0.0, "u_eff_wk")
    u_kt_raw = _prep_col("u_eff_kantoor", 0.0, "u_eff_kt")
    u_pump_raw = _prep_col("u_pump_sec", 0.0)
    u_slaap_col = _prep_col("u_eff_slaapkamer", 0.0)
    u_bad_col = _prep_col("u_eff_badkamer", 0.0)
    u_werk1e_col = _prep_col("u_eff_werkkamer_1e", 0.0)
    u_mid1e_col = _prep_col("u_eff_kamer_1e_midden", 0.0)
    u_zuid1e_col = _prep_col("u_eff_kamer_1e_zuid", 0.0)
    u_badbov_col = _prep_col("u_eff_badkamer_boven", 0.0)
    u_logeer_col = _prep_col("u_eff_logeerkamer", 0.0)

    v_prim_raw = _prep_col("v_prim_l_min", 0.0)
    p_prim_th_w = _prep_col("p_prim_th_w", 0.0)
    f_comp_raw = _prep_col("f_comp_hz", 0.0)
    p_wp_raw_in = _prep_col("p_wp_raw_el_w", 0.0)

    has_q_fh_wk_col = "q_fh_wk_w" in columns
    has_q_fh_kt_col = "q_fh_kt_w" in columns
    q_fh_wk_in = _prep_col("q_fh_wk_w", 0.0) if has_q_fh_wk_col else []
    q_fh_kt_in = _prep_col("q_fh_kt_w", 0.0) if has_q_fh_kt_col else []

    def _prep_col_fb(
        key: str,
        fb_arr: Sequence[float],
        offset: float = 0.0,
        alt_key: str | None = None,
    ) -> list[float]:
        seq = columns.get(key)
        if seq is None and alt_key is not None:
            seq = columns.get(alt_key)
        if seq is None:
            return (
                list(fb_arr)
                if offset == 0.0
                else [v + offset for v in fb_arr]
            )
        m_len = len(seq)
        pos_inf = math.inf
        neg_inf = -math.inf
        out = [0.0] * n
        for i in range(n):
            if i < m_len:
                v = seq[i]
                if v is not None:
                    fv = float(v)
                    if fv == fv and fv != pos_inf and fv != neg_inf:
                        out[i] = fv
                        continue
            out[i] = fb_arr[i] + offset
        return out

    t_top_e_col = _prep_col("t_buf_top_end_c", 22.0, "t_buf_top_c")
    t_top_s_col = _prep_col_fb("t_buf_top_start_c", t_top_e_col)
    t_bot_e_col = _prep_col_fb(
        "t_buf_bot_end_c", t_top_e_col, -1.0, "t_buf_bot_c"
    )
    t_bot_s_col = _prep_col_fb("t_buf_bot_start_c", t_bot_e_col)

    p_ac_raw = _prep_col("p_el_ac_kt_w", 0.0)
    ac_mode_code = _prep_col("ac_mode_code", 0.0)
    pir_occ_kt = _prep_col("pir_occ_kt", 0.0)
    p_grow_raw = _prep_col("p_growlights_kt_w", 0.0)
    d_est_col_wk_w = _prep_col("d_est_wk_w", 0.0)
    d_est_col_kt_w = _prep_col("d_est_kt_w", 0.0)

    t_floor_wk_running: float = (t_air_wk_c[0] if n > 0 else 19.5) + 0.8
    t_floor_kt_running: float = (t_air_kt_c[0] if n > 0 else 19.0) + 0.5

    default_flags = (
        const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
    )
    c_wk_j = 3600.0 * house_config.DAY1_PRIORS_WK.c_floor_wh_k
    h_fh_wk = 1000.0 / house_config.DAY1_PRIORS_WK.r_fh_k_kw
    c_kt_j = 3600.0 * house_config.DAY1_PRIORS_KT.c_floor_wh_k
    h_fh_kt = 1000.0 / house_config.DAY1_PRIORS_KT.r_fh_k_kw
    buf_bot_off = house_config.BUF_BOT_SENSOR_OFFSET_C
    baseload_wk = house_config.BASELOAD_WK_W
    ac_cool_modes = (int(const.AC_MODE_COOL), int(const.AC_MODE_DRY))
    cool_flags_mask = const.FLAG_FLOOR_COOL | const.FLAG_AC_COOL_KT

    room_temps: dict[str, float | None] = {
        "wk": 19.5,
        "gang": 18.7,
        "slaapkamer": 18.0,
        "badkamer": 18.5,
        "werkkamer_1e": 18.0,
        "kamer_1e_midden": 18.0,
        "kamer_1e_zuid": 18.0,
        "kantoor": 19.0,
        "badkamer_boven": 18.0,
        "logeerkamer": 17.5,
    }
    u_valves: dict[str, float] = {
        g: 0.0 for g in const.ALL_HYDRAULIC_GROUPS
    }

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
            _opt_float(t_dak_seq[k])
            if t_dak_seq is not None and k < len(t_dak_seq)
            else None
        )
        t_wea = (
            _opt_float(t_wea_seq[k])
            if t_wea_seq is not None and k < len(t_wea_seq)
            else None
        )
        t_hm = (
            _opt_float(t_hm_seq[k])
            if t_hm_seq is not None and k < len(t_hm_seq)
            else None
        )
        if (
            t_dak is None
            and t_wea is None
            and t_hm is None
            and t_out_direct_seq is not None
            and k < len(t_out_direct_seq)
        ):
            t_dak = _opt_float(t_out_direct_seq[k])
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
        vw_k = v_wind_raw[k]
        v_wind_ms[k] = vw_k if vw_k > 0.0 else 0.0

        pv_fb_flag = bool(fl_k & const.FLAG_PV_FALLBACK)
        if has_preproj_sol:
            i_wk = i_wk_in[k]
            i_kt = i_kt_in[k]
            tau_scr = tau_scr_in[k]
            pv_fb = pv_fb_flag
        else:
            if pv_seq is not None and k < len(pv_seq):
                raw_pv = pv_seq[k]
                p_pv_k = math.nan if raw_pv is None else float(raw_pv)
            else:
                p_pv_k = 0.0
            i_wk, i_kt, tau_scr, pv_fb = solar_weather.project_solar_irradiance(
                ts_utc=ts_k,
                p_pv_w=p_pv_k,
                p_pv_forecast_w=pv_fc_raw[k],
                epex_eur_kwh=epex_raw[k],
                screen_small_pos=scr_s_raw[k],
                screen_large_pos=scr_l_raw[k],
                pv_stale=pv_fb_flag,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
            )
        i_sol_wk_wm2[k] = i_wk if i_wk > 0.0 else 0.0
        i_sol_kt_wm2[k] = i_kt if i_kt > 0.0 else 0.0
        tau_screen_wk[k] = tau_scr
        pv_fb_k = pv_fb or pv_fb_flag
        pv_fallback[k] = pv_fb_k
        if pv_fb_k:
            flags[k] |= const.FLAG_PV_FALLBACK

        t_wk = t_air_wk_c[k]
        t_kt = t_air_kt_c[k]
        room_temps["wk"] = t_wk
        room_temps["gang"] = t_wk - 0.8
        room_temps["slaapkamer"] = t_slaap_col[k]
        room_temps["badkamer"] = t_bad_col[k]
        room_temps["werkkamer_1e"] = t_werk1e_col[k]
        room_temps["kamer_1e_midden"] = t_mid1e_col[k]
        room_temps["kamer_1e_zuid"] = t_zuid1e_col[k]
        room_temps["kantoor"] = t_kt
        room_temps["badkamer_boven"] = t_badbov_col[k]
        room_temps["logeerkamer"] = t_logeer_col[k]

        u_bg = min(1.0, max(0.0, u_bg_raw[k]))
        u_kt_v = min(1.0, max(0.0, u_kt_raw[k]))
        u_pump = min(1.0, max(0.0, u_pump_raw[k]))
        u_pump_sec[k] = u_pump
        u_eff_wk[k] = u_bg
        u_eff_kt[k] = u_kt_v

        u_valves["wk"] = u_bg
        u_valves["gang"] = u_bg
        u_valves["slaapkamer"] = min(1.0, max(0.0, u_slaap_col[k]))
        u_valves["badkamer"] = min(1.0, max(0.0, u_bad_col[k]))
        u_valves["werkkamer_1e"] = min(1.0, max(0.0, u_werk1e_col[k]))
        u_valves["kamer_1e_midden"] = min(1.0, max(0.0, u_mid1e_col[k]))
        u_valves["kamer_1e_zuid"] = min(1.0, max(0.0, u_zuid1e_col[k]))
        u_valves["kantoor"] = u_kt_v
        u_valves["badkamer_boven"] = min(1.0, max(0.0, u_badbov_col[k]))
        u_valves["logeerkamer"] = min(1.0, max(0.0, u_logeer_col[k]))

        threeway_room = (fl_k & const.FLAG_DHW_TANK) == 0
        wp_defrost = bool(fl_k & const.FLAG_DEFROST_WP)
        v_prim = max(0.0, v_prim_raw[k])
        p_prim = p_prim_th_w[k]
        f_hz = max(0.0, f_comp_raw[k])
        f_comp_hz[k] = f_hz
        p_wp_raw_el_w[k] = max(0.0, p_wp_raw_in[k])

        t_top_e = t_top_e_col[k]
        t_top_s = t_top_s_col[k]
        t_bot_e = t_bot_e_col[k]
        t_bot_s = t_bot_s_col[k]
        t_buf_top_c[k] = t_top_e
        t_buf_bot_cal_c[k] = t_bot_e + buf_bot_off

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

        v_sec_tot = u_pump * (
            u_valves["wk"] * flows["wk"]
            + u_valves["gang"] * flows["gang"]
            + u_valves["slaapkamer"] * flows["slaapkamer"]
            + u_valves["badkamer"] * flows["badkamer"]
            + u_valves["werkkamer_1e"] * flows["werkkamer_1e"]
            + u_valves["kamer_1e_midden"] * flows["kamer_1e_midden"]
            + u_valves["kamer_1e_zuid"] * flows["kamer_1e_zuid"]
            + u_valves["kantoor"] * flows["kantoor"]
            + u_valves["badkamer_boven"] * flows["badkamer_boven"]
            + u_valves["logeerkamer"] * flows["logeerkamer"]
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
            has_q_fh_wk_col
            and v_prim == 0.0
            and u_pump == 0.0
            and q_fh_wk_in[k] > 0.0
        ):
            q_wk = q_fh_wk_in[k]
        if (
            has_q_fh_kt_col
            and v_prim == 0.0
            and u_pump == 0.0
            and q_fh_kt_in[k] > 0.0
        ):
            q_kt = q_fh_kt_in[k]

        q_fh_wk_hyd_w[k] = q_wk
        q_fh_kt_hyd_w[k] = q_kt
        q_sec_tot_w[k] = q_tot
        f_flow_wk[k] = hyd_res.f_flow_wk
        f_flow_kt[k] = hyd_res.f_flow_kt

        dt_k = dt_s[k]
        t_floor_wk_running = tf_wk_in + (dt_k / c_wk_j) * (
            q_wk - (tf_wk_in - t_wk) * h_fh_wk
        )
        t_floor_kt_running = tf_kt_in + (dt_k / c_kt_j) * (
            q_kt - (tf_kt_in - t_kt) * h_fh_kt
        )

        t_up_w, t_dn_k = solar_weather.compute_interfloor_boundaries(
            room_temps_c=room_temps,
            last_valid_cache=interfloor_cache,
            ts_utc=ts_k,
        )
        t_up_wk_c[k] = t_up_w
        t_down_kt_c[k] = t_dn_k
        leak_w = hyd_res.q_leak_1e_to_wk_w
        q_int_wk_w[k] = baseload_wk + (leak_w if leak_w > 0.0 else 0.0)

        p_ac = max(0.0, p_ac_raw[k])
        ac_mode = ac_mode_code[k]
        p_desk_opt = (
            _opt_float(desk_seq[k], 0.0)
            if desk_seq is not None and k < len(desk_seq)
            else None
        )
        pir_k = pir_occ_kt[k]
        p_grow = max(0.0, p_grow_raw[k])
        ac_def = bool(fl_k & const.FLAG_DEFROST_AC)

        p_el_ac_kt_w[k] = p_ac
        p_desk_kt_w[k] = p_desk_opt if p_desk_opt is not None else 0.0

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
            dt_s=dt_k,
        )
        q_int_kt_w[k] = q_int_k
        q_ac_base_kt_w[k] = q_ac_k

        if surf_seq is not None and k < len(surf_seq):
            t_surf_meas_wk_c[k] = _opt_float(surf_seq[k], 5.0)

        cooling_raw[k] = bool(fl_k & cool_flags_mask) or (
            round(ac_mode) in ac_cool_modes
        )

    cooling_or_tail: list[bool] = [False] * n
    countdown = 0
    for k in range(n):
        if cooling_raw[k]:
            countdown = COOLING_TAIL_STEPS + 1
        if countdown > 0:
            cooling_or_tail[k] = True
            countdown -= 1

    is_lts_row, missing_prim_th = detect_lts_and_missing_prim_rows(columns)

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
        p_wp_raw_el_w=p_wp_raw_el_w,
        is_lts_row=is_lts_row,
        missing_prim_th=missing_prim_th,
    )


def recompute_history_for_khx(
    base_hist: PreprocessedHistory,
    columns: Mapping[str, Sequence[float | int]],
    flow_rates_l_min: Mapping[str, float] | None = None,
    k_hx_by_group: Mapping[str, float] | None = None,
) -> PreprocessedHistory:
    """Recomputes only hydraulic fields on `base_hist` for updated `k_hx`."""
    n = base_hist.n_rows
    flows = dict(const.DEFAULT_FLOW_RATES_L_MIN)
    if flow_rates_l_min is not None:
        flows.update({k: float(v) for k, v in flow_rates_l_min.items()})

    k_hx_map = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
    if k_hx_by_group is not None:
        for group_name, val in k_hx_by_group.items():
            target_key = "kantoor" if group_name == "kt" else group_name
            k_hx_map[target_key] = float(val)

    if (
        abs(base_hist.hyd_k_hx_wk - float(k_hx_map["wk"])) <= 1e-9
        and abs(base_hist.hyd_k_hx_kt - float(k_hx_map["kantoor"])) <= 1e-9
    ):
        return base_hist

    hyd = hydraulics.HydraulicModel(
        flow_rates_l_min=flows, k_hx_by_group_w_k=k_hx_map
    )

    def _pcol(
        key: str, default: float, alt_key: str | None = None
    ) -> list[float]:
        seq = columns.get(key)
        if seq is None and alt_key is not None:
            seq = columns.get(alt_key)
        if seq is None:
            return [default] * n
        m_len = len(seq)
        pos_inf = math.inf
        neg_inf = -math.inf
        out = [default] * n
        lim = min(n, m_len)
        for i in range(lim):
            v = seq[i]
            if v is not None:
                fv = float(v)
                if fv == fv and fv != pos_inf and fv != neg_inf:
                    out[i] = fv
        return out

    def _pcol_fb(
        key: str,
        fb_arr: Sequence[float],
        offset: float = 0.0,
        alt_key: str | None = None,
    ) -> list[float]:
        seq = columns.get(key)
        if seq is None and alt_key is not None:
            seq = columns.get(alt_key)
        if seq is None:
            return (
                list(fb_arr)
                if offset == 0.0
                else [v + offset for v in fb_arr]
            )
        m_len = len(seq)
        pos_inf = math.inf
        neg_inf = -math.inf
        out = [0.0] * n
        for i in range(n):
            if i < m_len:
                v = seq[i]
                if v is not None:
                    fv = float(v)
                    if fv == fv and fv != pos_inf and fv != neg_inf:
                        out[i] = fv
                        continue
            out[i] = fb_arr[i] + offset
        return out

    t_slaap_col = _pcol("t_air_slaapkamer_c", 18.0)
    t_bad_col = _pcol("t_air_badkamer_c", 18.5)
    t_werk1e_col = _pcol("t_air_werkkamer_1e_c", 18.0)
    t_mid1e_col = _pcol("t_air_kamer_1e_midden_c", 18.0)
    t_zuid1e_col = _pcol("t_air_kamer_1e_zuid_c", 18.0)
    t_badbov_col = _pcol("t_air_badkamer_boven_c", 18.0)
    t_logeer_col = _pcol("t_air_logeerkamer_c", 17.5)

    u_slaap_col = _pcol("u_eff_slaapkamer", 0.0)
    u_bad_col = _pcol("u_eff_badkamer", 0.0)
    u_werk1e_col = _pcol("u_eff_werkkamer_1e", 0.0)
    u_mid1e_col = _pcol("u_eff_kamer_1e_midden", 0.0)
    u_zuid1e_col = _pcol("u_eff_kamer_1e_zuid", 0.0)
    u_badbov_col = _pcol("u_eff_badkamer_boven", 0.0)
    u_logeer_col = _pcol("u_eff_logeerkamer", 0.0)

    v_prim_raw = _pcol("v_prim_l_min", 0.0)
    has_q_fh_wk_col = "q_fh_wk_w" in columns
    has_q_fh_kt_col = "q_fh_kt_w" in columns
    q_fh_wk_in = _pcol("q_fh_wk_w", 0.0) if has_q_fh_wk_col else []
    q_fh_kt_in = _pcol("q_fh_kt_w", 0.0) if has_q_fh_kt_col else []

    t_top_e_col = list(base_hist.t_buf_top_c)
    t_top_s_col = _pcol_fb("t_buf_top_start_c", t_top_e_col)
    t_bot_e_col = _pcol_fb(
        "t_buf_bot_end_c", t_top_e_col, -1.0, "t_buf_bot_c"
    )
    t_bot_s_col = _pcol_fb("t_buf_bot_start_c", t_bot_e_col)
    bg_sup_seq = columns.get("t_bg_sup_c")
    bg_ret_seq = columns.get("t_bg_ret_c")

    t_sup_sec_c: list[float] = [20.0] * n
    q_fh_wk_hyd_w: list[float] = [0.0] * n
    q_fh_kt_hyd_w: list[float] = [0.0] * n
    q_sec_tot_w: list[float] = [0.0] * n
    f_flow_wk: list[float] = [0.0] * n
    f_flow_kt: list[float] = [0.0] * n
    q_int_wk_w: list[float] = [house_config.BASELOAD_WK_W] * n

    t_air_wk_c = base_hist.t_air_wk_c
    t_air_kt_c = base_hist.t_air_kt_c
    u_pump_sec = base_hist.u_pump_sec
    u_eff_wk = base_hist.u_eff_wk
    u_eff_kt = base_hist.u_eff_kt
    flags = base_hist.flags
    dt_s = base_hist.dt_s
    p_prim_th_w = base_hist.p_prim_th_w
    f_comp_hz = base_hist.f_comp_hz

    t_floor_wk_running: float = (t_air_wk_c[0] if n > 0 else 19.5) + 0.8
    t_floor_kt_running: float = (t_air_kt_c[0] if n > 0 else 19.0) + 0.5
    c_wk_j = 3600.0 * house_config.DAY1_PRIORS_WK.c_floor_wh_k
    h_fh_wk = 1000.0 / house_config.DAY1_PRIORS_WK.r_fh_k_kw
    c_kt_j = 3600.0 * house_config.DAY1_PRIORS_KT.c_floor_wh_k
    h_fh_kt = 1000.0 / house_config.DAY1_PRIORS_KT.r_fh_k_kw
    baseload_wk = house_config.BASELOAD_WK_W

    room_temps: dict[str, float | None] = {
        "wk": 19.5,
        "gang": 18.7,
        "slaapkamer": 18.0,
        "badkamer": 18.5,
        "werkkamer_1e": 18.0,
        "kamer_1e_midden": 18.0,
        "kamer_1e_zuid": 18.0,
        "kantoor": 19.0,
        "badkamer_boven": 18.0,
        "logeerkamer": 17.5,
    }
    u_valves: dict[str, float] = {
        g: 0.0 for g in const.ALL_HYDRAULIC_GROUPS
    }

    for k in range(n):
        fl_k = flags[k]
        t_wk = t_air_wk_c[k]
        t_kt = t_air_kt_c[k]
        room_temps["wk"] = t_wk
        room_temps["gang"] = t_wk - 0.8
        room_temps["slaapkamer"] = t_slaap_col[k]
        room_temps["badkamer"] = t_bad_col[k]
        room_temps["werkkamer_1e"] = t_werk1e_col[k]
        room_temps["kamer_1e_midden"] = t_mid1e_col[k]
        room_temps["kamer_1e_zuid"] = t_zuid1e_col[k]
        room_temps["kantoor"] = t_kt
        room_temps["badkamer_boven"] = t_badbov_col[k]
        room_temps["logeerkamer"] = t_logeer_col[k]

        u_bg = u_eff_wk[k]
        u_kt_v = u_eff_kt[k]
        u_pump = u_pump_sec[k]
        u_valves["wk"] = u_bg
        u_valves["gang"] = u_bg
        u_valves["slaapkamer"] = min(1.0, max(0.0, u_slaap_col[k]))
        u_valves["badkamer"] = min(1.0, max(0.0, u_bad_col[k]))
        u_valves["werkkamer_1e"] = min(1.0, max(0.0, u_werk1e_col[k]))
        u_valves["kamer_1e_midden"] = min(1.0, max(0.0, u_mid1e_col[k]))
        u_valves["kamer_1e_zuid"] = min(1.0, max(0.0, u_zuid1e_col[k]))
        u_valves["kantoor"] = u_kt_v
        u_valves["badkamer_boven"] = min(1.0, max(0.0, u_badbov_col[k]))
        u_valves["logeerkamer"] = min(1.0, max(0.0, u_logeer_col[k]))

        threeway_room = (fl_k & const.FLAG_DHW_TANK) == 0
        wp_defrost = bool(fl_k & const.FLAG_DEFROST_WP)
        v_prim = max(0.0, v_prim_raw[k])
        t_top_e = t_top_e_col[k]
        t_bot_e = t_bot_e_col[k]
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

        v_sec_tot = u_pump * (
            u_valves["wk"] * flows["wk"]
            + u_valves["gang"] * flows["gang"]
            + u_valves["slaapkamer"] * flows["slaapkamer"]
            + u_valves["badkamer"] * flows["badkamer"]
            + u_valves["werkkamer_1e"] * flows["werkkamer_1e"]
            + u_valves["kamer_1e_midden"] * flows["kamer_1e_midden"]
            + u_valves["kamer_1e_zuid"] * flows["kamer_1e_zuid"]
            + u_valves["kantoor"] * flows["kantoor"]
            + u_valves["badkamer_boven"] * flows["badkamer_boven"]
            + u_valves["logeerkamer"] * flows["logeerkamer"]
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
            p_prim_th_w=p_prim_th_w[k],
            f_comp_hz=f_comp_hz[k],
            t_buf_top_start_c=t_top_s_col[k],
            t_buf_top_end_c=t_top_e,
            t_buf_bot_start_c=t_bot_s_col[k],
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
        if (
            has_q_fh_wk_col
            and v_prim == 0.0
            and u_pump == 0.0
            and q_fh_wk_in[k] > 0.0
        ):
            q_wk = q_fh_wk_in[k]
        if (
            has_q_fh_kt_col
            and v_prim == 0.0
            and u_pump == 0.0
            and q_fh_kt_in[k] > 0.0
        ):
            q_kt = q_fh_kt_in[k]

        q_fh_wk_hyd_w[k] = q_wk
        q_fh_kt_hyd_w[k] = q_kt
        q_sec_tot_w[k] = sum(hyd_res.q_fh_by_group_w.values())
        f_flow_wk[k] = hyd_res.f_flow_wk
        f_flow_kt[k] = hyd_res.f_flow_kt

        dt_k = dt_s[k]
        t_floor_wk_running = tf_wk_in + (dt_k / c_wk_j) * (
            q_wk - (tf_wk_in - t_wk) * h_fh_wk
        )
        t_floor_kt_running = tf_kt_in + (dt_k / c_kt_j) * (
            q_kt - (tf_kt_in - t_kt) * h_fh_kt
        )
        leak_w = hyd_res.q_leak_1e_to_wk_w
        q_int_wk_w[k] = baseload_wk + (leak_w if leak_w > 0.0 else 0.0)

    cloned_tape_wk = (
        dataclasses.replace(
            base_hist._tape_wk,
            hyd_k_hx=float(k_hx_map["wk"]),
            t_sup_sec=t_sup_sec_c,
            q_fh_hyd=q_fh_wk_hyd_w,
            q_int=q_int_wk_w,
        )
        if base_hist._tape_wk is not None
        else None
    )
    cloned_tape_kt = (
        dataclasses.replace(
            base_hist._tape_kt,
            hyd_k_hx=float(k_hx_map["kantoor"]),
            t_sup_sec=t_sup_sec_c,
            q_fh_hyd=q_fh_kt_hyd_w,
        )
        if base_hist._tape_kt is not None
        else None
    )

    return PreprocessedHistory(
        n_rows=n,
        ts_utc=base_hist.ts_utc,
        flags=base_hist.flags,
        dt_s=base_hist.dt_s,
        t_out_c=base_hist.t_out_c,
        v_wind_ms=base_hist.v_wind_ms,
        i_sol_wk_wm2=base_hist.i_sol_wk_wm2,
        i_sol_kt_wm2=base_hist.i_sol_kt_wm2,
        tau_screen_wk=base_hist.tau_screen_wk,
        pv_fallback=base_hist.pv_fallback,
        t_up_wk_c=base_hist.t_up_wk_c,
        t_down_wk_c=base_hist.t_down_wk_c,
        t_down_kt_c=base_hist.t_down_kt_c,
        q_int_wk_w=q_int_wk_w,
        q_int_kt_w=base_hist.q_int_kt_w,
        q_ac_base_kt_w=base_hist.q_ac_base_kt_w,
        p_el_ac_kt_w=base_hist.p_el_ac_kt_w,
        ac_mode_code=base_hist.ac_mode_code,
        p_desk_kt_w=base_hist.p_desk_kt_w,
        pir_occ_kt=base_hist.pir_occ_kt,
        u_pump_sec=base_hist.u_pump_sec,
        u_eff_wk=base_hist.u_eff_wk,
        u_eff_kt=base_hist.u_eff_kt,
        t_buf_top_c=base_hist.t_buf_top_c,
        t_buf_bot_cal_c=base_hist.t_buf_bot_cal_c,
        t_sup_sec_c=t_sup_sec_c,
        f_comp_hz=base_hist.f_comp_hz,
        p_prim_th_w=base_hist.p_prim_th_w,
        q_fh_wk_hyd_w=q_fh_wk_hyd_w,
        q_fh_kt_hyd_w=q_fh_kt_hyd_w,
        q_sec_tot_w=q_sec_tot_w,
        f_flow_wk=f_flow_wk,
        f_flow_kt=f_flow_kt,
        t_air_wk_c=base_hist.t_air_wk_c,
        t_air_kt_c=base_hist.t_air_kt_c,
        t_surf_meas_wk_c=base_hist.t_surf_meas_wk_c,
        d_est_col_wk_w=base_hist.d_est_col_wk_w,
        d_est_col_kt_w=base_hist.d_est_col_kt_w,
        cooling_or_tail=base_hist.cooling_or_tail,
        hyd_k_hx_wk=float(k_hx_map["wk"]),
        hyd_k_hx_kt=float(k_hx_map["kantoor"]),
        p_wp_raw_el_w=base_hist.p_wp_raw_el_w,
        is_lts_row=base_hist.is_lts_row,
        missing_prim_th=base_hist.missing_prim_th,
        _tape_wk=cloned_tape_wk,
        _tape_kt=cloned_tape_kt,
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


@dataclasses.dataclass(slots=True)
class PrecomputedZoneTape:
    """Parameter-independent 1D exogenous & sensor tape for fast calibration."""

    zone: contracts.ZoneId
    n_rows: int
    hyd_k_hx: float
    dt_s: list[float]
    ts_utc: list[int]
    t_meas: list[float]
    valid: list[bool]
    temp_changed: list[bool]
    open_win: list[bool]
    defrost_ac: list[bool]
    ac_heat: list[bool]
    u_pump: list[float]
    u_zone: list[float]
    u_prod: list[float]
    t_sup_sec: list[float]
    t_buf_top: list[float]
    q_fh_hyd: list[float]
    is_harvest: list[bool]
    p_el_ac: list[float]
    q_ac_base: list[float]
    ac_net_el: list[float]
    ac_temp_factor: list[float]
    ac_eff_el: list[float]
    q_ac_fixed: list[float]
    t_out: list[float]
    v_excess: list[float]
    i_sol: list[float]
    t_down: list[float]
    q_int: list[float]
    t_up: list[float]
    sigma_d_w: list[float]
    sigma_d_sq2: list[float]
    qc11_pump_base: list[float]
    r_v: list[float]
    obs_slope: list[float]
    meas_mode: list[int]
    y_obs: list[float]
    cold_cap_active: list[bool]
    t_surf_meas: list[float | None]
    has_surf_meas: bool
    fast_300_nowind: list[bool] = dataclasses.field(default_factory=list)
    fast_300_anywind: list[bool] = dataclasses.field(default_factory=list)


def _cooling_rate_from_history(
    air_obs_history: collections.deque[tuple[int, float]],
    ts_utc: int,
    t_meas: float,
) -> float:
    """Computes 15-min (`900 s`) finite-difference air cooling rate [K/h]."""
    if not air_obs_history:
        return 0.0
    target_ts = ts_utc - 900
    best_ts, best_temp = air_obs_history[0]
    best_diff = abs(best_ts - target_ts)
    for h_ts, h_temp in air_obs_history:
        if h_ts >= ts_utc:
            continue
        diff = abs(h_ts - target_ts)
        if diff <= best_diff:
            best_diff = diff
            best_ts, best_temp = h_ts, h_temp

    dt_hist_s = ts_utc - best_ts
    if dt_hist_s <= 0 or dt_hist_s > 1800:
        return 0.0
    denom_h = max(0.25, dt_hist_s / 3600.0)
    return (t_meas - best_temp) / denom_h


def build_zone_tape(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
) -> PrecomputedZoneTape:
    """Precomputes 1D parameter-independent arrays for fast LM calibration."""
    n = hist.n_rows
    is_wk = zone == "wk"
    valid_flag = const.FLAG_VALID_WK if is_wk else const.FLAG_VALID_KT
    open_win_flag = const.FLAG_OPEN_WIN_WK if is_wk else const.FLAG_OPEN_WIN_KT
    hyd_k_hx = hist.hyd_k_hx_wk if is_wk else hist.hyd_k_hx_kt

    dt_s = list(hist.dt_s)
    ts_utc = list(hist.ts_utc)
    t_meas = list(hist.t_air_wk_c if is_wk else hist.t_air_kt_c)
    u_pump = list(hist.u_pump_sec)
    u_zone = list(hist.u_eff_wk if is_wk else hist.u_eff_kt)
    t_sup_sec = list(hist.t_sup_sec_c)
    t_buf_top = list(hist.t_buf_top_c)
    q_fh_hyd = list(hist.q_fh_wk_hyd_w if is_wk else hist.q_fh_kt_hyd_w)
    p_el_ac = list(hist.p_el_ac_kt_w)
    q_ac_base = list(hist.q_ac_base_kt_w)
    t_out = list(hist.t_out_c)
    v_excess = [max(0.0, v - 3.0) for v in hist.v_wind_ms]
    i_sol = list(hist.i_sol_wk_wm2 if is_wk else hist.i_sol_kt_wm2)
    t_down = list(hist.t_down_wk_c if is_wk else hist.t_down_kt_c)
    q_int = list(hist.q_int_wk_w if is_wk else hist.q_int_kt_w)
    t_up = list(hist.t_up_wk_c) if is_wk else [0.0] * n
    t_surf_meas: list[float | None] = (
        list(hist.t_surf_meas_wk_c) if is_wk else [None] * n
    )
    has_surf_meas = is_wk and any(v is not None for v in t_surf_meas)

    valid: list[bool] = [False] * n
    temp_changed: list[bool] = [True] * n
    open_win: list[bool] = [False] * n
    defrost_ac: list[bool] = [False] * n
    ac_heat: list[bool] = [False] * n
    u_prod: list[float] = [0.0] * n
    is_harvest: list[bool] = [False] * n
    ac_net_el: list[float] = [0.0] * n
    ac_temp_factor: list[float] = [1.0] * n
    ac_eff_el: list[float] = [0.0] * n
    q_ac_fixed: list[float] = [0.0] * n
    sigma_d_w: list[float] = [0.0] * n
    sigma_d_sq2: list[float] = [0.0] * n
    qc11_pump_base: list[float] = [0.0] * n
    r_v: list[float] = [zoh_kalman.R_MEAS_COARSE_K2] * n
    obs_slope: list[float] = [0.0] * n
    meas_mode: list[int] = [0] * n
    y_obs: list[float] = [0.0] * n
    cold_cap_active: list[bool] = [False] * n

    default_sigma = 110.0 if is_wk else 75.0
    q_idle = zoh_kalman.Q_FLOOR_IDLE_K2_S
    q_pump_span = zoh_kalman.Q_FLOOR_PUMP_K2_S - q_idle
    harvest_mask = const.FLAG_DHW_TANK | const.FLAG_Z1_OFF_HARVEST
    ac_mode_heat_int = int(const.AC_MODE_HEAT)
    tz_name = house_config.TIMEZONE_NAME
    last_hr_key = -1
    cached_hr_base = 0

    for k in range(n):
        fl_k = hist.flags[k]
        val_k = bool(fl_k & valid_flag)
        valid[k] = val_k
        open_win[k] = bool(fl_k & open_win_flag)
        if k > 0:
            temp_changed[k] = abs(t_meas[k] - t_meas[k - 1]) >= 1e-4

        up_k = u_pump[k]
        uz_k = u_zone[k]
        if up_k > 1e-4 and uz_k > 1e-4:
            u_prod[k] = up_k * uz_k
            is_harvest[k] = bool(fl_k & harvest_mask) or (
                hist.f_comp_hz[k] <= 0.5 and hist.p_prim_th_w[k] <= 50.0
            )
        u_p_clamped = max(0.0, min(1.0, up_k))
        qc11_pump_base[k] = q_idle + u_p_clamped * q_pump_span

        if not is_wk:
            def_k = bool(fl_k & const.FLAG_DEFROST_AC)
            defrost_ac[k] = def_k
            pac_k = p_el_ac[k]
            qbase_k = q_ac_base[k]
            am_k = hist.ac_mode_code[k]
            if def_k:
                q_ac_fixed[k] = (
                    qbase_k
                    if qbase_k < 0.0
                    else -min(200.0, 50.0 + 0.15 * max(0.0, pac_k - 3.0))
                )
            elif (
                math.isfinite(am_k)
                and round(am_k) == ac_mode_heat_int
                and pac_k > 25.0
            ):
                ac_heat[k] = True
                net_el_k = max(0.0, pac_k - 3.0)
                tf_k = min(1.35, max(0.60, 1.0 + 0.025 * (t_out[k] - 7.0)))
                ac_net_el[k] = net_el_k
                ac_temp_factor[k] = tf_k
                ac_eff_el[k] = net_el_k * tf_k
            else:
                q_ac_fixed[k] = qbase_k

        aanwezig_k = bool(fl_k & const.FLAG_AANWEZIG)
        ts_k = ts_utc[k]
        hr_key = ts_k // 3600
        if hr_key != last_hr_key:
            cached_hr_base, _ = solar_weather.local_hour_minute(
                hr_key * 3600, tz_name
            )
            last_hr_key = hr_key
        mins = cached_hr_base * 60 + ((ts_k % 3600) // 60)
        if not aanwezig_k or 30 <= mins < 390:
            sd = 20.0
        elif is_wk and 1020 <= mins < 1230:
            sd = 280.0
        else:
            sd = default_sigma
        sigma_d_w[k] = sd
        sigma_d_sq2[k] = 2.0 * sd * sd

    if n > 0:
        det = zoh_kalman.ResolutionDetector()
        det.observe(t_meas[0])
        r_v[0] = det.r_meas_k2
        meas_mode[0] = 4
        y_obs[0] = t_meas[0]
        last_meas_temp_c: float | None = t_meas[0]
        time_since_change = 0.0
        air_obs_history: collections.deque[tuple[int, float]] = (
            collections.deque(maxlen=12)
        )
        cold_rem = (
            0.0 if u_pump[0] <= 0.5 else zoh_kalman.COLD_START_CAP_DURATION_S
        )

        for k in range(1, n):
            dt_k = dt_s[k]
            if dt_k > 10800.0:
                det.observe(t_meas[k])
                r_v[k] = det.r_meas_k2
                meas_mode[k] = 4
                y_obs[k] = t_meas[k]
                last_meas_temp_c = t_meas[k]
                time_since_change = 0.0
                air_obs_history.clear()
                cold_rem = (
                    0.0
                    if u_pump[k] <= 0.5
                    else zoh_kalman.COLD_START_CAP_DURATION_S
                )
                continue

            step_s = max(1.0, dt_k)
            cold_cap_active[k] = cold_rem > 0.0
            if not air_obs_history and last_meas_temp_c is not None:
                air_obs_history.append(
                    (ts_utc[k] - int(round(step_s)), last_meas_temp_c)
                )

            if valid[k]:
                y_m = t_meas[k]
                if temp_changed[k]:
                    time_since_change = 0.0
                    last_meas_temp_c = y_m
                    det.observe(y_m)
                    r_v[k] = det.r_meas_k2
                    obs_slope[k] = _cooling_rate_from_history(
                        air_obs_history, ts_utc[k], y_m
                    )
                    meas_mode[k] = 1
                    y_obs[k] = y_m
                else:
                    time_since_change += step_s
                    y_ref = (
                        last_meas_temp_c
                        if last_meas_temp_c is not None
                        else y_m
                    )
                    y_obs[k] = y_ref
                    if not det.is_fine_resolution:
                        meas_mode[k] = 2
                        r_v[k] = det.r_meas_k2
                    elif (k % 6 == 0) and time_since_change >= 1800.0:
                        meas_mode[k] = 3
                        r_v[k] = zoh_kalman.R_MEAS_HEARTBEAT_FINE_K2
                    else:
                        meas_mode[k] = 0
                        r_v[k] = det.r_meas_k2
                air_obs_history.append((ts_utc[k], y_m))
            else:
                time_since_change += step_s
                meas_mode[k] = 0
                r_v[k] = det.r_meas_k2

            if cold_rem > 0.0:
                cold_rem = max(0.0, cold_rem - step_s)

    ow_thr = zoh_kalman.OPEN_WINDOW_SLOPE_K_PER_H
    fast_300_anywind = [
        (
            dt_s[k] == 300.0
            and meas_mode[k] == 1
            and obs_slope[k] >= ow_thr
            and not cold_cap_active[k]
            and not has_surf_meas
            and not is_harvest[k]
        )
        for k in range(n)
    ]
    fast_300_nowind = [
        fast_300_anywind[k] and v_excess[k] == 0.0 for k in range(n)
    ]

    return PrecomputedZoneTape(
        zone=zone,
        n_rows=n,
        hyd_k_hx=hyd_k_hx,
        dt_s=dt_s,
        ts_utc=ts_utc,
        t_meas=t_meas,
        valid=valid,
        temp_changed=temp_changed,
        open_win=open_win,
        defrost_ac=defrost_ac,
        ac_heat=ac_heat,
        u_pump=u_pump,
        u_zone=u_zone,
        u_prod=u_prod,
        t_sup_sec=t_sup_sec,
        t_buf_top=t_buf_top,
        q_fh_hyd=q_fh_hyd,
        is_harvest=is_harvest,
        p_el_ac=p_el_ac,
        q_ac_base=q_ac_base,
        ac_net_el=ac_net_el,
        ac_temp_factor=ac_temp_factor,
        ac_eff_el=ac_eff_el,
        q_ac_fixed=q_ac_fixed,
        t_out=t_out,
        v_excess=v_excess,
        i_sol=i_sol,
        t_down=t_down,
        q_int=q_int,
        t_up=t_up,
        sigma_d_w=sigma_d_w,
        sigma_d_sq2=sigma_d_sq2,
        qc11_pump_base=qc11_pump_base,
        r_v=r_v,
        obs_slope=obs_slope,
        meas_mode=meas_mode,
        y_obs=y_obs,
        cold_cap_active=cold_cap_active,
        t_surf_meas=t_surf_meas,
        has_surf_meas=has_surf_meas,
        fast_300_nowind=fast_300_nowind,
        fast_300_anywind=fast_300_anywind,
    )


def _cold_start_scalars(
    params: contracts.ZoneParameters,
    t_air: float,
    t_out: float,
    v_excess: float,
    i_sol: float,
    t_down: float,
    q_int: float,
    t_up: float,
    q_fh_w: float,
    q_ac_w: float,
    u_pump: float,
    t_buf_top_c: float,
    r_v0: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Computes scalar `(x0, x1, x2, p00..p22)` matching `cold_start`."""
    h_fh = 1000.0 / params.r_fh_k_kw
    v_eff = 0.0 if params.c_wind == 0.0 else v_excess
    h_oa = params.h_loss_w_k * (1.0 + params.c_wind * v_eff)
    h_up = params.h_up_w_k
    h_down = params.h_down_w_k

    q_air_loss = (
        h_oa * (t_air - t_out)
        + h_up * (t_air - t_up)
        - q_int
        - q_ac_w
        - params.alpha_sol * params.a_sol_m2 * i_sol
    )
    c_tot_wh_k = params.c_air_wh_k + params.c_floor_wh_k
    q_tot_net = (
        q_air_loss
        + h_down * (t_air - t_down)
        - (1.0 - params.alpha_sol) * params.a_sol_m2 * i_sol
    )
    c_air_dot_t_slow = -(params.c_air_wh_k / c_tot_wh_k) * q_tot_net
    t_floor_passive = t_air + (q_air_loss + c_air_dot_t_slow) / h_fh

    if u_pump <= 0.5:
        t_floor_0 = t_floor_passive
        p11_0 = 0.25
    else:
        q_sol_floor = (1.0 - params.alpha_sol) * params.a_sol_m2 * i_sol
        if math.isfinite(t_buf_top_c):
            k_hx = params.k_hx_w_k * min(1.0, u_pump)
            t_floor_pipe = (
                k_hx * t_buf_top_c
                + h_fh * t_air
                + h_down * t_down
                + q_sol_floor
            ) / (k_hx + h_fh + h_down)
        else:
            t_floor_pipe = (
                q_fh_w + h_fh * t_air + h_down * t_down + q_sol_floor
            ) / (h_fh + h_down)
        t_floor_0 = 0.5 * (t_floor_passive + t_floor_pipe)
        p11_0 = 1.0

    return t_air, t_floor_0, 0.0, r_v0, 0.0, 0.0, p11_0, 0.0, 6400.0


def _run_kalman_on_tape(
    tape: PrecomputedZoneTape,
    params: contracts.ZoneParameters,
    *,
    end_k: int,
    save_k0_set: set[int] | None = None,
    record_all: bool = False,
) -> tuple[
    list[tuple[float, float, float]] | None,
    dict[int, tuple[float, float]] | None,
    zoh_kalman._Mat3,
    zoh_kalman._Mat3x7,
    float,
    float,
]:
    """Runs unrolled scalar Kalman filter on `tape` up to `end_k` inclusive."""
    limit_k = min(end_k, tape.n_rows - 1)
    states_out: list[tuple[float, float, float]] | None = (
        [(0.0, 0.0, 0.0)] * (limit_k + 1) if record_all else None
    )
    saved_k0: dict[int, tuple[float, float]] | None = (
        {} if save_k0_set is not None else None
    )
    save_mask: list[bool] | None = None
    if save_k0_set is not None:
        save_mask = [False] * (limit_k + 1)
        for kk in save_k0_set:
            if 0 <= kk <= limit_k:
                save_mask[kk] = True

    is_kt = tape.zone == "kt"
    khx = params.k_hx_w_k
    base_khx = tape.hyd_k_hx
    khx_ratio = (khx / base_khx) if base_khx > 1e-6 else 1.0
    khx_scale_act = (
        khx_ratio
        if (base_khx > 1e-6 and abs(khx - base_khx) > 1e-6)
        else 1.0
    )
    eta_ac = params.eta_ac
    c_wind = params.c_wind
    c_floor_j = 3600.0 * params.c_floor_wh_k
    qc11_qfh_factor = (
        (0.08 * 0.08) * zoh_kalman.TAU_HYDR_S / (c_floor_j * c_floor_j)
    )
    inv_tau_d = 1.0 / (3600.0 * params.tau_d_h)
    inv_tau_d_vent = 1.0 / zoh_kalman.TAU_D_VENT_S

    # Unpack standard 300s zero-wind ZOH + Q_d basis once outside the k-loop.
    ad_std, bd_std, _, _, _ = zoh_kalman._discretize_ab_cached(
        params, 300.0, 0.0
    )
    g0_std, g1_std, g2_std = zoh_kalman._discretize_qd_basis_cached(
        params, 300.0, 0.0
    )
    a00, a01, a02 = ad_std[0][0], ad_std[0][1], ad_std[0][2]
    a10, a11, a12 = ad_std[1][0], ad_std[1][1], ad_std[1][2]
    a22 = ad_std[2][2]
    b00, b01, b02, b03, b04, b05, b06 = bd_std[0]
    b10, b11, b12, b13, b14, b15, b16 = bd_std[1]
    b06_eta = b06 * eta_ac
    b16_eta = b16 * eta_ac

    g0_00, g0_01, g0_02 = g0_std[0][0], g0_std[0][1], g0_std[0][2]
    g0_11, g0_12, g0_22 = g0_std[1][1], g0_std[1][2], g0_std[2][2]
    g1_00, g1_01, g1_02 = g1_std[0][0], g1_std[0][1], g1_std[0][2]
    g1_11, g1_12, g1_22 = g1_std[1][1], g1_std[1][2], g1_std[2][2]
    g2_00, g2_01, g2_02 = g2_std[0][0], g2_std[0][1], g2_std[0][2]
    g2_11, g2_12, g2_22 = g2_std[1][1], g2_std[1][2], g2_std[2][2]

    dt_arr = tape.dt_s
    t_meas_arr = tape.t_meas
    u_pump_arr = tape.u_pump
    u_prod_arr = tape.u_prod
    t_buf_top_arr = tape.t_buf_top
    q_fh_hyd_arr = tape.q_fh_hyd
    is_harvest_arr = tape.is_harvest
    ac_eff_arr = tape.ac_eff_el
    q_ac_fixed_arr = tape.q_ac_fixed
    t_out_arr = tape.t_out
    v_exc_arr = tape.v_excess
    i_sol_arr = tape.i_sol
    t_down_arr = tape.t_down
    q_int_arr = tape.q_int
    t_up_arr = tape.t_up
    sd_sq2_arr = tape.sigma_d_sq2
    qc11_base_arr = tape.qc11_pump_base
    r_v_arr = tape.r_v
    obs_slope_arr = tape.obs_slope
    meas_mode_arr = tape.meas_mode
    y_obs_arr = tape.y_obs
    cold_cap_arr = tape.cold_cap_active
    has_surf = tape.has_surf_meas
    t_surf_arr = tape.t_surf_meas

    surf_gamma = 0.60
    if has_surf:
        area_m2 = (
            house_config.AREA_WK_M2
            if tape.zone == "wk"
            else house_config.AREA_KT_M2
        )
        r_surf_k_kw = 1000.0 / (house_config.H_SURF_UPWARD_W_M2K * area_m2)
        surf_gamma = max(0.10, min(0.95, r_surf_k_kw / params.r_fh_k_kw))

    q_air_base = zoh_kalman.Q_AIR_BASE_K2_S
    q_air_ow = zoh_kalman.Q_AIR_OPEN_WINDOW_K2_S
    qg0_00 = q_air_base * g0_00
    qg0_01 = q_air_base * g0_01
    qg0_02 = q_air_base * g0_02
    qg0_11 = q_air_base * g0_11
    qg0_12 = q_air_base * g0_12
    qg0_22 = q_air_base * g0_22
    fast_arr = (
        tape.fast_300_anywind if c_wind == 0.0 else tape.fast_300_nowind
    )
    ow_slope_thr = zoh_kalman.OPEN_WINDOW_SLOPE_K_PER_H
    ow_hold_const = zoh_kalman.OPEN_WINDOW_HOLD_S
    floor_cap_k = zoh_kalman.COLD_START_FLOOR_CAP_K
    vent_params: contracts.ZoneParameters | None = None

    # k = 0 cold start
    q_ac_0 = (ac_eff_arr[0] * eta_ac + q_ac_fixed_arr[0]) if is_kt else 0.0
    q_fh_0 = q_fh_hyd_arr[0]
    x0, x1, x2, p00, p01, p02, p11, p12, p22 = _cold_start_scalars(
        params,
        t_meas_arr[0],
        t_out_arr[0],
        v_exc_arr[0],
        i_sol_arr[0],
        t_down_arr[0],
        q_int_arr[0],
        t_up_arr[0],
        q_fh_0,
        q_ac_0,
        u_pump_arr[0],
        t_buf_top_arr[0],
        r_v_arr[0],
    )
    open_window_hold_rem_s = 0.0
    open_window_active = False
    if states_out is not None:
        states_out[0] = (x0, x1, x2)
    if save_mask is not None and saved_k0 is not None and save_mask[0]:
        saved_k0[0] = (x0, x1)

    a22_sq = a22 * a22
    u4_fast: Sequence[float] = (
        q_fh_hyd_arr
        if (khx_scale_act == 1.0 and abs(khx_ratio - 1.0) <= 1e-9)
        else [
            ((qh * khx_ratio) if qh > 0.0 else 0.0)
            if up == 0.0
            else (qh * khx_scale_act)
            for up, qh in zip(u_prod_arr, q_fh_hyd_arr)
        ]
    )

    for k in range(1, limit_k + 1):
        if fast_arr[k] and open_window_hold_rem_s == 0.0 and x2 >= -150.0:
            u4 = u4_fast[k]
            if is_kt:
                ac_e = ac_eff_arr[k]
                ac_f = q_ac_fixed_arr[k]
                xm0 = (
                    a00 * x0 + a01 * x1 + a02 * x2 + b00 * t_out_arr[k]
                    + b01 * t_up_arr[k] + b02 * t_down_arr[k]
                    + b03 * i_sol_arr[k] + b04 * u4 + b05 * q_int_arr[k]
                    + b06_eta * ac_e + b06 * ac_f
                )
                xm1 = (
                    a10 * x0 + a11 * x1 + a12 * x2 + b10 * t_out_arr[k]
                    + b11 * t_up_arr[k] + b12 * t_down_arr[k]
                    + b13 * i_sol_arr[k] + b14 * u4 + b15 * q_int_arr[k]
                    + b16_eta * ac_e + b16 * ac_f
                )
            else:
                xm0 = (
                    a00 * x0 + a01 * x1 + a02 * x2 + b00 * t_out_arr[k]
                    + b01 * t_up_arr[k] + b02 * t_down_arr[k]
                    + b03 * i_sol_arr[k] + b04 * u4 + b05 * q_int_arr[k]
                )
                xm1 = (
                    a10 * x0 + a11 * x1 + a12 * x2 + b10 * t_out_arr[k]
                    + b11 * t_up_arr[k] + b12 * t_down_arr[k]
                    + b13 * i_sol_arr[k] + b14 * u4 + b15 * q_int_arr[k]
                )
            xm2 = a22 * x2
            qc11 = qc11_base_arr[k] + qc11_qfh_factor * u4 * u4
            qc22 = sd_sq2_arr[k] * inv_tau_d
            ap00 = a00 * p00 + a01 * p01 + a02 * p02
            ap01 = a00 * p01 + a01 * p11 + a02 * p12
            ap02 = a00 * p02 + a01 * p12 + a02 * p22
            ap10 = a10 * p00 + a11 * p01 + a12 * p02
            ap11 = a10 * p01 + a11 * p11 + a12 * p12
            ap12 = a10 * p02 + a11 * p12 + a12 * p22
            pm00 = ap00 * a00 + ap01 * a01 + ap02 * a02 + (
                qg0_00 + qc11 * g1_00 + qc22 * g2_00
            )
            pm01 = ap00 * a10 + ap01 * a11 + ap02 * a12 + (
                qg0_01 + qc11 * g1_01 + qc22 * g2_01
            )
            pm02 = ap02 * a22 + (qg0_02 + qc11 * g1_02 + qc22 * g2_02)
            pm11 = ap10 * a10 + ap11 * a11 + ap12 * a12 + (
                qg0_11 + qc11 * g1_11 + qc22 * g2_11
            )
            pm12 = ap12 * a22 + (qg0_12 + qc11 * g1_12 + qc22 * g2_12)
            pm22 = a22_sq * p22 + (qg0_22 + qc11 * g1_22 + qc22 * g2_22)
            rv_eff = r_v_arr[k]
            e_innov = y_obs_arr[k] - xm0
            inv_s = 1.0 / (pm00 + rv_eff)
            gk0 = pm00 * inv_s
            gk1 = pm01 * inv_s
            gk2 = pm02 * inv_s
            x0 = xm0 + gk0 * e_innov
            x1 = xm1 + gk1 * e_innov
            x2 = xm2 + gk2 * e_innov
            p00 = rv_eff * gk0
            p01 = rv_eff * gk1
            p02 = rv_eff * gk2
            p11 = pm11 - gk1 * pm01
            p12 = pm12 - gk1 * pm02
            p22 = pm22 - gk2 * pm02
            if states_out is not None:
                states_out[k] = (x0, x1, x2)
            if save_mask is not None and saved_k0 is not None and save_mask[k]:
                saved_k0[k] = (x0, x1)
            continue

        up_uz = u_prod_arr[k]
        if up_uz == 0.0:
            qh = q_fh_hyd_arr[k]
            q_fh_k = (qh * khx_ratio) if qh > 0.0 else 0.0
        elif is_harvest_arr[k]:
            q_fh_k = up_uz * khx * (t_buf_top_arr[k] - x1)
            if q_fh_k < 0.0:
                q_fh_k = 0.0
        else:
            q_fh_k = q_fh_hyd_arr[k] * khx_scale_act

        q_ac_k = (
            (ac_eff_arr[k] * eta_ac + q_ac_fixed_arr[k]) if is_kt else 0.0
        )

        mm = meas_mode_arr[k]
        if mm == 4:
            x0, x1, x2, p00, p01, p02, p11, p12, p22 = _cold_start_scalars(
                params,
                t_meas_arr[k],
                t_out_arr[k],
                v_exc_arr[k],
                i_sol_arr[k],
                t_down_arr[k],
                q_int_arr[k],
                t_up_arr[k],
                q_fh_k,
                q_ac_k,
                u_pump_arr[k],
                t_buf_top_arr[k],
                r_v_arr[k],
            )
            open_window_hold_rem_s = 0.0
            open_window_active = False
            if states_out is not None:
                states_out[k] = (x0, x1, x2)
            if save_mask is not None and saved_k0 is not None and save_mask[k]:
                saved_k0[k] = (x0, x1)
            continue

        step_s = dt_arr[k]
        if step_s < 1.0:
            step_s = 1.0

        if open_window_hold_rem_s > 0.0:
            open_window_hold_rem_s -= step_s
            if open_window_hold_rem_s <= 0.0:
                open_window_hold_rem_s = 0.0
                open_window_active = False
            else:
                open_window_active = True
        else:
            open_window_active = False

        use_fast_vent = x2 < -1.0 and (open_window_active or x2 < -150.0)
        v_exc_k = v_exc_arr[k]

        u0 = t_out_arr[k]
        u1 = t_up_arr[k]
        u2 = t_down_arr[k]
        u3 = i_sol_arr[k]
        u4 = q_fh_k
        u5 = q_int_arr[k]
        u6 = q_ac_k

        qc00 = q_air_base + (q_air_ow if open_window_active else 0.0)
        qc11 = qc11_base_arr[k] + qc11_qfh_factor * q_fh_k * q_fh_k

        if (
            step_s == 300.0
            and not use_fast_vent
            and (c_wind == 0.0 or v_exc_k == 0.0)
        ):
            qc22 = sd_sq2_arr[k] * inv_tau_d
            xm0 = (
                a00 * x0
                + a01 * x1
                + a02 * x2
                + b00 * u0
                + b01 * u1
                + b02 * u2
                + b03 * u3
                + b04 * u4
                + b05 * u5
                + b06 * u6
            )
            xm1 = (
                a10 * x0
                + a11 * x1
                + a12 * x2
                + b10 * u0
                + b11 * u1
                + b12 * u2
                + b13 * u3
                + b14 * u4
                + b15 * u5
                + b16 * u6
            )
            xm2 = a22 * x2

            q00 = qc00 * g0_00 + qc11 * g1_00 + qc22 * g2_00
            q01 = qc00 * g0_01 + qc11 * g1_01 + qc22 * g2_01
            q02 = qc00 * g0_02 + qc11 * g1_02 + qc22 * g2_02
            q11 = qc00 * g0_11 + qc11 * g1_11 + qc22 * g2_11
            q12 = qc00 * g0_12 + qc11 * g1_12 + qc22 * g2_12
            q22 = qc00 * g0_22 + qc11 * g1_22 + qc22 * g2_22

            ca00, ca01, ca02 = a00, a01, a02
            ca10, ca11, ca12 = a10, a11, a12
            ca22 = a22
        else:
            v_eff_k = 0.0 if c_wind == 0.0 else v_exc_k
            if use_fast_vent:
                if vent_params is None:
                    vent_params = dataclasses.replace(
                        params, tau_d_h=zoh_kalman.TAU_D_VENT_S / 3600.0
                    )
                ad_k, bd_k, _, _, _ = zoh_kalman._discretize_ab_cached(
                    vent_params, step_s, v_eff_k
                )
                g0_k, g1_k, g2_k = zoh_kalman._discretize_qd_basis_cached(
                    vent_params, step_s, v_eff_k
                )
                ad_nom, bd_nom, _, _, _ = zoh_kalman._discretize_ab_cached(
                    params, step_s, v_eff_k
                )
                af10, af11, af12 = ad_nom[1][0], ad_nom[1][1], ad_nom[1][2]
                bf1 = bd_nom[1]
                qc22 = sd_sq2_arr[k] * inv_tau_d_vent
            else:
                ad_k, bd_k, _, _, _ = zoh_kalman._discretize_ab_cached(
                    params, step_s, v_eff_k
                )
                g0_k, g1_k, g2_k = zoh_kalman._discretize_qd_basis_cached(
                    params, step_s, v_eff_k
                )
                af10, af11, af12 = ad_k[1][0], ad_k[1][1], ad_k[1][2]
                bf1 = bd_k[1]
                qc22 = sd_sq2_arr[k] * inv_tau_d

            ca00, ca01, ca02 = ad_k[0][0], ad_k[0][1], ad_k[0][2]
            ca10, ca11, ca12 = ad_k[1][0], ad_k[1][1], ad_k[1][2]
            ca22 = ad_k[2][2]
            bk0 = bd_k[0]
            xm0 = (
                ca00 * x0
                + ca01 * x1
                + ca02 * x2
                + bk0[0] * u0
                + bk0[1] * u1
                + bk0[2] * u2
                + bk0[3] * u3
                + bk0[4] * u4
                + bk0[5] * u5
                + bk0[6] * u6
            )
            xm1 = (
                af10 * x0
                + af11 * x1
                + af12 * x2
                + bf1[0] * u0
                + bf1[1] * u1
                + bf1[2] * u2
                + bf1[3] * u3
                + bf1[4] * u4
                + bf1[5] * u5
                + bf1[6] * u6
            )
            xm2 = ca22 * x2

            q00 = qc00 * g0_k[0][0] + qc11 * g1_k[0][0] + qc22 * g2_k[0][0]
            q01 = qc00 * g0_k[0][1] + qc11 * g1_k[0][1] + qc22 * g2_k[0][1]
            q02 = qc00 * g0_k[0][2] + qc11 * g1_k[0][2] + qc22 * g2_k[0][2]
            q11 = qc00 * g0_k[1][1] + qc11 * g1_k[1][1] + qc22 * g2_k[1][1]
            q12 = qc00 * g0_k[1][2] + qc11 * g1_k[1][2] + qc22 * g2_k[1][2]
            q22 = qc00 * g0_k[2][2] + qc11 * g1_k[2][2] + qc22 * g2_k[2][2]

        ap00 = ca00 * p00 + ca01 * p01 + ca02 * p02
        ap01 = ca00 * p01 + ca01 * p11 + ca02 * p12
        ap02 = ca00 * p02 + ca01 * p12 + ca02 * p22
        ap10 = ca10 * p00 + ca11 * p01 + ca12 * p02
        ap11 = ca10 * p01 + ca11 * p11 + ca12 * p12
        ap12 = ca10 * p02 + ca11 * p12 + ca12 * p22
        ap20 = ca22 * p02
        ap21 = ca22 * p12
        ap22 = ca22 * p22

        pm00 = ap00 * ca00 + ap01 * ca01 + ap02 * ca02 + q00
        pm01 = (
            0.5
            * (
                (ap00 * ca10 + ap01 * ca11 + ap02 * ca12)
                + (ap10 * ca00 + ap11 * ca01 + ap12 * ca02)
            )
            + q01
        )
        pm02 = 0.5 * (ap02 * ca22 + (ap20 * ca00 + ap21 * ca01 + ap22 * ca02)) + q02
        pm11 = ap10 * ca10 + ap11 * ca11 + ap12 * ca12 + q11
        pm12 = 0.5 * (ap12 * ca22 + (ap20 * ca10 + ap21 * ca11 + ap22 * ca12)) + q12
        pm22 = ap22 * ca22 + q22

        do_air_update = False
        e_innov = 0.0
        rv_eff = 0.0

        if mm == 1:
            rv_eff = r_v_arr[k]
            e_innov = y_obs_arr[k] - xm0
            if obs_slope_arr[k] < ow_slope_thr:
                s_k = pm00 + rv_eff
                gate_thresh = max(0.20, 2.5 * math.sqrt(max(1e-12, s_k)))
                if e_innov < -gate_thresh:
                    open_window_hold_rem_s = ow_hold_const
                    open_window_active = True
            do_air_update = True
        elif mm == 2:
            tobit = zoh_kalman._tobit_equivalent_innovation(
                y_center=y_obs_arr[k],
                half_width_k=0.10,
                mu_minus=xm0,
                p00_minus=pm00,
                sigma_eps_k=0.03,
            )
            if tobit is not None:
                e_innov, rv_eff = tobit
                do_air_update = True
        elif mm == 3:
            rv_eff = r_v_arr[k]
            e_innov = y_obs_arr[k] - xm0
            do_air_update = True

        if do_air_update:
            s_val = pm00 + rv_eff
            if s_val < 1e-12:
                s_val = 1e-12
            gk0 = pm00 / s_val
            gk1 = 0.0 if open_window_active else (pm01 / s_val)
            gk2 = pm02 / s_val

            if (
                cold_cap_arr[k]
                and abs(e_innov) > 1e-12
                and abs(gk1 * e_innov) > floor_cap_k
            ):
                gk1 = math.copysign(floor_cap_k / abs(e_innov), gk1)

            x0 = xm0 + gk0 * e_innov
            x1 = xm1 + gk1 * e_innov
            x2 = xm2 + gk2 * e_innov

            w00 = 1.0 - gk0
            wp00 = w00 * pm00
            wp01 = w00 * pm01
            wp02 = w00 * pm02
            wp10 = -gk1 * pm00 + pm01
            wp11 = -gk1 * pm01 + pm11
            wp12 = -gk1 * pm02 + pm12
            wp20 = -gk2 * pm00 + pm02
            wp21 = -gk2 * pm01 + pm12
            wp22 = -gk2 * pm02 + pm22

            p00 = wp00 * w00 + rv_eff * gk0 * gk0
            if p00 < 1e-14:
                p00 = 1e-14
            p01 = 0.5 * (-wp00 * gk1 + wp01 + wp10 * w00) + rv_eff * gk0 * gk1
            p02 = 0.5 * (-wp00 * gk2 + wp02 + wp20 * w00) + rv_eff * gk0 * gk2
            p11 = -wp10 * gk1 + wp11 + rv_eff * gk1 * gk1
            if p11 < 1e-14:
                p11 = 1e-14
            p12 = 0.5 * (-wp10 * gk2 + wp12 - wp20 * gk1 + wp21) + rv_eff * gk1 * gk2
            p22 = -wp20 * gk2 + wp22 + rv_eff * gk2 * gk2
            if p22 < 1e-14:
                p22 = 1e-14
        else:
            x0, x1, x2 = xm0, xm1, xm2
            p00, p01, p02, p11, p12, p22 = pm00, pm01, pm02, pm11, pm12, pm22

        if has_surf:
            ys_opt = t_surf_arr[k]
            if ys_opt is not None:
                h_surf = (1.0 - surf_gamma, surf_gamma, 0.0)
                e_surf = ys_opt - (h_surf[0] * x0 + h_surf[1] * x1)
                x_u, p_u = zoh_kalman._joseph_scalar_update(
                    (x0, x1, x2),
                    [[p00, p01, p02], [p01, p11, p12], [p02, p12, p22]],
                    h_row=h_surf,
                    innovation=e_surf,
                    r_v=zoh_kalman.R_MEAS_SURF_K2,
                )
                x0, x1, x2 = x_u
                p00, p01, p02 = p_u[0][0], p_u[0][1], p_u[0][2]
                p11, p12, p22 = p_u[1][1], p_u[1][2], p_u[2][2]

        if states_out is not None:
            states_out[k] = (x0, x1, x2)
        if save_mask is not None and saved_k0 is not None and save_mask[k]:
            saved_k0[k] = (x0, x1)

    return states_out, saved_k0, ad_std, bd_std, khx_ratio, khx_scale_act


def evaluate_windows_on_tape(
    tape: PrecomputedZoneTape,
    params: contracts.ZoneParameters,
    window_specs: Sequence[tuple[int, int, Sequence[int], float]],
) -> list[float]:
    """Evaluates trajectory MAP residuals via unrolled scalar Kalman + ZOH."""
    if not window_specs or tape.n_rows == 0:
        return []

    max_k0 = max(spec[0] for spec in window_specs)
    needed_k0 = {spec[0] for spec in window_specs}
    _, saved_k0, ad_std, bd_std, khx_ratio, khx_scale_act = (
        _run_kalman_on_tape(
            tape,
            params,
            end_k=max_k0,
            save_k0_set=needed_k0,
            record_all=False,
        )
    )
    assert saved_k0 is not None

    is_kt = tape.zone == "kt"
    khx = params.k_hx_w_k
    eta_ac = params.eta_ac
    c_wind = params.c_wind

    a00, a01 = ad_std[0][0], ad_std[0][1]
    a10, a11 = ad_std[1][0], ad_std[1][1]
    b00, b01, b02, b03, b04, b05, b06 = bd_std[0]
    b10, b11, b12, b13, b14, b15, b16 = bd_std[1]

    dt_arr = tape.dt_s
    t_meas_arr = tape.t_meas
    u_prod_arr = tape.u_prod
    t_buf_top_arr = tape.t_buf_top
    q_fh_hyd_arr = tape.q_fh_hyd
    is_harvest_arr = tape.is_harvest
    ac_eff_arr = tape.ac_eff_el
    q_ac_fixed_arr = tape.q_ac_fixed
    t_out_arr = tape.t_out
    v_exc_arr = tape.v_excess
    i_sol_arr = tape.i_sol
    t_down_arr = tape.t_down
    q_int_arr = tape.q_int
    t_up_arr = tape.t_up

    # Open-loop ZOH rollout on all requested windows (d_est = 0 at k0).
    residuals: list[float] = []
    for k0_w, len_w, sample_pts, w_factor in window_specs:
        ox0, ox1 = saved_k0[k0_w]
        ptr = 0
        n_pts = len(sample_pts)
        next_pt = sample_pts[0] if n_pts > 0 else -1
        for s in range(1, len_w + 1):
            idx = k0_w + s
            up_uz = u_prod_arr[idx]
            if up_uz == 0.0:
                qh = q_fh_hyd_arr[idx]
                q_fh_s = (qh * khx_ratio) if qh > 0.0 else 0.0
            elif is_harvest_arr[idx]:
                q_fh_s = up_uz * khx * (t_buf_top_arr[idx] - ox1)
                if q_fh_s < 0.0:
                    q_fh_s = 0.0
            else:
                q_fh_s = q_fh_hyd_arr[idx] * khx_scale_act

            q_ac_s = (
                (ac_eff_arr[idx] * eta_ac + q_ac_fixed_arr[idx])
                if is_kt
                else 0.0
            )

            dt_idx = dt_arr[idx]
            v_exc_idx = v_exc_arr[idx]
            if dt_idx == 300.0 and (c_wind == 0.0 or v_exc_idx == 0.0):
                nx0 = (
                    a00 * ox0
                    + a01 * ox1
                    + b00 * t_out_arr[idx]
                    + b01 * t_up_arr[idx]
                    + b02 * t_down_arr[idx]
                    + b03 * i_sol_arr[idx]
                    + b04 * q_fh_s
                    + b05 * q_int_arr[idx]
                    + b06 * q_ac_s
                )
                nx1 = (
                    a10 * ox0
                    + a11 * ox1
                    + b10 * t_out_arr[idx]
                    + b11 * t_up_arr[idx]
                    + b12 * t_down_arr[idx]
                    + b13 * i_sol_arr[idx]
                    + b14 * q_fh_s
                    + b15 * q_int_arr[idx]
                    + b16 * q_ac_s
                )
            else:
                v_eff_idx = 0.0 if c_wind == 0.0 else v_exc_idx
                ad_o, bd_o, _, _, _ = zoh_kalman._discretize_ab_cached(
                    params, dt_idx, v_eff_idx
                )
                bo0 = bd_o[0]
                bo1 = bd_o[1]
                nx0 = (
                    ad_o[0][0] * ox0
                    + ad_o[0][1] * ox1
                    + bo0[0] * t_out_arr[idx]
                    + bo0[1] * t_up_arr[idx]
                    + bo0[2] * t_down_arr[idx]
                    + bo0[3] * i_sol_arr[idx]
                    + bo0[4] * q_fh_s
                    + bo0[5] * q_int_arr[idx]
                    + bo0[6] * q_ac_s
                )
                nx1 = (
                    ad_o[1][0] * ox0
                    + ad_o[1][1] * ox1
                    + bo1[0] * t_out_arr[idx]
                    + bo1[1] * t_up_arr[idx]
                    + bo1[2] * t_down_arr[idx]
                    + bo1[3] * i_sol_arr[idx]
                    + bo1[4] * q_fh_s
                    + bo1[5] * q_int_arr[idx]
                    + bo1[6] * q_ac_s
                )
            ox0, ox1 = nx0, nx1
            if s == next_pt:
                residuals.append((ox0 - t_meas_arr[idx]) * w_factor)
                ptr += 1
                next_pt = sample_pts[ptr] if ptr < n_pts else -1

    return residuals


def get_zone_tape(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
) -> PrecomputedZoneTape:
    """Returns cached `PrecomputedZoneTape` attached to `hist` for `zone`."""
    if zone == "wk":
        if hist._tape_wk is None:
            hist._tape_wk = build_zone_tape(hist, "wk")
        return hist._tape_wk
    if hist._tape_kt is None:
        hist._tape_kt = build_zone_tape(hist, "kt")
    return hist._tape_kt


def _reconstruct_states_on_tape(
    tape: PrecomputedZoneTape,
    params: contracts.ZoneParameters,
    end_k: int | None = None,
) -> list[tuple[float, float, float]]:
    """Runs unrolled scalar Kalman filter on `tape` up to `end_k` inclusive."""
    n = tape.n_rows
    if n == 0:
        return []
    limit_k = (n - 1) if end_k is None else min(end_k, n - 1)
    states_out, _, _, _, _, _ = _run_kalman_on_tape(
        tape,
        params,
        end_k=limit_k,
        save_k0_set=None,
        record_all=True,
    )
    assert states_out is not None
    return states_out


def reconstruct_states_tuples(
    hist: PreprocessedHistory,
    zone: contracts.ZoneId,
    params: contracts.ZoneParameters,
) -> list[tuple[float, float, float]]:
    """Fast scalar Kalman state reconstruction returning `(t_air, t_floor, d)`."""
    return _reconstruct_states_on_tape(get_zone_tape(hist, zone), params)


def _rollout_open_loop_window_fast(
    tape: PrecomputedZoneTape,
    k0: int,
    length_steps: int,
    state_init: tuple[float, float, float],
    params: contracts.ZoneParameters,
    ad_std: zoh_kalman._Mat3,
    bd_std: zoh_kalman._Mat3x7,
    khx_ratio: float,
    khx_scale_act: float,
) -> tuple[list[float], list[float]]:
    """Fast unrolled open-loop ZOH rollout preserving all 3 states & 7 inputs."""
    ox0, ox1, ox2 = state_init
    is_kt = tape.zone == "kt"
    khx = params.k_hx_w_k
    eta_ac = params.eta_ac
    c_wind = params.c_wind

    a00, a01, a02 = ad_std[0][0], ad_std[0][1], ad_std[0][2]
    a10, a11, a12 = ad_std[1][0], ad_std[1][1], ad_std[1][2]
    a22 = ad_std[2][2]
    b00, b01, b02, b03, b04, b05, b06 = bd_std[0]
    b10, b11, b12, b13, b14, b15, b16 = bd_std[1]

    dt_arr = tape.dt_s
    t_meas_arr = tape.t_meas
    u_prod_arr = tape.u_prod
    t_buf_top_arr = tape.t_buf_top
    q_fh_hyd_arr = tape.q_fh_hyd
    is_harvest_arr = tape.is_harvest
    ac_eff_arr = tape.ac_eff_el
    q_ac_fixed_arr = tape.q_ac_fixed
    t_out_arr = tape.t_out
    v_exc_arr = tape.v_excess
    i_sol_arr = tape.i_sol
    t_down_arr = tape.t_down
    q_int_arr = tape.q_int
    t_up_arr = tape.t_up

    errs: list[float] = [0.0] * length_steps
    q_fhs: list[float] = [0.0] * length_steps

    for s in range(1, length_steps + 1):
        idx = k0 + s
        up_uz = u_prod_arr[idx]
        if up_uz == 0.0:
            qh = q_fh_hyd_arr[idx]
            q_fh_s = (qh * khx_ratio) if qh > 0.0 else 0.0
        elif is_harvest_arr[idx]:
            q_fh_s = up_uz * khx * (t_buf_top_arr[idx] - ox1)
            if q_fh_s < 0.0:
                q_fh_s = 0.0
        else:
            q_fh_s = q_fh_hyd_arr[idx] * khx_scale_act

        q_ac_s = (
            (ac_eff_arr[idx] * eta_ac + q_ac_fixed_arr[idx]) if is_kt else 0.0
        )

        dt_idx = dt_arr[idx]
        v_exc_idx = v_exc_arr[idx]
        if dt_idx == 300.0 and (c_wind == 0.0 or v_exc_idx == 0.0):
            nx0 = (
                a00 * ox0
                + a01 * ox1
                + a02 * ox2
                + b00 * t_out_arr[idx]
                + b01 * t_up_arr[idx]
                + b02 * t_down_arr[idx]
                + b03 * i_sol_arr[idx]
                + b04 * q_fh_s
                + b05 * q_int_arr[idx]
                + b06 * q_ac_s
            )
            nx1 = (
                a10 * ox0
                + a11 * ox1
                + a12 * ox2
                + b10 * t_out_arr[idx]
                + b11 * t_up_arr[idx]
                + b12 * t_down_arr[idx]
                + b13 * i_sol_arr[idx]
                + b14 * q_fh_s
                + b15 * q_int_arr[idx]
                + b16 * q_ac_s
            )
            ox2 = a22 * ox2
        else:
            v_eff_idx = 0.0 if c_wind == 0.0 else v_exc_idx
            ad_o, bd_o, _, _, _ = zoh_kalman._discretize_ab_cached(
                params, dt_idx, v_eff_idx
            )
            bo0 = bd_o[0]
            bo1 = bd_o[1]
            nx0 = (
                ad_o[0][0] * ox0
                + ad_o[0][1] * ox1
                + ad_o[0][2] * ox2
                + bo0[0] * t_out_arr[idx]
                + bo0[1] * t_up_arr[idx]
                + bo0[2] * t_down_arr[idx]
                + bo0[3] * i_sol_arr[idx]
                + bo0[4] * q_fh_s
                + bo0[5] * q_int_arr[idx]
                + bo0[6] * q_ac_s
            )
            nx1 = (
                ad_o[1][0] * ox0
                + ad_o[1][1] * ox1
                + ad_o[1][2] * ox2
                + bo1[0] * t_out_arr[idx]
                + bo1[1] * t_up_arr[idx]
                + bo1[2] * t_down_arr[idx]
                + bo1[3] * i_sol_arr[idx]
                + bo1[4] * q_fh_s
                + bo1[5] * q_int_arr[idx]
                + bo1[6] * q_ac_s
            )
            ox2 = ad_o[2][2] * ox2

        ox0, ox1 = nx0, nx1
        errs[s - 1] = ox0 - t_meas_arr[idx]
        q_fhs[s - 1] = q_fh_s

    return errs, q_fhs


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
    precomputed_hist_eval: PreprocessedHistory | None = None,
    precomputed_hist_prior: PreprocessedHistory | None = None,
    eval_steps: int = EVAL_14D_STEPS,
) -> contracts.ZoneEvaluationReport:
    """Evaluates 14-day (or custom tail) 6h & `kt` 1h AC open-loop hindcast.

    Runs dual parameter-consistent forward Kalman reconstructions under
    `params_eval` and `params_prior` over up to `48h` spin-up + `eval_steps`
    evaluation so comparisons against Day-1 priors suffer zero initial-state
    enthalpy jump.

    Args:
        columns: RingBuffer column sequences.
        zone: Target zone (`"wk"` or `"kt"`).
        params_eval: Active or candidate `ZoneParameters` to score.
        params_prior: Reference Day-1 prior `ZoneParameters` for Skill Score.
        flow_rates_l_min: Optional manifold flow rates [L/min].
        k_hx_by_group: Optional `K_hx` [W/K] mapping per group.
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.
        _ref_d_act: Optional shared disturbance series used by the holdout gate
            to guarantee identical window selection between two parameter sets.
        precomputed_hist_eval: Optional precomputed history for `params_eval`.
        precomputed_hist_prior: Optional precomputed history for `params_prior`.
        eval_steps: Number of tail evaluation steps after 48h spin-up
            (default `EVAL_14D_STEPS = 4032`).

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

    eff_eval_steps = max(HORIZON_6H_STEPS + 1, int(eval_steps))
    max_slice_steps = SPINUP_48H_STEPS + eff_eval_steps
    start_slice = max(0, n_total - max_slice_steps)
    n_slice = n_total - start_slice
    sub_cols_cache: Mapping[str, Sequence[float | int]] | None = None

    def _get_sub_cols() -> Mapping[str, Sequence[float | int]]:
        nonlocal sub_cols_cache
        if sub_cols_cache is None:
            sub_cols_cache = (
                columns
                if start_slice == 0
                else slice_columns(columns, start_slice, n_total)
            )
        return sub_cols_cache

    if n_slice >= SPINUP_48H_STEPS + HORIZON_6H_STEPS:
        spinup_steps = SPINUP_48H_STEPS
    else:
        spinup_steps = max(288, n_slice - 2 * HORIZON_6H_STEPS)

    k_hx_eval_map = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
    if k_hx_by_group is not None:
        for group_name, val in k_hx_by_group.items():
            target_key = "kantoor" if group_name == "kt" else group_name
            k_hx_eval_map[target_key] = float(val)
    zone_khx_key = "wk" if zone == "wk" else "kantoor"
    k_hx_eval_map[zone_khx_key] = params_eval.k_hx_w_k

    k_hx_prior_map = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
    k_hx_prior_map[zone_khx_key] = params_prior.k_hx_w_k

    def _hist_matches_khx(
        cand_h: PreprocessedHistory | None,
        target_khx: float,
    ) -> bool:
        if cand_h is None or cand_h.n_rows != n_slice:
            return False
        h_khx = cand_h.hyd_k_hx_wk if zone == "wk" else cand_h.hyd_k_hx_kt
        return abs(h_khx - target_khx) <= 1e-9

    if _hist_matches_khx(precomputed_hist_eval, k_hx_eval_map[zone_khx_key]):
        assert precomputed_hist_eval is not None
        hist_eval = precomputed_hist_eval
    elif _hist_matches_khx(precomputed_hist_prior, k_hx_eval_map[zone_khx_key]):
        assert precomputed_hist_prior is not None
        hist_eval = precomputed_hist_prior
    elif (
        precomputed_hist_eval is not None
        and precomputed_hist_eval.n_rows == n_slice
    ):
        hist_eval = recompute_history_for_khx(
            precomputed_hist_eval,
            _get_sub_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_eval_map,
        )
    else:
        hist_eval = preprocess_history_columns(
            _get_sub_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_eval_map,
            eta_ac_ref=params_eval.eta_ac,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )

    if _hist_matches_khx(precomputed_hist_prior, k_hx_prior_map[zone_khx_key]):
        assert precomputed_hist_prior is not None
        hist_prior = precomputed_hist_prior
    elif _hist_matches_khx(hist_eval, k_hx_prior_map[zone_khx_key]):
        hist_prior = hist_eval
    else:
        hist_prior = recompute_history_for_khx(
            hist_eval,
            _get_sub_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_prior_map,
        )

    tape_eval = get_zone_tape(hist_eval, zone)
    tape_prior = (
        tape_eval
        if hist_prior is hist_eval
        else get_zone_tape(hist_prior, zone)
    )

    states_eval = _reconstruct_states_on_tape(tape_eval, params_eval)
    states_prior = _reconstruct_states_on_tape(tape_prior, params_prior)

    ad_eval, bd_eval, _, _, _ = zoh_kalman._discretize_ab_cached(
        params_eval, 300.0, 0.0
    )
    ad_prior, bd_prior, _, _, _ = zoh_kalman._discretize_ab_cached(
        params_prior, 300.0, 0.0
    )
    eval_khx_ratio = (
        (params_eval.k_hx_w_k / tape_eval.hyd_k_hx)
        if tape_eval.hyd_k_hx > 1e-6
        else 1.0
    )
    eval_khx_act = (
        eval_khx_ratio
        if (
            tape_eval.hyd_k_hx > 1e-6
            and abs(params_eval.k_hx_w_k - tape_eval.hyd_k_hx) > 1e-6
        )
        else 1.0
    )
    prior_khx_ratio = (
        (params_prior.k_hx_w_k / tape_prior.hyd_k_hx)
        if tape_prior.hyd_k_hx > 1e-6
        else 1.0
    )
    prior_khx_act = (
        prior_khx_ratio
        if (
            tape_prior.hyd_k_hx > 1e-6
            and abs(params_prior.k_hx_w_k - tape_prior.hyd_k_hx) > 1e-6
        )
        else 1.0
    )

    if _ref_d_act is not None and len(_ref_d_act) == n_slice:
        d_act = list(_ref_d_act)
    else:
        d_act = [st[2] for st in states_eval]

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

        e_eval, q_fh_ol = _rollout_open_loop_window_fast(
            tape_eval,
            k0,
            lw,
            states_eval[k0],
            params_eval,
            ad_eval,
            bd_eval,
            eval_khx_ratio,
            eval_khx_act,
        )
        e_prior, _ = _rollout_open_loop_window_fast(
            tape_prior,
            k0,
            lw,
            states_prior[k0],
            params_prior,
            ad_prior,
            bd_prior,
            prior_khx_ratio,
            prior_khx_act,
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

            e_eval_ac, q_fh_ac = _rollout_open_loop_window_fast(
                tape_eval,
                k0,
                HORIZON_AC_1H_STEPS,
                states_eval[k0],
                params_eval,
                ad_eval,
                bd_eval,
                eval_khx_ratio,
                eval_khx_act,
            )
            if max(q_fh_ac) >= 300.0:
                continue

            e_prior_ac, _ = _rollout_open_loop_window_fast(
                tape_prior,
                k0,
                HORIZON_AC_1H_STEPS,
                states_prior[k0],
                params_prior,
                ad_prior,
                bd_prior,
                prior_khx_ratio,
                prior_khx_act,
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
