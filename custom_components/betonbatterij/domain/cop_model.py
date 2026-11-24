"""Heat-pump COP curve model, P1 electrical bias calibration, and MAP fitter.

Owns the Panasonic 9 kW T-CAP Carnot COP model for space heating
(`predict_cop_cv`), two-stage domestic hot water (`predict_cop_dhw`), smooth
evaporator frost penalty (`frost_penalty_factor`), authoritative live expected
CV-COP evaluation from raw telemetry (`predict_live_expected_cop_from_row`),
nocturnal P1 electrical power bias identification (`calibrate_p1_power_bias`),
and Huber IRLS/LM parameter fitting (`fit_cop_parameters`).

Does not own zone thermal 2R2C dynamics (`zoh_kalman.py`), waterside manifold
hydraulics (`hydraulics.py`), or Home Assistant sensor publishing.

Units: temperatures in °C, relative humidity in %, electrical/thermal powers in
W, compressor frequency in Hz, COP dimensionless (`>= 1.0`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any, TypeAlias, TypeGuard

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import solar_weather

MIN_CV_RUN_STEPS: int = 12  # >= 60 min steady CV run (incl. 15 min startup).
DHW_COMP_MAX_TEMP_C: float = 51.0
DHW_FIT_MAX_TEMP_C: float = 50.5
BACKUP_HEATER_POWER_W: float = 3000.0
MEMORY_DECAY_GAMMA: float = preprocessing.MEMORY_WEEKLY_GAMMA

_CvRunSample: TypeAlias = tuple[int, float, float, float, float]
_DhwRunSample: TypeAlias = tuple[int, float, float, float, float, float]


def _is_finite(val: float | int | None) -> TypeGuard[float | int]:
    """Checks whether `val` is a finite non-boolean `float` or `int`."""
    return preprocessing.is_finite_number(val)


def frost_penalty_factor(t_out_c: float, rh_out_pct: float = 80.0) -> float:
    """Computes the smooth evaporator frost penalty factor `phi_frost`.

    Peaks around `T_out = +1.0 °C` in humid outdoor air (`RH > 78%`) and is
    strictly zero outside `[-5.0, +5.5] °C`.

    Args:
        t_out_c: Outdoor air temperature [°C].
        rh_out_pct: Outdoor relative humidity [%].

    Returns:
        Dimensionless frost penalty intensity in `[0.0, 1.0]`.
    """
    t_out = float(t_out_c)
    if t_out < -5.0 or t_out > 5.5:
        return 0.0

    rh = 80.0 if not _is_finite(rh_out_pct) else float(rh_out_pct)
    rh_factor = max(0.0, min(1.0, (rh - 78.0) / 17.0))
    if rh_factor <= 0.0:
        return 0.0

    temp_bell = math.exp(-((t_out - 1.0) ** 2) / (2.0 * (2.2**2)))
    return rh_factor * temp_bell


def predict_cop_cv(
    params: contracts.CopParameters,
    t_out_c: float,
    t_sup_c: float,
    rh_out_pct: float = 80.0,
) -> float:
    """Predicts net space-heating COP including primary pump and defrost loss.

    Args:
        params: Calibrated Carnot and defrost `CopParameters`.
        t_out_c: Outdoor air temperature [°C].
        t_sup_c: Heat-pump outlet / supply water temperature [°C].
        rh_out_pct: Outdoor relative humidity [%].

    Returns:
        Expected net space-heating COP (`>= 1.0`).
    """
    t_out = float(t_out_c)
    t_sup = float(t_sup_c)
    t_evap_c = t_out - 4.0
    lift_k = max(14.0, (t_sup + 3.0) - t_evap_c)
    # Algebraically identical to `t_sup + 276.15` for all lift_k >= 14.0 while
    # preserving thermodynamic monotonicity at small lifts (< 14 K).
    t_cond_k = (t_evap_c + 273.15) + lift_k
    frost = frost_penalty_factor(t_out, rh_out_pct)
    cop = (
        params.eta_carnot_cv
        * (t_cond_k / lift_k)
        * (1.0 - params.beta_defrost * frost)
    )
    return max(1.0, cop)


def predict_live_expected_cop_from_row(
    cop_params: contracts.CopParameters,
    raw_row: Mapping[str, Any],
    cop_diagnostics: Mapping[str, Any] | None = None,
    buffer_offsets: contracts.BufferSensorOffsets = priors.DAY1_BUFFER_OFFSETS,
) -> float:
    """Computes authoritative live expected CV-COP from a raw telemetry row.

    Applies the unified DHW, residual boiler-water, and idle `0 Hz` guard:
    whenever `FLAG_DHW_TANK` is active, `t_wp_out_c > 40.0 °C`, or the
    compressor is idle (`f_comp_hz <= 0.5 Hz`), the supply temperature is read
    first from offset-calibrated `t_buf_top_end_c` / `t_buf_top_start_c` in
    `(15.0, 40.0] °C`, falling back to `t_wp_out_c` in `(15.0, 40.0] °C` if
    buffer columns are missing/invalid, else `30.0 °C`. During active CV
    compressor runs, `t_wp_out_c` in `(15.0, 40.0] °C` is preferred over
    offset-calibrated buffer columns.

    Args:
        cop_params: Active calibrated `CopParameters`.
        raw_row: Latest raw 5-minute telemetry row mapping.
        cop_diagnostics: Optional diagnostic mapping (retained for signature
            uniformity; `predict_cop_cv` is always authoritative).
        buffer_offsets: Active 1-Wire buffer tank sensor offsets.

    Returns:
        Expected live space-heating COP (`>= 1.0`).
    """
    del cop_diagnostics  # `predict_cop_cv` is the single authoritative formula.

    t_out = 7.0
    for col in ("t_out_dak_c", "t_out_weather_c", "t_out_hm_c"):
        cand = raw_row.get(col)
        if _is_finite(cand):
            t_out = float(cand)
            break

    raw_flags = raw_row.get("flags", 0)
    flags_val = int(raw_flags) if _is_finite(raw_flags) else 0
    is_dhw_tank = bool(flags_val & const.FLAG_DHW_TANK)

    wp_out_raw = raw_row.get("t_wp_out_c")
    wp_out_hot_dhw = _is_finite(wp_out_raw) and float(wp_out_raw) > 40.0

    f_comp_raw = raw_row.get("f_comp_hz")
    comp_idle = _is_finite(f_comp_raw) and float(f_comp_raw) <= 0.5

    top_offset = float(buffer_offsets.buf_top_offset_c)
    t_sup = 30.0
    if is_dhw_tank or wp_out_hot_dhw or comp_idle:
        found_buf = False
        for col in ("t_buf_top_end_c", "t_buf_top_start_c"):
            cand = raw_row.get(col)
            if _is_finite(cand):
                cand_cal = float(cand) + top_offset
                if 15.0 < cand_cal <= 40.0:
                    t_sup = cand_cal
                    found_buf = True
                    break
        if (
            not found_buf
            and _is_finite(wp_out_raw)
            and 15.0 < float(wp_out_raw) <= 40.0
        ):
            t_sup = float(wp_out_raw)
    else:
        if _is_finite(wp_out_raw) and 15.0 < float(wp_out_raw) <= 40.0:
            t_sup = float(wp_out_raw)
        else:
            for col in ("t_buf_top_end_c", "t_buf_top_start_c"):
                cand = raw_row.get(col)
                if _is_finite(cand):
                    cand_cal = float(cand) + top_offset
                    if 15.0 < cand_cal <= 40.0:
                        t_sup = cand_cal
                        break

    rh_out = 80.0
    cand_rh = raw_row.get("rh_out_pct")
    if _is_finite(cand_rh) and 0.0 <= float(cand_rh) <= 100.0:
        rh_out = float(cand_rh)

    return predict_cop_cv(cop_params, t_out, t_sup, rh_out)


def predict_cop_dhw(
    params: contracts.CopParameters,
    t_out_c: float,
    t_dhw_start_c: float,
    t_dhw_target_c: float,
    rh_out_pct: float = 80.0,
) -> float:
    """Predicts two-stage DHW run COP across the `51.0 °C` compressor ceiling.

    Below `51.0 °C`, the R410A compressor heats the tank (`+8.5 K` coil pinch);
    above `51.0 °C`, the 3 kW internal electric backup element completes the
    heating cycle at `COP = 1.0`.

    Args:
        params: Calibrated Carnot and defrost `CopParameters`.
        t_out_c: Outdoor air temperature [°C].
        t_dhw_start_c: Initial DHW tank temperature [°C].
        t_dhw_target_c: Target DHW tank setpoint [°C].
        rh_out_pct: Outdoor relative humidity [%].

    Returns:
        Harmonic energy-weighted DHW cycle COP (`>= 1.0`).
    """
    t_out = float(t_out_c)
    t_start = float(t_dhw_start_c)
    t_target = float(t_dhw_target_c)
    t_evap_c = t_out - 4.0
    frost_mult = 1.0 - params.beta_defrost * frost_penalty_factor(
        t_out, rh_out_pct
    )

    if t_target <= t_start:
        if t_target >= DHW_COMP_MAX_TEMP_C:
            return 1.0
        t_cond_c = t_target + 8.5
        lift_k = max(14.0, t_cond_c - t_evap_c)
        t_cond_k = (t_evap_c + 273.15) + lift_k
        return max(
            1.0, params.eta_carnot_dhw * (t_cond_k / lift_k) * frost_mult
        )

    dt_total = t_target - t_start
    t_comp_end = min(t_target, DHW_COMP_MAX_TEMP_C)
    dt_comp = max(0.0, t_comp_end - t_start)
    dt_heater = max(0.0, t_target - max(t_start, DHW_COMP_MAX_TEMP_C))

    if dt_comp > 0.0:
        t_cond_mean_c = 0.5 * (t_start + t_comp_end) + 8.5
        lift_k = max(14.0, t_cond_mean_c - t_evap_c)
        t_cond_k = (t_evap_c + 273.15) + lift_k
        cop_comp = max(
            1.0, params.eta_carnot_dhw * (t_cond_k / lift_k) * frost_mult
        )
    else:
        cop_comp = 1.0

    denom = (dt_comp / cop_comp) + (dt_heater / 1.0)
    if denom <= 1e-12:
        return 1.0
    return max(1.0, dt_total / denom)


def corrected_wp_electrical_power_w(
    params: contracts.CopParameters,
    p_wp_raw_el_w: float,
    f_comp_hz: float,
    internal_heater_on: bool = False,
) -> float:
    """Corrects 200 W floor-quantized Panasonic electrical power during runs.

    Args:
        params: Calibrated `CopParameters` containing `delta_p_wp_bias_w`.
        p_wp_raw_el_w: Raw Panasonic reported electrical power [W].
        f_comp_hz: Compressor operating frequency [Hz].
        internal_heater_on: True when the 3 kW internal backup heater is active.

    Returns:
        Bias- and heater-corrected heat-pump electrical power [W].
    """
    p_raw = (
        0.0 if not _is_finite(p_wp_raw_el_w) else max(0.0, float(p_wp_raw_el_w))
    )
    f_hz = 0.0 if not _is_finite(f_comp_hz) else float(f_comp_hz)
    p_heater = BACKUP_HEATER_POWER_W if internal_heater_on else 0.0

    if f_hz > 0.0:
        return p_raw + params.delta_p_wp_bias_w + p_heater
    if internal_heater_on:
        return p_raw + p_heater
    return p_raw


def calibrate_p1_power_bias(
    columns: Mapping[str, Sequence[float | int]],
    tz_name: str = priors.TIMEZONE_NAME,
) -> float | None:
    """Estimates `delta_p_wp_bias_w` from clean nocturnal (`00:30-06:00`) flanks.

    Selects compressor start/stop edges during deep night where PV output is
    zero and non-WP background house load is flat (`< 30 W` drift).

    Args:
        columns: Columnar history mapping containing `f_comp_hz`,
            `p_wp_raw_el_w`, `p_p1_net_w`, and optional `ts_utc`, `flags`,
            and `p_el_ac_kt_w`.
        tz_name: Local timezone name used to identify `00:30-06:00`.

    Returns:
        Robust median bias [W] clamped to `COP_PARAM_BOUNDS["delta_p_wp_bias_w"]`,
        or `None` when no clean nocturnal flanks are found.
    """
    f_comp = columns.get("f_comp_hz")
    p_raw = columns.get("p_wp_raw_el_w")
    p_p1 = columns.get("p_p1_net_w")
    ts_col = columns.get("ts_utc")
    if f_comp is None or p_raw is None or p_p1 is None:
        return None

    n = min(len(f_comp), len(p_raw), len(p_p1))
    if ts_col is not None:
        n = min(n, len(ts_col))
    if n < 4:
        return None

    flags_col = columns.get("flags")
    p_ac_col = columns.get("p_el_ac_kt_w")
    is_lts_row, _ = preprocessing.detect_lts_and_missing_prim_rows(columns)

    is_off: list[bool] = [False] * n
    is_on: list[bool] = [False] * n
    p_house_minus_raw: list[float] = [math.nan] * n

    for i in range(n):
        if i < len(is_lts_row) and is_lts_row[i]:
            continue
        if ts_col is not None:
            hr, mn = preprocessing.local_hour_minute(int(ts_col[i]), tz_name)
            mins = hr * 60 + mn
            night_ok = 30 <= mins < 360  # 00:30..06:00 local time.
        else:
            night_ok = True

        fl = (
            int(flags_col[i])
            if flags_col is not None and i < len(flags_col)
            else 0
        )
        if (fl & const.FLAG_GAP_BRIDGED) != 0:
            continue
        heater_on = (fl & const.FLAG_INTERNAL_HEATER) != 0
        defrost_on = (fl & const.FLAG_DEFROST_WP) != 0

        f_i = float(f_comp[i])
        pr_i = float(p_raw[i])
        p1_i = float(p_p1[i])
        ac_i = (
            float(p_ac_col[i])
            if p_ac_col is not None
            and i < len(p_ac_col)
            and _is_finite(p_ac_col[i])
            else 0.0
        )

        if not (
            night_ok
            and _is_finite(f_i)
            and _is_finite(pr_i)
            and _is_finite(p1_i)
        ):
            continue
        if heater_on or defrost_on:
            continue

        p_house_minus_raw[i] = (p1_i - ac_i) - pr_i
        if f_i <= 0.5 and pr_i <= 20.0:
            is_off[i] = True
        elif f_i >= 12.0:
            is_on[i] = True

    flank_estimates: list[float] = []

    for i in range(2, n - 1):
        # 1. Start flank: clean OFF window -> steady ON window.
        if (
            is_off[i - 1]
            and is_off[i - 2]
            and (is_on[i] or (i + 1 < n and is_on[i + 1]))
        ):
            off_idxs = [j for j in range(max(0, i - 4), i) if is_off[j]]
            on_start = i if is_on[i] else i + 1
            if (
                on_start + 2 < n
                and is_on[on_start]
                and is_on[on_start + 1]
                and is_on[on_start + 2]
            ):
                steady_start = on_start + 1
            else:
                steady_start = on_start
            on_idxs: list[int] = []
            for j in range(steady_start, min(n, steady_start + 4)):
                if is_on[j]:
                    on_idxs.append(j)
                else:
                    break

            if len(off_idxs) >= 2 and len(on_idxs) >= 2:
                off_vals = [p_house_minus_raw[j] for j in off_idxs]
                on_vals = [p_house_minus_raw[j] for j in on_idxs]
                off_flat = (max(off_vals) - min(off_vals)) < 30.0
                on_flat = (max(on_vals) - min(on_vals)) < 120.0

                post_check_ok = True
                run_end = steady_start
                while (
                    run_end < min(n, steady_start + 36) and not is_off[run_end]
                ):
                    run_end += 1
                if (
                    run_end + 1 < n
                    and is_off[run_end]
                    and is_off[run_end + 1]
                ):
                    post_vals = [
                        p_house_minus_raw[run_end],
                        p_house_minus_raw[run_end + 1],
                    ]
                    if (max(post_vals) - min(post_vals)) < 30.0:
                        drift = abs(
                            statistics.median(post_vals)
                            - statistics.median(off_vals)
                        )
                        if drift >= 30.0:
                            post_check_ok = False

                if off_flat and on_flat and post_check_ok:
                    est = statistics.median(on_vals) - statistics.median(
                        off_vals
                    )
                    if 20.0 <= est <= 350.0:
                        flank_estimates.append(est)

        # 2. Stop flank: steady ON window -> clean OFF window.
        if (
            is_on[i - 1]
            and is_on[i - 2]
            and (is_off[i] or (i + 1 < n and is_off[i + 1]))
        ):
            off_start = i if is_off[i] else i + 1
            if (
                off_start + 2 < n
                and is_off[off_start]
                and is_off[off_start + 1]
                and is_off[off_start + 2]
            ):
                clean_off_start = off_start + 1
            else:
                clean_off_start = off_start
            off_idxs = []
            for j in range(clean_off_start, min(n, clean_off_start + 4)):
                if is_off[j]:
                    off_idxs.append(j)
                else:
                    break

            on_end = i - 1 if i >= 3 and is_on[i - 3] else i
            on_idxs = [j for j in range(max(0, on_end - 4), on_end) if is_on[j]]

            if len(off_idxs) >= 2 and len(on_idxs) >= 2:
                off_vals = [p_house_minus_raw[j] for j in off_idxs]
                on_vals = [p_house_minus_raw[j] for j in on_idxs]
                off_ok = (max(off_vals) - min(off_vals)) < 30.0
                on_ok = (max(on_vals) - min(on_vals)) < 120.0
                if off_ok and on_ok:
                    est = statistics.median(on_vals) - statistics.median(
                        off_vals
                    )
                    if 20.0 <= est <= 350.0:
                        flank_estimates.append(est)

    if not flank_estimates:
        return None

    bnd = priors.COP_PARAM_BOUNDS["delta_p_wp_bias_w"]
    return max(
        bnd.lower, min(bnd.upper, float(statistics.median(flank_estimates)))
    )


def _optional_float_at(
    columns: Mapping[str, Sequence[float | int]],
    name: str,
    idx: int,
) -> float | None:
    """Reads finite float at `columns[name][idx]` or returns `None`."""
    seq = columns.get(name)
    if seq is not None and idx < len(seq) and _is_finite(seq[idx]):
        return float(seq[idx])
    return None


def _step_outdoor_temp(
    columns: Mapping[str, Sequence[float | int]],
    idx: int,
    ts_i: int,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
) -> float:
    """Resolves outdoor temperature at row `idx`."""
    direct = _optional_float_at(columns, "t_out_c", idx)
    if direct is not None:
        return direct

    return solar_weather.select_outdoor_temp(
        ts_utc=ts_i,
        t_dak_c=_optional_float_at(columns, "t_out_dak_c", idx),
        t_weather_c=_optional_float_at(columns, "t_out_weather_c", idx),
        t_hm_ema_c=_optional_float_at(columns, "t_out_hm_c", idx),
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )


def _step_primary_th_w(
    columns: Mapping[str, Sequence[float | int]],
    idx: int,
) -> float:
    """Resolves primary waterside thermal power [W] at row `idx`."""
    p_prim = _optional_float_at(columns, "p_prim_th_w", idx)
    if p_prim is not None:
        return p_prim
    v_val = _optional_float_at(columns, "v_prim_l_min", idx)
    t_out_val = _optional_float_at(columns, "t_wp_out_c", idx)
    t_in_val = _optional_float_at(columns, "t_wp_in_c", idx)
    if v_val is not None and t_out_val is not None and t_in_val is not None:
        return (
            priors.WATER_POWER_FACTOR_W_PER_LMIN_K
            * v_val
            * (t_out_val - t_in_val)
        )
    return 0.0


def _extract_cop_runs(
    columns: Mapping[str, Sequence[float | int]],
    bias_ref_params: contracts.CopParameters,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
) -> tuple[list[_CvRunSample], list[_DhwRunSample]]:
    """Extracts qualifying CV (`>= 60 min`) and DHW (`<= 50.5 °C`) runs."""
    f_comp = columns.get("f_comp_hz", ())
    p_raw = columns.get("p_wp_raw_el_w", ())
    n = min(len(f_comp), len(p_raw))
    ts_col = columns.get("ts_utc")
    if ts_col is not None:
        n = min(n, len(ts_col))
    flags_col = columns.get("flags")
    t_sup_key = (
        "t_wp_out_c"
        if columns.get("t_wp_out_c") is not None
        else "t_buf_top_end_c"
    )
    is_lts_row, missing_prim_th = preprocessing.detect_lts_and_missing_prim_rows(
        columns
    )

    cv_runs: list[_CvRunSample] = []
    dhw_runs: list[_DhwRunSample] = []

    dt_h = const.DT_STEP_S / 3600.0
    idx = 0
    while idx < n:
        fl = (
            int(flags_col[idx])
            if flags_col is not None and idx < len(flags_col)
            else 0
        )
        if (
            (idx < len(is_lts_row) and is_lts_row[idx])
            or (idx < len(missing_prim_th) and missing_prim_th[idx])
            or (fl & const.FLAG_GAP_BRIDGED) != 0
        ):
            idx += 1
            continue
        is_dhw = (fl & const.FLAG_DHW_TANK) != 0
        is_cool = (fl & const.FLAG_FLOOR_COOL) != 0
        is_defrost = (fl & const.FLAG_DEFROST_WP) != 0
        f_val = float(f_comp[idx]) if _is_finite(f_comp[idx]) else 0.0

        if not is_dhw and not is_cool and f_val > 5.0:
            start_i = idx
            end_i = idx + 1
            while end_i < n:
                fl_e = (
                    int(flags_col[end_i])
                    if flags_col is not None and end_i < len(flags_col)
                    else 0
                )
                if (
                    (end_i < len(is_lts_row) and is_lts_row[end_i])
                    or (end_i < len(missing_prim_th) and missing_prim_th[end_i])
                    or (fl_e & const.FLAG_GAP_BRIDGED) != 0
                ):
                    break
                if (fl_e & const.FLAG_DHW_TANK) != 0 or (
                    fl_e & const.FLAG_FLOOR_COOL
                ) != 0:
                    break
                f_e = float(f_comp[end_i]) if _is_finite(f_comp[end_i]) else 0.0
                if f_e > 5.0 or (fl_e & const.FLAG_DEFROST_WP) != 0:
                    end_i += 1
                else:
                    break

            if end_i - start_i >= MIN_CV_RUN_STEPS:
                e_th_wh = 0.0
                e_el_wh = 0.0
                t_outs: list[float] = []
                t_sups: list[float] = []
                rhs: list[float] = []
                for k in range(start_i + 3, end_i):
                    ts_k = (
                        int(ts_col[k])
                        if ts_col is not None
                        else k * const.DT_STEP_S
                    )
                    fl_k = (
                        int(flags_col[k])
                        if flags_col is not None and k < len(flags_col)
                        else 0
                    )
                    p_th = _step_primary_th_w(columns, k)
                    p_el = corrected_wp_electrical_power_w(
                        bias_ref_params,
                        float(p_raw[k]),
                        float(f_comp[k]),
                        internal_heater_on=(fl_k & const.FLAG_INTERNAL_HEATER)
                        != 0,
                    )
                    e_th_wh += p_th * dt_h
                    e_el_wh += p_el * dt_h
                    t_outs.append(
                        _step_outdoor_temp(
                            columns, k, ts_k, lat_deg=lat_deg, lon_deg=lon_deg
                        )
                    )
                    if (fl_k & const.FLAG_DEFROST_WP) == 0:
                        sup_k = _optional_float_at(columns, t_sup_key, k)
                        if sup_k is not None:
                            t_sups.append(sup_k)
                    rh_k = _optional_float_at(columns, "rh_out_pct", k)
                    rhs.append(rh_k if rh_k is not None else 80.0)

                if e_el_wh > 80.0 and e_th_wh > 80.0 and t_outs:
                    cop_obs = e_th_wh / e_el_wh
                    if 1.0 <= cop_obs <= 9.0:
                        ts_end = (
                            int(ts_col[end_i - 1])
                            if ts_col is not None
                            else (end_i - 1) * const.DT_STEP_S
                        )
                        t_sup_mean = (
                            (sum(t_sups) / len(t_sups)) if t_sups else 28.0
                        )
                        cv_runs.append(
                            (
                                ts_end,
                                sum(t_outs) / len(t_outs),
                                t_sup_mean,
                                sum(rhs) / len(rhs),
                                cop_obs,
                            )
                        )
            idx = end_i
            continue

        if is_dhw and (f_val > 5.0 or is_defrost):
            start_i = idx
            end_i = idx + 1
            while end_i < n:
                fl_e = (
                    int(flags_col[end_i])
                    if flags_col is not None and end_i < len(flags_col)
                    else 0
                )
                if (
                    (end_i < len(is_lts_row) and is_lts_row[end_i])
                    or (end_i < len(missing_prim_th) and missing_prim_th[end_i])
                    or (fl_e & const.FLAG_GAP_BRIDGED) != 0
                ):
                    break
                if (fl_e & const.FLAG_DHW_TANK) == 0:
                    break
                f_e = float(f_comp[end_i]) if _is_finite(f_comp[end_i]) else 0.0
                if (
                    f_e > 5.0
                    or (fl_e & const.FLAG_DEFROST_WP) != 0
                    or (fl_e & const.FLAG_INTERNAL_HEATER) != 0
                ):
                    end_i += 1
                else:
                    break

            comp_steps: list[int] = []
            dhw_temps: list[float] = []
            for k in range(start_i, end_i):
                fl_k = (
                    int(flags_col[k])
                    if flags_col is not None and k < len(flags_col)
                    else 0
                )
                if (fl_k & const.FLAG_INTERNAL_HEATER) != 0:
                    continue
                f_k = float(f_comp[k]) if _is_finite(f_comp[k]) else 0.0
                if f_k <= 5.0 and (fl_k & const.FLAG_DEFROST_WP) == 0:
                    continue

                t_top = _optional_float_at(columns, "t_dhw_top_c", k)
                t_bot = _optional_float_at(columns, "t_dhw_bot_c", k)
                sup_k = _optional_float_at(columns, t_sup_key, k)
                if t_top is not None:
                    t_tank = (
                        0.5 * (t_top + t_bot) if t_bot is not None else t_top
                    )
                elif sup_k is not None:
                    t_tank = sup_k - 8.5
                else:
                    t_tank = 45.0

                if t_tank <= DHW_FIT_MAX_TEMP_C:
                    comp_steps.append(k)
                    dhw_temps.append(t_tank)

            if len(comp_steps) >= 4:
                comp_steps = comp_steps[1:]
                dhw_temps = dhw_temps[1:]

            if len(comp_steps) >= 2:
                e_th_wh = 0.0
                e_el_wh = 0.0
                t_outs_d: list[float] = []
                rhs_d: list[float] = []
                for k in comp_steps:
                    ts_k = (
                        int(ts_col[k])
                        if ts_col is not None
                        else k * const.DT_STEP_S
                    )
                    e_th_wh += _step_primary_th_w(columns, k) * dt_h
                    e_el_wh += (
                        corrected_wp_electrical_power_w(
                            bias_ref_params,
                            float(p_raw[k]),
                            float(f_comp[k]),
                            False,
                        )
                        * dt_h
                    )
                    t_outs_d.append(
                        _step_outdoor_temp(
                            columns, k, ts_k, lat_deg=lat_deg, lon_deg=lon_deg
                        )
                    )
                    rh_k = _optional_float_at(columns, "rh_out_pct", k)
                    rhs_d.append(rh_k if rh_k is not None else 80.0)

                if e_el_wh > 50.0 and e_th_wh > 50.0:
                    cop_obs = e_th_wh / e_el_wh
                    if 1.0 <= cop_obs <= 6.5:
                        ts_end = (
                            int(ts_col[comp_steps[-1]])
                            if ts_col is not None
                            else comp_steps[-1] * const.DT_STEP_S
                        )
                        t_s = dhw_temps[0]
                        t_e = max(
                            t_s + 1.0, min(DHW_FIT_MAX_TEMP_C, dhw_temps[-1])
                        )
                        dhw_runs.append(
                            (
                                ts_end,
                                sum(t_outs_d) / len(t_outs_d),
                                t_s,
                                t_e,
                                sum(rhs_d) / len(rhs_d),
                                cop_obs,
                            )
                        )
            idx = end_i
            continue

        idx += 1

    return cv_runs, dhw_runs


def _fit_cv_carnot_huber(
    cv_runs: Sequence[_CvRunSample],
    mu_prior: Mapping[str, float],
    tau_prior: Mapping[str, float],
) -> tuple[float, float, float, float]:
    """Fits `(eta_carnot_cv, beta_defrost)` via 2D Box-Projected Huber LM."""
    sigma_cop = 0.20
    inv_var = 1.0 / (sigma_cop * sigma_cop)
    huber_delta = 1.35
    lam = 1e-2

    bnd_eta = priors.COP_PARAM_BOUNDS["eta_carnot_cv"]
    bnd_beta = priors.COP_PARAM_BOUNDS["beta_defrost"]

    eta_cv = float(mu_prior["eta_carnot_cv"])
    beta_def = float(mu_prior["beta_defrost"])
    fisher_eta = 0.0
    fisher_beta = 0.0

    for _ in range(20):
        h00 = float(tau_prior["eta_carnot_cv"])
        h01 = 0.0
        h11 = float(tau_prior["beta_defrost"])
        g0 = h00 * (eta_cv - mu_prior["eta_carnot_cv"])
        g1 = h11 * (beta_def - mu_prior["beta_defrost"])
        f_e = 0.0
        f_b = 0.0

        for _, t_out_r, t_sup_r, rh_r, y_obs in cv_runs:
            t_evap_c = t_out_r - 4.0
            lift_k = max(14.0, (t_sup_r + 3.0) - t_evap_c)
            c_carnot = ((t_evap_c + 273.15) + lift_k) / lift_k
            phi = frost_penalty_factor(t_out_r, rh_r)
            y_hat = eta_cv * c_carnot * (1.0 - beta_def * phi)
            err = y_hat - y_obs
            z = abs(err) / sigma_cop
            w = 1.0 if z <= huber_delta else huber_delta / z
            w_eff = w * inv_var

            j0 = c_carnot * (1.0 - beta_def * phi)
            j1 = -eta_cv * c_carnot * phi

            f_e += w_eff * j0 * j0
            f_b += w_eff * j1 * j1
            h00 += w_eff * j0 * j0
            h01 += w_eff * j0 * j1
            h11 += w_eff * j1 * j1
            g0 += w_eff * err * j0
            g1 += w_eff * err * j1

        fisher_eta = f_e
        fisher_beta = f_b

        a00 = h00 + lam
        a11 = h11 + lam
        det = a00 * a11 - h01 * h01
        if det <= 1e-15:
            break
        d0 = -(a11 * g0 - h01 * g1) / det
        d1 = -(-h01 * g0 + a00 * g1) / det

        eta_next = max(bnd_eta.lower, min(bnd_eta.upper, eta_cv + d0))
        beta_next = max(bnd_beta.lower, min(bnd_beta.upper, beta_def + d1))
        if abs(eta_next - eta_cv) < 1e-7 and abs(beta_next - beta_def) < 1e-7:
            eta_cv, beta_def = eta_next, beta_next
            break
        eta_cv, beta_def = eta_next, beta_next

    return eta_cv, beta_def, fisher_eta, fisher_beta


def _fit_dhw_carnot_huber(
    dhw_runs: Sequence[_DhwRunSample],
    beta_def: float,
    mu_prior_dhw: float,
    tau_prior_dhw: float,
) -> tuple[float, float]:
    """Fits `eta_carnot_dhw` via 1D Box-Projected Huber IRLS."""
    sigma_cop = 0.20
    inv_var = 1.0 / (sigma_cop * sigma_cop)
    huber_delta = 1.35
    bnd_dhw = priors.COP_PARAM_BOUNDS["eta_carnot_dhw"]

    eta_dhw = mu_prior_dhw
    fisher_dhw = 0.0
    for _ in range(15):
        h_d = tau_prior_dhw
        g_d = tau_prior_dhw * (eta_dhw - mu_prior_dhw)
        f_d = 0.0
        for _, t_out_r, t_s_r, t_e_r, rh_r, y_obs in dhw_runs:
            t_evap_c = t_out_r - 4.0
            t_cond_mean_c = (
                0.5 * (t_s_r + min(t_e_r, DHW_COMP_MAX_TEMP_C)) + 8.5
            )
            lift_k = max(14.0, t_cond_mean_c - t_evap_c)
            c_carnot = ((t_evap_c + 273.15) + lift_k) / lift_k
            frost_m = 1.0 - beta_def * frost_penalty_factor(t_out_r, rh_r)
            j_d = c_carnot * frost_m
            y_hat = eta_dhw * j_d
            err = y_hat - y_obs
            z = abs(err) / sigma_cop
            w = 1.0 if z <= huber_delta else huber_delta / z
            w_eff = w * inv_var
            f_d += w_eff * j_d * j_d
            h_d += w_eff * j_d * j_d
            g_d += w_eff * err * j_d
        fisher_dhw = f_d
        if h_d <= 1e-15:
            break
        eta_next = max(
            bnd_dhw.lower, min(bnd_dhw.upper, eta_dhw - g_d / h_d)
        )
        if abs(eta_next - eta_dhw) < 1e-7:
            eta_dhw = eta_next
            break
        eta_dhw = eta_next

    return eta_dhw, fisher_dhw


def fit_cop_parameters(
    columns: Mapping[str, Sequence[float | int]],
    current_params: contracts.CopParameters,
    memory: contracts.ParameterPosteriorMemory | None = None,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
    eval_columns: Mapping[str, Sequence[float | int]] | None = None,
) -> tuple[
    contracts.CopParameters,
    contracts.ParameterPosteriorMemory,
    dict[str, Any],
]:
    """Calibrates `CopParameters` via Huber IRLS/LM and seasonal memory.

    Args:
        columns: Columnar telemetry mapping from `ColumnarRingBuffer`.
        current_params: Currently active `CopParameters`.
        memory: Optional precision-weighted seasonal posterior memory.
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.
        eval_columns: Optional full column mapping used to compute 14-day
            run-COP RMSE diagnostics (`rmse_cop_14d`, `prior_rmse_cop_14d`).

    Returns:
        A tuple `(fitted_params, updated_memory, diagnostics)`, where
        `diagnostics` includes `rmse_cop_14d`, `prior_rmse_cop_14d`,
        `n_cv_runs`, `n_dhw_runs`, `p1_bias_calibrated_w`, and `posterior_std`.
    """
    mu_prior, tau_prior, tau_day1 = preprocessing.age_precision_memory(
        memory, priors.COP_PARAM_BOUNDS, MEMORY_DECAY_GAMMA
    )

    # 1. Calibrate electrical bias from P1 nocturnal flanks (information-gated).
    p1_bias_meas = calibrate_p1_power_bias(columns)
    bnd_bias = priors.COP_PARAM_BOUNDS["delta_p_wp_bias_w"]
    if p1_bias_meas is not None:
        tau_obs_bias = 1.0 / (10.0 * 10.0)
        tau_post_bias = min(
            25.0 * tau_day1["delta_p_wp_bias_w"],
            tau_prior["delta_p_wp_bias_w"] + tau_obs_bias,
        )
        bias_w = (
            tau_prior["delta_p_wp_bias_w"] * mu_prior["delta_p_wp_bias_w"]
            + tau_obs_bias * p1_bias_meas
        ) / (tau_prior["delta_p_wp_bias_w"] + tau_obs_bias)
        bias_w = max(bnd_bias.lower, min(bnd_bias.upper, bias_w))
        mem_bias_mean = bias_w
        mem_bias_tau = tau_post_bias
    elif memory is not None and "delta_p_wp_bias_w" in memory.mean:
        bias_w = max(
            bnd_bias.lower,
            min(bnd_bias.upper, float(memory.mean["delta_p_wp_bias_w"])),
        )
        mem_bias_mean = bias_w
        mem_bias_tau = tau_prior["delta_p_wp_bias_w"]
    else:
        bias_w = current_params.delta_p_wp_bias_w
        mem_bias_mean = bias_w
        mem_bias_tau = tau_prior["delta_p_wp_bias_w"]

    bias_ref_params = contracts.CopParameters(
        eta_carnot_cv=current_params.eta_carnot_cv,
        beta_defrost=current_params.beta_defrost,
        eta_carnot_dhw=current_params.eta_carnot_dhw,
        delta_p_wp_bias_w=bias_w,
    )

    # 2. Extract qualifying CV and DHW compressor runs.
    cv_runs, dhw_runs = _extract_cop_runs(
        columns, bias_ref_params, lat_deg=lat_deg, lon_deg=lon_deg
    )

    # 3. Fit CV Carnot + defrost penalty with Fisher information gating.
    if cv_runs:
        eta_cv, beta_def, fisher_eta_cv, fisher_beta = _fit_cv_carnot_huber(
            cv_runs, mu_prior, tau_prior
        )
    else:
        eta_cv = current_params.eta_carnot_cv
        beta_def = current_params.beta_defrost
        fisher_eta_cv = 0.0
        fisher_beta = 0.0

    if fisher_eta_cv >= 1e-3 * tau_day1["eta_carnot_cv"]:
        mem_eta_cv_mean = eta_cv
        mem_eta_cv_tau = min(
            25.0 * tau_day1["eta_carnot_cv"],
            tau_prior["eta_carnot_cv"] + fisher_eta_cv,
        )
    else:
        eta_cv = (
            float(memory.mean["eta_carnot_cv"])
            if memory is not None and "eta_carnot_cv" in memory.mean
            else current_params.eta_carnot_cv
        )
        mem_eta_cv_mean = eta_cv
        mem_eta_cv_tau = tau_prior["eta_carnot_cv"]

    if fisher_beta >= 1e-3 * tau_day1["beta_defrost"]:
        mem_beta_mean = beta_def
        mem_beta_tau = min(
            25.0 * tau_day1["beta_defrost"],
            tau_prior["beta_defrost"] + fisher_beta,
        )
    else:
        beta_def = (
            float(memory.mean["beta_defrost"])
            if memory is not None and "beta_defrost" in memory.mean
            else current_params.beta_defrost
        )
        mem_beta_mean = beta_def
        mem_beta_tau = tau_prior["beta_defrost"]

    # 4. Fit DHW Carnot efficiency with Fisher information gating.
    if dhw_runs:
        eta_dhw, fisher_eta_dhw = _fit_dhw_carnot_huber(
            dhw_runs,
            beta_def,
            mu_prior["eta_carnot_dhw"],
            tau_prior["eta_carnot_dhw"],
        )
    else:
        eta_dhw = current_params.eta_carnot_dhw
        fisher_eta_dhw = 0.0

    if fisher_eta_dhw >= 1e-3 * tau_day1["eta_carnot_dhw"]:
        mem_eta_dhw_mean = eta_dhw
        mem_eta_dhw_tau = min(
            25.0 * tau_day1["eta_carnot_dhw"],
            tau_prior["eta_carnot_dhw"] + fisher_eta_dhw,
        )
    else:
        eta_dhw = (
            float(memory.mean["eta_carnot_dhw"])
            if memory is not None and "eta_carnot_dhw" in memory.mean
            else current_params.eta_carnot_dhw
        )
        mem_eta_dhw_mean = eta_dhw
        mem_eta_dhw_tau = tau_prior["eta_carnot_dhw"]

    fitted = contracts.CopParameters(
        eta_carnot_cv=eta_cv,
        beta_defrost=beta_def,
        eta_carnot_dhw=eta_dhw,
        delta_p_wp_bias_w=bias_w,
    )
    new_memory = contracts.ParameterPosteriorMemory(
        mean={
            "eta_carnot_cv": mem_eta_cv_mean,
            "beta_defrost": mem_beta_mean,
            "eta_carnot_dhw": mem_eta_dhw_mean,
            "delta_p_wp_bias_w": mem_bias_mean,
        },
        precision={
            "eta_carnot_cv": mem_eta_cv_tau,
            "beta_defrost": mem_beta_tau,
            "eta_carnot_dhw": mem_eta_dhw_tau,
            "delta_p_wp_bias_w": mem_bias_tau,
        },
    )

    # 5. Evaluate 14-day run-COP RMSE (active fitted vs Day-1 priors).
    eval_src = eval_columns if eval_columns is not None else columns
    if eval_src is columns:
        eval_cv_runs, eval_dhw_runs = cv_runs, dhw_runs
    else:
        eval_cv_runs, eval_dhw_runs = _extract_cop_runs(
            eval_src, bias_ref_params, lat_deg=lat_deg, lon_deg=lon_deg
        )

    all_end_ts = [r[0] for r in eval_cv_runs] + [r[0] for r in eval_dhw_runs]
    ts_eval_col = eval_src.get("ts_utc")
    if ts_eval_col:
        cutoff_14d = int(ts_eval_col[-1]) - 14 * 86400
    else:
        cutoff_14d = (max(all_end_ts) - 14 * 86400) if all_end_ts else 0

    sq_err_fit: list[float] = []
    sq_err_prior: list[float] = []
    day1 = priors.DAY1_COP_PARAMS
    for ts_e, t_out_r, t_sup_r, rh_r, y_obs in eval_cv_runs:
        if ts_e >= cutoff_14d:
            sq_err_fit.append(
                (predict_cop_cv(fitted, t_out_r, t_sup_r, rh_r) - y_obs) ** 2
            )
            sq_err_prior.append(
                (predict_cop_cv(day1, t_out_r, t_sup_r, rh_r) - y_obs) ** 2
            )
    for ts_e, t_out_r, t_s_r, t_e_r, rh_r, y_obs in eval_dhw_runs:
        if ts_e >= cutoff_14d:
            sq_err_fit.append(
                (
                    predict_cop_dhw(fitted, t_out_r, t_s_r, t_e_r, rh_r)
                    - y_obs
                )
                ** 2
            )
            sq_err_prior.append(
                (predict_cop_dhw(day1, t_out_r, t_s_r, t_e_r, rh_r) - y_obs)
                ** 2
            )

    rmse_cop_14d = (
        math.sqrt(sum(sq_err_fit) / len(sq_err_fit)) if sq_err_fit else None
    )
    prior_rmse_cop_14d = (
        math.sqrt(sum(sq_err_prior) / len(sq_err_prior))
        if sq_err_prior
        else None
    )

    diagnostics: dict[str, Any] = {
        "rmse_cop_14d": rmse_cop_14d,
        "prior_rmse_cop_14d": prior_rmse_cop_14d,
        "n_cv_runs": len(cv_runs),
        "n_dhw_runs": len(dhw_runs),
        "p1_bias_calibrated_w": p1_bias_meas,
        "posterior_std": {
            k: 1.0 / math.sqrt(max(1e-12, v))
            for k, v in new_memory.precision.items()
        },
    }
    return fitted, new_memory, diagnostics


fit_cop_model = fit_cop_parameters
