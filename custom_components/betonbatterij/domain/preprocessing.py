"""Telemetry ETL, time/zone normalization, and RingBuffer preprocessing.

Owns all pure-Python preprocessing steps shared across Kalman estimation,
hindcast evaluation, COP calibration, and batch parameter identification:
- Local wall-clock conversion (`local_hour_minute`) with DST fallback,
- Canonical zone-to-hydraulic-group normalization (`zone_to_hydraulic_group`),
- Strict finite-number and raw 5-minute step parsing (`parse_raw_step`),
- Legacy/current LTS backfill detection (`detect_lts_and_missing_prim_rows`),
- Chronological raw-aware train/holdout splitting (`compute_train_holdout_split`),
- Shared 52-week Bayesian precision aging (`age_precision_memory`), and
- Full/incremental columnar history preprocessing (`PreprocessedHistory`,
  `preprocess_history_columns`, `recompute_history_for_khx`,
  `build_zone_exog_list`).

Does not own Kalman filter recursion (`zoh_kalman.py`, `kalman_tape.py`),
hindcast scoring (`evaluation.py`), or file/JSON I/O (`reporting.py`).

Units: timestamps in UTC seconds, temperatures in °C, thermal/electrical powers
in W, wind speeds in m/s, irradiance in W/m².
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import functools
import math
from typing import TYPE_CHECKING, Any, TypeGuard
import zoneinfo

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import hydraulics
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import solar_weather

if TYPE_CHECKING:
    from custom_components.betonbatterij.domain import kalman_tape

EVAL_14D_STEPS: int = 14 * const.STEPS_PER_DAY  # 4,032 steps
HORIZON_6H_STEPS: int = 72  # 6.0 hours
COOLING_TAIL_STEPS: int = 24  # 2.0 hours post-cooling exclusion
MEMORY_WEEKLY_GAMMA: float = 2.0 ** (-1.0 / 52.0)  # 52-week half-life (~0.9868)

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


# ============================================================================
# Timezone, Zone Normalization, Number Parsing & Bayesian Memory Aging
# ============================================================================
@functools.lru_cache(maxsize=8)
def _get_zoneinfo(tz_name: str) -> zoneinfo.ZoneInfo | None:
    """Returns a cached `ZoneInfo` instance, or `None` if unavailable."""
    try:
        return zoneinfo.ZoneInfo(tz_name)
    except (zoneinfo.ZoneInfoNotFoundError, KeyError, ValueError):
        return None


@functools.lru_cache(maxsize=16)
def _eu_dst_bounds_utc(year: int) -> tuple[int, int]:
    """Computes EU CEST start and end Unix UTC timestamps for `year`."""
    utc = datetime.timezone.utc
    mar31_wd = datetime.datetime(year, 3, 31, tzinfo=utc).weekday()
    last_sun_mar = 31 - ((mar31_wd + 1) % 7)
    dst_start = int(
        datetime.datetime(year, 3, last_sun_mar, 1, 0, tzinfo=utc).timestamp()
    )
    oct31_wd = datetime.datetime(year, 10, 31, tzinfo=utc).weekday()
    last_sun_oct = 31 - ((oct31_wd + 1) % 7)
    dst_end = int(
        datetime.datetime(year, 10, last_sun_oct, 1, 0, tzinfo=utc).timestamp()
    )
    return dst_start, dst_end


def local_hour_minute(
    ts_utc: int,
    tz_name: str = priors.TIMEZONE_NAME,
) -> tuple[int, int]:
    """Converts a Unix UTC timestamp to local `(hour, minute)`.

    Uses cached `zoneinfo.ZoneInfo` when OS timezone data is available and falls
    back to exact European Union CET/CEST transition rules on bare Windows
    installs without the external `tzdata` package.

    Args:
        ts_utc: Unix timestamp in seconds (UTC).
        tz_name: IANA timezone identifier (defaults to `Europe/Amsterdam`).

    Returns:
        A tuple `(hour, minute)` in local wall-clock time (`0..23`, `0..59`).
    """
    tz_obj = _get_zoneinfo(tz_name)
    if tz_obj is not None:
        dt_local = datetime.datetime.fromtimestamp(ts_utc, tz=tz_obj)
        return dt_local.hour, dt_local.minute

    utc = datetime.timezone.utc
    dt_utc = datetime.datetime.fromtimestamp(ts_utc, tz=utc)
    dst_start, dst_end = _eu_dst_bounds_utc(dt_utc.year)
    offset_s = 7200 if dst_start <= ts_utc < dst_end else 3600
    sec_of_day = math.floor(ts_utc + offset_s) % 86400
    return sec_of_day // 3600, (sec_of_day % 3600) // 60


def zone_to_hydraulic_group(zone: str) -> str:
    """Maps a zone or group identifier to its canonical hydraulic group key."""
    return "kantoor" if zone == "kt" else zone


def is_finite_number(val: Any) -> TypeGuard[float | int]:
    """Returns True if `val` is a finite non-boolean `int` or `float`."""
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(val)
    )


def opt_float(val: Any, min_valid: float = -100.0) -> float | None:
    """Returns `float(val)` if finite, non-boolean, and `>= min_valid`."""
    if not is_finite_number(val):
        return None
    fval = float(val)
    return fval if fval >= min_valid else None


def opt_nonneg_float(val: Any) -> float | None:
    """Returns `float(val)` if finite, non-boolean, and `>= 0.0`, else None."""
    return opt_float(val, min_valid=0.0)


def opt_temp_c(val: Any, min_valid: float = -100.0) -> float | None:
    """Returns a finite temperature reading [°C] (`>= min_valid`), else None."""
    return opt_float(val, min_valid=min_valid)


def safe_float(val: Any, default: float = 0.0) -> float:
    """Returns `float(val)` if finite and non-boolean, else `default`."""
    return float(val) if is_finite_number(val) else default


def row_float(
    raw_row: Mapping[str, Any],
    key: str,
    default: float = 0.0,
    alt_key: str | None = None,
) -> float:
    """Reads a finite float from `raw_row[key]` (or `alt_key`)."""
    val = raw_row.get(key)
    if val is None and alt_key is not None:
        val = raw_row.get(alt_key)
    return safe_float(val, default)


def age_precision_memory(
    old_memory: contracts.ParameterPosteriorMemory | None,
    bounds: Mapping[str, contracts.ParameterBounds],
    gamma: float = MEMORY_WEEKLY_GAMMA,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Ages Bayesian precision-weighted posterior memory toward Day-1 priors.

    Args:
        old_memory: Previously persisted seasonal posterior memory, or `None`.
        bounds: Parameter box bounds and Day-1 Gaussian priors per parameter.
        gamma: Weekly precision retention factor (`2**(-1/52)` for 52w
            half-life).

    Returns:
        Tuple `(aged_mean, aged_precision, day1_precision)` keyed by parameter
        name.
    """
    prior_mean: dict[str, float] = {}
    prior_prec: dict[str, float] = {}
    day1_prec: dict[str, float] = {}

    for name, bnd in bounds.items():
        tau_0 = 1.0 / max(1e-12, bnd.prior_std * bnd.prior_std)
        day1_prec[name] = tau_0
        if (
            old_memory is not None
            and name in old_memory.mean
            and name in old_memory.precision
        ):
            tau_mem = max(0.1 * tau_0, float(old_memory.precision[name]))
            mu_mem = float(old_memory.mean[name])
            tau_w = (1.0 - gamma) * tau_0 + gamma * tau_mem
            mu_w = (
                (1.0 - gamma) * tau_0 * bnd.prior_mean + gamma * tau_mem * mu_mem
            ) / tau_w
            prior_mean[name] = min(bnd.upper, max(bnd.lower, mu_w))
            prior_prec[name] = tau_w
        else:
            prior_mean[name] = bnd.prior_mean
            prior_prec[name] = tau_0

    return prior_mean, prior_prec, day1_prec


@dataclasses.dataclass(frozen=True, slots=True)
class RawStepParsed:
    """Canonical parsed inputs for a single real-time 5-minute step.

    Attributes:
        ts_utc: Window-end Unix UTC timestamp [s].
        has_flags: True when the raw row included an explicit `flags` bitmask.
        flags: Status bitmask for this step.
        aanwezig: Household occupancy flag.
        t_dak_c: Optional primary roof/garden outdoor sensor [°C].
        t_weather_c: Optional meteorological outdoor temperature [°C].
        t_hm_c: Optional HeishaMon outdoor sensor [°C].
        v_wind_ms: Non-negative outdoor wind speed [m/s].
        has_preproj_solar: True when pre-projected `i_sol_*` replace `p_pv_w`.
        i_sol_wk_preproj_wm2: Pre-projected `wk` irradiance if provided [W/m²].
        i_sol_kt_preproj_wm2: Pre-projected `kt` irradiance if provided [W/m²].
        p_pv_w: Measured PV inverter AC power [W] (NaN if explicitly `None`).
        p_pv_forecast_w: Forecast.Solar power estimate [W].
        epex_eur_kwh: Day-ahead electricity price [EUR/kWh].
        screen_small_pos: Small exterior screen opening `[0..100]`.
        screen_large_pos: Large exterior screen opening `[0..100]`.
        t_air_wk_c: Living-room air temperature [°C].
        t_air_kt_c: Attic-office air temperature [°C].
        room_temps: Room air temperature mapping across all 10 groups [°C].
        u_pump_sec: Secondary circulation pump duty `[0..1]`.
        u_valves: Effective valve openings `[0..1]` across all 10 groups.
        threeway_room: True when primary 3-way valve is set to `Room`.
        wp_defrost: True when primary heat pump defrost is active.
        v_prim_l_min: Primary heat-pump water flow rate [L/min].
        p_prim_th_w: Integrated primary thermal power [W].
        f_comp_hz: Compressor operating frequency [Hz].
        t_buf_top_start_c: Buffer top temperature at step start [°C].
        t_buf_top_end_c: Buffer top temperature at step end [°C].
        t_buf_bot_start_c: Raw buffer bottom temperature at step start [°C].
        t_buf_bot_end_c: Raw buffer bottom temperature at step end [°C].
        t_bg_sup_c: Optional clamp-on BG manifold supply sensor [°C].
        t_bg_ret_c: Optional clamp-on BG manifold return sensor [°C].
        p_el_ac_kt_w: Split-AC electrical power [W].
        ac_mode_code: Numeric AC mode code.
        p_desk_kt_w: Non-negative desk smart-plug power [W], or `None` if
            missing or negative (activating PIR fallback).
        pir_occ_kt: Split-AC radar occupancy fraction `[0..1]`.
        p_growlights_kt_w: Office grow-light electrical power [W].
        ac_defrost: True when split-AC defrost is active.
        t_surf_meas_wk_c: Optional PVC floor surface sensor reading in `wk` [°C].
        q_fh_wk_override_w: Optional explicit `q_fh_wk_w` column value [W].
        q_fh_kt_override_w: Optional explicit `q_fh_kt_w` column value [W].
    """

    ts_utc: int
    has_flags: bool
    flags: int
    aanwezig: bool
    t_dak_c: float | None
    t_weather_c: float | None
    t_hm_c: float | None
    v_wind_ms: float
    has_preproj_solar: bool
    i_sol_wk_preproj_wm2: float
    i_sol_kt_preproj_wm2: float
    p_pv_w: float
    p_pv_forecast_w: float
    epex_eur_kwh: float
    screen_small_pos: float
    screen_large_pos: float
    t_air_wk_c: float
    t_air_kt_c: float
    room_temps: dict[str, float | None]
    u_pump_sec: float
    u_valves: dict[str, float]
    threeway_room: bool
    wp_defrost: bool
    v_prim_l_min: float
    p_prim_th_w: float
    f_comp_hz: float
    t_buf_top_start_c: float
    t_buf_top_end_c: float
    t_buf_bot_start_c: float
    t_buf_bot_end_c: float
    t_bg_sup_c: float | None
    t_bg_ret_c: float | None
    p_el_ac_kt_w: float
    ac_mode_code: float
    p_desk_kt_w: float | None
    pir_occ_kt: float
    p_growlights_kt_w: float
    ac_defrost: bool
    t_surf_meas_wk_c: float | None
    q_fh_wk_override_w: float
    q_fh_kt_override_w: float


def parse_raw_step(
    raw_row: Mapping[str, Any],
    *,
    default_ts_utc: int = 0,
    default_t_wk: float = 19.5,
    default_t_kt: float = 19.0,
) -> RawStepParsed:
    """Parses one raw 5-minute row with unified bounds and glitch rejection.

    Applies `opt_nonneg_float` to `p_desk_kt_w` so negative smart-plug glitches
    are rejected consistently with `preprocess_history_columns`, triggering PIR
    fallback cleanly.

    Args:
        raw_row: Raw 5-minute accumulator or RingBuffer row mapping.
        default_ts_utc: Fallback UTC timestamp when `ts_utc` is omitted.
        default_t_wk: Fallback living-room air temperature [°C].
        default_t_kt: Fallback attic-office air temperature [°C].

    Returns:
        Frozen `RawStepParsed` structure.
    """
    ts_utc = int(raw_row.get("ts_utc", default_ts_utc))
    has_flags = "flags" in raw_row
    default_fl = (
        const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
    )
    fl = int(raw_row["flags"]) if has_flags else default_fl
    aanwezig = (
        bool(fl & const.FLAG_AANWEZIG)
        if has_flags
        else bool(raw_row.get("aanwezig", True))
    )

    t_dak = opt_temp_c(raw_row.get("t_out_dak_c"))
    t_wea = opt_temp_c(raw_row.get("t_out_weather_c"))
    t_hm = opt_temp_c(raw_row.get("t_out_hm_c"))
    if t_dak is None and t_wea is None and t_hm is None:
        t_dak = opt_temp_c(raw_row.get("t_out_c"))

    v_wind = max(0.0, row_float(raw_row, "v_wind_ms", 3.0))

    has_preproj_solar = (
        "i_sol_wk_wm2" in raw_row
        and "i_sol_kt_wm2" in raw_row
        and "p_pv_w" not in raw_row
    )
    if has_preproj_solar:
        i_sol_wk_pre = max(0.0, row_float(raw_row, "i_sol_wk_wm2", 0.0))
        i_sol_kt_pre = max(0.0, row_float(raw_row, "i_sol_kt_wm2", 0.0))
        p_pv = 0.0
    else:
        i_sol_wk_pre = 0.0
        i_sol_kt_pre = 0.0
        if "p_pv_w" in raw_row:
            raw_pv = raw_row["p_pv_w"]
            p_pv = math.nan if raw_pv is None else float(raw_pv)
        else:
            p_pv = 0.0

    t_wk = row_float(raw_row, "t_air_wk_c", default_t_wk)
    t_kt = row_float(raw_row, "t_air_kantoor_c", default_t_kt, "t_air_kt_c")
    room_temps: dict[str, float | None] = {
        "wk": t_wk,
        "gang": t_wk - 0.8,
        "slaapkamer": row_float(raw_row, "t_air_slaapkamer_c", 18.0),
        "badkamer": row_float(raw_row, "t_air_badkamer_c", 18.5),
        "werkkamer_1e": row_float(raw_row, "t_air_werkkamer_1e_c", 18.0),
        "kamer_1e_midden": row_float(raw_row, "t_air_kamer_1e_midden_c", 18.0),
        "kamer_1e_zuid": row_float(raw_row, "t_air_kamer_1e_zuid_c", 18.0),
        "kantoor": t_kt,
        "badkamer_boven": row_float(raw_row, "t_air_badkamer_boven_c", 18.0),
        "logeerkamer": row_float(raw_row, "t_air_logeerkamer_c", 17.5),
    }

    u_bg = min(1.0, max(0.0, row_float(raw_row, "u_eff_bg", 0.0, "u_eff_wk")))
    u_kt = min(
        1.0, max(0.0, row_float(raw_row, "u_eff_kantoor", 0.0, "u_eff_kt"))
    )
    u_pump = min(1.0, max(0.0, row_float(raw_row, "u_pump_sec", 0.0)))
    u_valves: dict[str, float] = {
        "wk": u_bg,
        "gang": u_bg,
        "slaapkamer": min(
            1.0, max(0.0, row_float(raw_row, "u_eff_slaapkamer", 0.0))
        ),
        "badkamer": min(
            1.0, max(0.0, row_float(raw_row, "u_eff_badkamer", 0.0))
        ),
        "werkkamer_1e": min(
            1.0, max(0.0, row_float(raw_row, "u_eff_werkkamer_1e", 0.0))
        ),
        "kamer_1e_midden": min(
            1.0, max(0.0, row_float(raw_row, "u_eff_kamer_1e_midden", 0.0))
        ),
        "kamer_1e_zuid": min(
            1.0, max(0.0, row_float(raw_row, "u_eff_kamer_1e_zuid", 0.0))
        ),
        "kantoor": u_kt,
        "badkamer_boven": min(
            1.0, max(0.0, row_float(raw_row, "u_eff_badkamer_boven", 0.0))
        ),
        "logeerkamer": min(
            1.0, max(0.0, row_float(raw_row, "u_eff_logeerkamer", 0.0))
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

    t_top_e = row_float(raw_row, "t_buf_top_end_c", 22.0, "t_buf_top_c")
    t_top_s = row_float(raw_row, "t_buf_top_start_c", t_top_e)
    t_bot_e = row_float(
        raw_row, "t_buf_bot_end_c", t_top_e - 1.0, "t_buf_bot_c"
    )
    t_bot_s = row_float(raw_row, "t_buf_bot_start_c", t_bot_e)

    return RawStepParsed(
        ts_utc=ts_utc,
        has_flags=has_flags,
        flags=fl,
        aanwezig=aanwezig,
        t_dak_c=t_dak,
        t_weather_c=t_wea,
        t_hm_c=t_hm,
        v_wind_ms=v_wind,
        has_preproj_solar=has_preproj_solar,
        i_sol_wk_preproj_wm2=i_sol_wk_pre,
        i_sol_kt_preproj_wm2=i_sol_kt_pre,
        p_pv_w=p_pv,
        p_pv_forecast_w=row_float(raw_row, "p_pv_forecast_w", 0.0),
        epex_eur_kwh=row_float(raw_row, "epex_eur_kwh", 0.20),
        screen_small_pos=row_float(raw_row, "screen_small_pos", 100.0),
        screen_large_pos=row_float(raw_row, "screen_large_pos", 100.0),
        t_air_wk_c=t_wk,
        t_air_kt_c=t_kt,
        room_temps=room_temps,
        u_pump_sec=u_pump,
        u_valves=u_valves,
        threeway_room=threeway_room,
        wp_defrost=wp_defrost,
        v_prim_l_min=max(0.0, row_float(raw_row, "v_prim_l_min", 0.0)),
        p_prim_th_w=row_float(raw_row, "p_prim_th_w", 0.0),
        f_comp_hz=max(0.0, row_float(raw_row, "f_comp_hz", 0.0)),
        t_buf_top_start_c=t_top_s,
        t_buf_top_end_c=t_top_e,
        t_buf_bot_start_c=t_bot_s,
        t_buf_bot_end_c=t_bot_e,
        t_bg_sup_c=opt_temp_c(raw_row.get("t_bg_sup_c")),
        t_bg_ret_c=opt_temp_c(raw_row.get("t_bg_ret_c")),
        p_el_ac_kt_w=max(0.0, row_float(raw_row, "p_el_ac_kt_w", 0.0)),
        ac_mode_code=row_float(raw_row, "ac_mode_code", 0.0),
        p_desk_kt_w=opt_nonneg_float(raw_row.get("p_desk_kt_w")),
        pir_occ_kt=row_float(raw_row, "pir_occ_kt", 0.0),
        p_growlights_kt_w=max(
            0.0, row_float(raw_row, "p_growlights_kt_w", 0.0)
        ),
        ac_defrost=bool(fl & const.FLAG_DEFROST_AC),
        t_surf_meas_wk_c=opt_temp_c(raw_row.get("t_floor_surf_meas_wk_c")),
        q_fh_wk_override_w=row_float(raw_row, "q_fh_wk_w", 0.0),
        q_fh_kt_override_w=row_float(raw_row, "q_fh_kt_w", 0.0),
    )


# ============================================================================
# LTS Detection, Train/Holdout Split, and Columnar History Preprocessing
# ============================================================================
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
        t_sup_sec_c: Effective midpoint secondary manifold supply temp [°C].
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
        gamma_cal: Per-step 30-min telescoped calorimetric scale factor [-].
        buf_top_offset_c: Applied additive top 1-Wire buffer offset [°C].
        buf_bot_offset_c: Applied additive bottom 1-Wire buffer offset [°C].
        final_gamma_cal: End-of-slice `HydraulicModel.gamma_cal` [-].
        final_t_floor_proxy_c: End-of-slice unmodeled slab proxies [°C].
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
    tape_wk: kalman_tape.PrecomputedZoneTape | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    tape_kt: kalman_tape.PrecomputedZoneTape | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    gamma_cal: list[float] = dataclasses.field(default_factory=list)
    buf_top_offset_c: float = 0.0
    buf_bot_offset_c: float = priors.BUF_BOT_SENSOR_OFFSET_C
    final_gamma_cal: float = 1.0
    final_t_floor_proxy_c: dict[str, float] = dataclasses.field(
        default_factory=dict
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
    whenever `>= 1` day of non-LTS telemetry (within `>= 4` total days) is
    available.

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

    # Case C1: 1..4 days of raw data (with >= 4d total) -> 65/35 raw-tail split
    # so non-LTS rows are never buried 100% inside holdout.
    min_holdout = HORIZON_6H_STEPS + 24
    if n_raw >= const.STEPS_PER_DAY and n_total >= 4 * const.STEPS_PER_DAY:
        holdout_raw = max(min_holdout, round(0.35 * n_raw))
        holdout_cap = max(min_holdout, n_raw - min_holdout)
        holdout_raw = min(holdout_raw, holdout_cap)
        return max(first_raw_idx + 1, n_total - holdout_raw)

    # Case C2: < 1 day of raw data -> total-length fallback.
    if n_total >= 28 * const.STEPS_PER_DAY:
        return n_total - EVAL_14D_STEPS
    if n_total >= 4 * const.STEPS_PER_DAY:
        train_end = max(2 * const.STEPS_PER_DAY, int(0.70 * n_total))
        if n_total - train_end < min_holdout:
            train_end = n_total - min_holdout
        return train_end
    return n_total


def slice_columns(
    columns: Mapping[str, Sequence[float | int]],
    start: int,
    end: int,
) -> dict[str, list[float | int]]:
    """Returns a `[start:end]` list slice across all columns in `columns`."""
    return {key: list(seq[start:end]) for key, seq in columns.items()}


def _prep_float_col(
    columns: Mapping[str, Sequence[float | int]],
    n: int,
    key: str,
    default: float,
    alt_key: str | None = None,
) -> list[float]:
    """Extracts a clean `n`-element float list from `columns[key]`."""
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


def _prep_float_col_fallback(
    columns: Mapping[str, Sequence[float | int]],
    n: int,
    key: str,
    fb_arr: Sequence[float],
    offset: float = 0.0,
    alt_key: str | None = None,
) -> list[float]:
    """Extracts a clean float list falling back elementwise to `fb_arr + offset`."""
    seq = columns.get(key)
    if seq is None and alt_key is not None:
        seq = columns.get(alt_key)
    if seq is None:
        return (
            list(fb_arr) if offset == 0.0 else [v + offset for v in fb_arr]
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


def preprocess_history_columns(
    columns: Mapping[str, Sequence[float | int]],
    flow_rates_l_min: Mapping[str, float] | None = None,
    k_hx_by_group: Mapping[str, float] | None = None,
    eta_ac_ref: float | None = None,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
    buffer_offsets: contracts.BufferSensorOffsets = priors.DAY1_BUFFER_OFFSETS,
) -> PreprocessedHistory:
    """Preprocesses raw RingBuffer columns into step-aligned physical series.

    Uses `HydraulicModel.step()`'s midpoint buffer supply temperature
    (`hyd_res.t_sup_sec_c`) directly so `t_sup_sec_c` matches the exact
    waterside energy balance across all hydraulic regimes.

    Args:
        columns: Mapping of RingBuffer column names to equal-length sequences.
        flow_rates_l_min: Optional manifold rotameter flow rates [L/min].
        k_hx_by_group: Optional `K_hx` [W/K] overrides per hydraulic group.
        eta_ac_ref: Optional reference split-AC efficiency for `kt`.
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.
        buffer_offsets: Calibrated 1-Wire buffer sensor offsets [°C].

    Returns:
        Populated `PreprocessedHistory` over all rows in `columns`.
    """
    ts_seq = columns.get("ts_utc", ())
    n = len(ts_seq)
    flows = dict(const.DEFAULT_FLOW_RATES_L_MIN)
    if flow_rates_l_min is not None:
        flows.update({k: float(v) for k, v in flow_rates_l_min.items()})

    k_hx_map = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
    if k_hx_by_group is not None:
        for group_name, val in k_hx_by_group.items():
            k_hx_map[zone_to_hydraulic_group(group_name)] = float(val)

    hyd = hydraulics.HydraulicModel(
        flow_rates_l_min=flows,
        k_hx_by_group_w_k=k_hx_map,
        buffer_offsets=buffer_offsets,
    )
    eta_ref = (
        priors.DAY1_PRIORS_KT.eta_ac if eta_ac_ref is None else eta_ac_ref
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
    t_down_wk_c: list[float] = [priors.T_GROUND_MEAN_C] * n
    t_down_kt_c: list[float] = [18.0] * n
    q_int_wk_w: list[float] = [priors.BASELOAD_WK_W] * n
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
    gamma_cal: list[float] = [1.0] * n
    t_surf_meas_wk_c: list[float | None] = [None] * n
    cooling_raw: list[bool] = [False] * n

    hm_ema_c: float | None = None
    interfloor_cache: dict[str, tuple[float, int]] = {}
    desk_state: dict[str, Any] = {}

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

    v_wind_raw = _prep_float_col(columns, n, "v_wind_ms", 3.0)
    pv_fc_raw = _prep_float_col(columns, n, "p_pv_forecast_w", 0.0)
    epex_raw = _prep_float_col(columns, n, "epex_eur_kwh", 0.20)
    scr_s_raw = _prep_float_col(columns, n, "screen_small_pos", 100.0)
    scr_l_raw = _prep_float_col(columns, n, "screen_large_pos", 100.0)
    i_wk_in = (
        _prep_float_col(columns, n, "i_sol_wk_wm2", 0.0)
        if has_preproj_sol
        else []
    )
    i_kt_in = (
        _prep_float_col(columns, n, "i_sol_kt_wm2", 0.0)
        if has_preproj_sol
        else []
    )
    tau_scr_in = (
        _prep_float_col(columns, n, "tau_screen_wk", 1.0)
        if has_preproj_sol
        else []
    )

    t_air_wk_c = _prep_float_col(columns, n, "t_air_wk_c", 19.5)
    t_air_kt_c = _prep_float_col(
        columns, n, "t_air_kantoor_c", 19.0, "t_air_kt_c"
    )
    t_slaap_col = _prep_float_col(columns, n, "t_air_slaapkamer_c", 18.0)
    t_bad_col = _prep_float_col(columns, n, "t_air_badkamer_c", 18.5)
    t_werk1e_col = _prep_float_col(columns, n, "t_air_werkkamer_1e_c", 18.0)
    t_mid1e_col = _prep_float_col(columns, n, "t_air_kamer_1e_midden_c", 18.0)
    t_zuid1e_col = _prep_float_col(columns, n, "t_air_kamer_1e_zuid_c", 18.0)
    t_badbov_col = _prep_float_col(columns, n, "t_air_badkamer_boven_c", 18.0)
    t_logeer_col = _prep_float_col(columns, n, "t_air_logeerkamer_c", 17.5)

    u_bg_raw = _prep_float_col(columns, n, "u_eff_bg", 0.0, "u_eff_wk")
    u_kt_raw = _prep_float_col(columns, n, "u_eff_kantoor", 0.0, "u_eff_kt")
    u_pump_raw = _prep_float_col(columns, n, "u_pump_sec", 0.0)
    u_slaap_col = _prep_float_col(columns, n, "u_eff_slaapkamer", 0.0)
    u_bad_col = _prep_float_col(columns, n, "u_eff_badkamer", 0.0)
    u_werk1e_col = _prep_float_col(columns, n, "u_eff_werkkamer_1e", 0.0)
    u_mid1e_col = _prep_float_col(columns, n, "u_eff_kamer_1e_midden", 0.0)
    u_zuid1e_col = _prep_float_col(columns, n, "u_eff_kamer_1e_zuid", 0.0)
    u_badbov_col = _prep_float_col(columns, n, "u_eff_badkamer_boven", 0.0)
    u_logeer_col = _prep_float_col(columns, n, "u_eff_logeerkamer", 0.0)

    v_prim_raw = _prep_float_col(columns, n, "v_prim_l_min", 0.0)
    p_prim_th_w = _prep_float_col(columns, n, "p_prim_th_w", 0.0)
    f_comp_raw = _prep_float_col(columns, n, "f_comp_hz", 0.0)
    p_wp_raw_in = _prep_float_col(columns, n, "p_wp_raw_el_w", 0.0)

    has_q_fh_wk_col = "q_fh_wk_w" in columns
    has_q_fh_kt_col = "q_fh_kt_w" in columns
    q_fh_wk_in = (
        _prep_float_col(columns, n, "q_fh_wk_w", 0.0) if has_q_fh_wk_col else []
    )
    q_fh_kt_in = (
        _prep_float_col(columns, n, "q_fh_kt_w", 0.0) if has_q_fh_kt_col else []
    )

    t_top_e_col = _prep_float_col(
        columns, n, "t_buf_top_end_c", 22.0, "t_buf_top_c"
    )
    t_top_s_col = _prep_float_col_fallback(
        columns, n, "t_buf_top_start_c", t_top_e_col
    )
    t_bot_e_col = _prep_float_col_fallback(
        columns, n, "t_buf_bot_end_c", t_top_e_col, -1.0, "t_buf_bot_c"
    )
    t_bot_s_col = _prep_float_col_fallback(
        columns, n, "t_buf_bot_start_c", t_bot_e_col
    )

    p_ac_raw = _prep_float_col(columns, n, "p_el_ac_kt_w", 0.0)
    ac_mode_code = _prep_float_col(columns, n, "ac_mode_code", 0.0)
    pir_occ_kt = _prep_float_col(columns, n, "pir_occ_kt", 0.0)
    p_grow_raw = _prep_float_col(columns, n, "p_growlights_kt_w", 0.0)
    d_est_col_wk_w = _prep_float_col(columns, n, "d_est_wk_w", 0.0)
    d_est_col_kt_w = _prep_float_col(columns, n, "d_est_kt_w", 0.0)

    t_floor_wk_running: float = (t_air_wk_c[0] if n > 0 else 19.5) + 0.8
    t_floor_kt_running: float = (t_air_kt_c[0] if n > 0 else 19.0) + 0.5

    default_flags = (
        const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
    )
    c_wk_j = 3600.0 * priors.DAY1_PRIORS_WK.c_floor_wh_k
    h_fh_wk = 1000.0 / priors.DAY1_PRIORS_WK.r_fh_k_kw
    c_kt_j = 3600.0 * priors.DAY1_PRIORS_KT.c_floor_wh_k
    h_fh_kt = 1000.0 / priors.DAY1_PRIORS_KT.r_fh_k_kw
    buf_top_off = buffer_offsets.buf_top_offset_c
    buf_bot_off = buffer_offsets.buf_bot_offset_c
    baseload_wk = priors.BASELOAD_WK_W
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
            opt_float(t_dak_seq[k])
            if t_dak_seq is not None and k < len(t_dak_seq)
            else None
        )
        t_wea = (
            opt_float(t_wea_seq[k])
            if t_wea_seq is not None and k < len(t_wea_seq)
            else None
        )
        t_hm = (
            opt_float(t_hm_seq[k])
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
            t_dak = opt_float(t_out_direct_seq[k])
        if t_hm is not None:
            hm_ema_c = (
                t_hm
                if hm_ema_c is None
                else (t_hm / 6.0) + (5.0 / 6.0) * hm_ema_c
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
        t_buf_top_c[k] = t_top_e + buf_top_off
        t_buf_bot_cal_c[k] = t_bot_e + buf_bot_off

        tf_wk_in = t_floor_wk_running
        tf_kt_in = t_floor_kt_running

        t_bg_sup = (
            opt_float(bg_sup_seq[k], 5.0)
            if bg_sup_seq is not None and k < len(bg_sup_seq)
            else None
        )
        t_bg_ret = (
            opt_float(bg_ret_seq[k], 5.0)
            if bg_ret_seq is not None and k < len(bg_ret_seq)
            else None
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
        t_sup_sec_c[k] = hyd_res.t_sup_sec_c
        gamma_cal[k] = hyd_res.gamma_cal
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
            opt_nonneg_float(desk_seq[k])
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
            t_surf_meas_wk_c[k] = opt_float(surf_seq[k], 5.0)

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
        gamma_cal=gamma_cal,
        buf_top_offset_c=buf_top_off,
        buf_bot_offset_c=buf_bot_off,
        final_gamma_cal=hyd.gamma_cal,
        final_t_floor_proxy_c=dict(hyd.t_floor_proxy_c),
    )


def recompute_history_for_khx(
    base_hist: PreprocessedHistory,
    columns: Mapping[str, Sequence[float | int]],
    flow_rates_l_min: Mapping[str, float] | None = None,
    k_hx_by_group: Mapping[str, float] | None = None,
    buffer_offsets: contracts.BufferSensorOffsets | None = None,
) -> PreprocessedHistory:
    """Recomputes only hydraulic fields on `base_hist` for updated `k_hx`."""
    n = base_hist.n_rows
    flows = dict(const.DEFAULT_FLOW_RATES_L_MIN)
    if flow_rates_l_min is not None:
        flows.update({k: float(v) for k, v in flow_rates_l_min.items()})

    k_hx_map = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
    if k_hx_by_group is not None:
        for group_name, val in k_hx_by_group.items():
            k_hx_map[zone_to_hydraulic_group(group_name)] = float(val)

    eff_offsets = (
        buffer_offsets
        if buffer_offsets is not None
        else contracts.BufferSensorOffsets(
            buf_top_offset_c=base_hist.buf_top_offset_c,
            buf_bot_offset_c=base_hist.buf_bot_offset_c,
        )
    )

    if (
        abs(base_hist.hyd_k_hx_wk - float(k_hx_map["wk"])) <= 1e-9
        and abs(base_hist.hyd_k_hx_kt - float(k_hx_map["kantoor"])) <= 1e-9
        and abs(base_hist.buf_top_offset_c - eff_offsets.buf_top_offset_c)
        <= 1e-9
        and abs(base_hist.buf_bot_offset_c - eff_offsets.buf_bot_offset_c)
        <= 1e-9
    ):
        return base_hist

    hyd = hydraulics.HydraulicModel(
        flow_rates_l_min=flows,
        k_hx_by_group_w_k=k_hx_map,
        buffer_offsets=eff_offsets,
    )

    t_slaap_col = _prep_float_col(columns, n, "t_air_slaapkamer_c", 18.0)
    t_bad_col = _prep_float_col(columns, n, "t_air_badkamer_c", 18.5)
    t_werk1e_col = _prep_float_col(columns, n, "t_air_werkkamer_1e_c", 18.0)
    t_mid1e_col = _prep_float_col(columns, n, "t_air_kamer_1e_midden_c", 18.0)
    t_zuid1e_col = _prep_float_col(columns, n, "t_air_kamer_1e_zuid_c", 18.0)
    t_badbov_col = _prep_float_col(columns, n, "t_air_badkamer_boven_c", 18.0)
    t_logeer_col = _prep_float_col(columns, n, "t_air_logeerkamer_c", 17.5)

    u_slaap_col = _prep_float_col(columns, n, "u_eff_slaapkamer", 0.0)
    u_bad_col = _prep_float_col(columns, n, "u_eff_badkamer", 0.0)
    u_werk1e_col = _prep_float_col(columns, n, "u_eff_werkkamer_1e", 0.0)
    u_mid1e_col = _prep_float_col(columns, n, "u_eff_kamer_1e_midden", 0.0)
    u_zuid1e_col = _prep_float_col(columns, n, "u_eff_kamer_1e_zuid", 0.0)
    u_badbov_col = _prep_float_col(columns, n, "u_eff_badkamer_boven", 0.0)
    u_logeer_col = _prep_float_col(columns, n, "u_eff_logeerkamer", 0.0)

    v_prim_raw = _prep_float_col(columns, n, "v_prim_l_min", 0.0)
    has_q_fh_wk_col = "q_fh_wk_w" in columns
    has_q_fh_kt_col = "q_fh_kt_w" in columns
    q_fh_wk_in = (
        _prep_float_col(columns, n, "q_fh_wk_w", 0.0) if has_q_fh_wk_col else []
    )
    q_fh_kt_in = (
        _prep_float_col(columns, n, "q_fh_kt_w", 0.0) if has_q_fh_kt_col else []
    )

    if "t_buf_top_end_c" in columns or "t_buf_top_c" in columns:
        t_top_e_col = _prep_float_col(
            columns, n, "t_buf_top_end_c", 22.0, "t_buf_top_c"
        )
    else:
        t_top_e_col = [
            v - base_hist.buf_top_offset_c for v in base_hist.t_buf_top_c
        ]
    t_top_s_col = _prep_float_col_fallback(
        columns, n, "t_buf_top_start_c", t_top_e_col
    )
    t_bot_e_col = _prep_float_col_fallback(
        columns, n, "t_buf_bot_end_c", t_top_e_col, -1.0, "t_buf_bot_c"
    )
    t_bot_s_col = _prep_float_col_fallback(
        columns, n, "t_buf_bot_start_c", t_bot_e_col
    )
    buf_top_off = eff_offsets.buf_top_offset_c
    buf_bot_off = eff_offsets.buf_bot_offset_c
    t_buf_top_c = [v + buf_top_off for v in t_top_e_col]
    t_buf_bot_cal_c = [v + buf_bot_off for v in t_bot_e_col]
    bg_sup_seq = columns.get("t_bg_sup_c")
    bg_ret_seq = columns.get("t_bg_ret_c")

    t_sup_sec_c: list[float] = [20.0] * n
    q_fh_wk_hyd_w: list[float] = [0.0] * n
    q_fh_kt_hyd_w: list[float] = [0.0] * n
    q_sec_tot_w: list[float] = [0.0] * n
    f_flow_wk: list[float] = [0.0] * n
    f_flow_kt: list[float] = [0.0] * n
    gamma_cal: list[float] = [1.0] * n
    q_int_wk_w: list[float] = [priors.BASELOAD_WK_W] * n

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
    c_wk_j = 3600.0 * priors.DAY1_PRIORS_WK.c_floor_wh_k
    h_fh_wk = 1000.0 / priors.DAY1_PRIORS_WK.r_fh_k_kw
    c_kt_j = 3600.0 * priors.DAY1_PRIORS_KT.c_floor_wh_k
    h_fh_kt = 1000.0 / priors.DAY1_PRIORS_KT.r_fh_k_kw
    baseload_wk = priors.BASELOAD_WK_W

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
            opt_float(bg_sup_seq[k], 5.0)
            if bg_sup_seq is not None and k < len(bg_sup_seq)
            else None
        )
        t_bg_ret = (
            opt_float(bg_ret_seq[k], 5.0)
            if bg_ret_seq is not None and k < len(bg_ret_seq)
            else None
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
        t_sup_sec_c[k] = hyd_res.t_sup_sec_c
        gamma_cal[k] = hyd_res.gamma_cal
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
            base_hist.tape_wk,
            hyd_k_hx=float(k_hx_map["wk"]),
            t_sup_sec=t_sup_sec_c,
            t_buf_top=t_buf_top_c,
            q_fh_hyd=q_fh_wk_hyd_w,
            q_int=q_int_wk_w,
            gamma_cal=gamma_cal,
        )
        if base_hist.tape_wk is not None
        else None
    )
    cloned_tape_kt = (
        dataclasses.replace(
            base_hist.tape_kt,
            hyd_k_hx=float(k_hx_map["kantoor"]),
            t_sup_sec=t_sup_sec_c,
            t_buf_top=t_buf_top_c,
            q_fh_hyd=q_fh_kt_hyd_w,
            gamma_cal=gamma_cal,
        )
        if base_hist.tape_kt is not None
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
        t_buf_top_c=t_buf_top_c,
        t_buf_bot_cal_c=t_buf_bot_cal_c,
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
        tape_wk=cloned_tape_wk,
        tape_kt=cloned_tape_kt,
        gamma_cal=gamma_cal,
        buf_top_offset_c=buf_top_off,
        buf_bot_offset_c=buf_bot_off,
        final_gamma_cal=hyd.gamma_cal,
        final_t_floor_proxy_c=dict(hyd.t_floor_proxy_c),
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
        gamma_k = hist.gamma_cal[k] if k < len(hist.gamma_cal) else 1.0
        return max(
            0.0,
            (gamma_k * (u_pump * u_eff))
            * params.k_hx_w_k
            * (hist.t_buf_top_c[k] - t_floor_prev_c),
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
    return solar_weather.compute_ac_heating_power_w(
        p_el_ac_w=hist.p_el_ac_kt_w[k],
        ac_mode_code=hist.ac_mode_code[k],
        ac_defrost=ac_def,
        t_out_c=hist.t_out_c[k],
        eta_ac=params.eta_ac,
        base_q_ac_w=hist.q_ac_base_kt_w[k],
    )
