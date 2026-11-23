"""Solar geometry, glazing optics, outdoor selection, and zone load models.

Owns pure-Python NOAA solar position geometry, multi-pane Fresnel incidence
angle modifier (IAM), PV-to-facade/roof irradiance projection with exterior
screen attenuation and negative-EPEX curtailment detection, sun-guarded outdoor
temperature selection, soft 1st-floor interfloor boundaries, and attic-office
internal/split-AC thermal power modeling (`compute_ac_heating_power_w`,
`compute_internal_and_ac_loads_kt`).

Does not own timezone conversions (`preprocessing.py`), PVC floor surface
temperature estimation (`zoh_kalman.py`), or RingBuffer ETL (`preprocessing.py`).

Units: angles in degrees, irradiance in W/m², temperatures in °C, thermal and
electrical powers in W, time in s.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import math
from typing import Any, TypeGuard

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import priors

INTERFLOOR_HOLD_MAX_S: int = 43_200  # 12 h soft hold for 1st-floor ZHA sensors.
DESK_ON_THRESHOLD_W: float = 35.0
DESK_OFF_THRESHOLD_W: float = 22.0
DESK_HOLD_S: float = 600.0  # 10 min coffee-break hold.
DESK_METABOLIC_GAIN_W: float = 85.0
PIR_OCC_THRESHOLD: float = 0.05
PIR_HOLD_S: float = 900.0  # 15 min occupancy hold.
PIR_WORKPLACE_GAIN_W: float = 140.0

_SOUTH_1E_GROUP: tuple[str, ...] = (
    "kamer_1e_midden",
    "kamer_1e_zuid",
    "werkkamer_1e",
)
_NORTH_1E_GROUP: tuple[str, ...] = ("slaapkamer", "badkamer")
_ALL_1E_ROOMS: tuple[str, ...] = (
    "slaapkamer",
    "kamer_1e_zuid",
    "werkkamer_1e",
    "kamer_1e_midden",
    "badkamer",
)


def _is_valid_number(value: float | int | None) -> TypeGuard[float | int]:
    """Checks whether `value` is a finite non-boolean `float` or `int`."""
    return (
        value is not None
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@functools.lru_cache(maxsize=4096)
def solar_position(
    ts_utc: int,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
) -> tuple[float, float]:
    """Computes solar elevation and azimuth via NOAA solar geometry.

    Uses standard NOAA equation-of-time, apparent ecliptic longitude, and
    atmospheric refraction formulas in pure `math`.

    Args:
        ts_utc: Unix timestamp in seconds (UTC).
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.

    Returns:
        A tuple `(elevation_deg, azimuth_deg)` in degrees, where `azimuth_deg`
        is measured clockwise from North (`90.0 = East`, `180.0 = South`,
        `270.0 = West`).
    """
    jd = ts_utc / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude and anomaly [deg].
    l0_deg = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    m_deg = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    m_rad = math.radians(m_deg)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    # Equation of center and apparent solar longitude [deg].
    c_deg = (
        math.sin(m_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2.0 * m_rad) * (0.019993 - 0.000101 * jc)
        + math.sin(3.0 * m_rad) * 0.000289
    )
    true_lon_deg = l0_deg + c_deg
    omega_rad = math.radians(125.04 - 1934.136 * jc)
    app_lon_rad = math.radians(
        true_lon_deg - 0.00569 - 0.00478 * math.sin(omega_rad)
    )

    # Obliquity of the ecliptic [rad].
    eps0_arcsec = 21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))
    eps0_deg = 23.0 + (26.0 + eps0_arcsec / 60.0) / 60.0
    eps_rad = math.radians(eps0_deg + 0.00256 * math.cos(omega_rad))

    # Solar declination [rad].
    sin_delta = max(-1.0, min(1.0, math.sin(eps_rad) * math.sin(app_lon_rad)))
    delta_rad = math.asin(sin_delta)
    cos_delta = math.cos(delta_rad)

    # Equation of time [minutes].
    y_var = math.tan(0.5 * eps_rad) ** 2
    l0_rad = math.radians(l0_deg)
    eq_time_min = 4.0 * math.degrees(
        y_var * math.sin(2.0 * l0_rad)
        - 2.0 * ecc * math.sin(m_rad)
        + 4.0 * ecc * y_var * math.sin(m_rad) * math.cos(2.0 * l0_rad)
        - 0.5 * y_var * y_var * math.sin(4.0 * l0_rad)
        - 1.25 * ecc * ecc * math.sin(2.0 * m_rad)
    )

    utc_minutes = (((ts_utc % 86400) + 86400) % 86400) / 60.0
    true_solar_min = (utc_minutes + eq_time_min + 4.0 * lon_deg) % 1440.0
    ha_deg = true_solar_min / 4.0 - 180.0
    ha_rad = math.radians(ha_deg)

    # Local East-North-Up (ENU) unit vector toward the Sun.
    lat_rad = math.radians(lat_deg)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    cos_ha = math.cos(ha_rad)

    s_up = max(
        -1.0, min(1.0, sin_lat * sin_delta + cos_lat * cos_delta * cos_ha)
    )
    s_east = -cos_delta * math.sin(ha_rad)
    s_north = cos_lat * sin_delta - sin_lat * cos_delta * cos_ha

    elev_geom_deg = math.degrees(math.asin(s_up))

    # NOAA atmospheric refraction [deg].
    if elev_geom_deg > 85.0:
        refrac_deg = 0.0
    elif elev_geom_deg > 5.0:
        tan_e = math.tan(math.radians(elev_geom_deg))
        refrac_deg = (
            58.1 / tan_e - 0.07 / (tan_e**3) + 0.000086 / (tan_e**5)
        ) / 3600.0
    elif elev_geom_deg > -0.575:
        poly = -518.2 + elev_geom_deg * (
            103.4 + elev_geom_deg * (-12.79 + elev_geom_deg * 0.711)
        )
        refrac_deg = (1735.0 + elev_geom_deg * poly) / 3600.0
    else:
        tan_e = math.tan(math.radians(elev_geom_deg))
        refrac_deg = (-20.774 / tan_e) / 3600.0 if abs(tan_e) > 1e-9 else 0.0

    elevation_deg = elev_geom_deg + refrac_deg
    azimuth_deg = math.degrees(math.atan2(s_east, s_north)) % 360.0
    return elevation_deg, azimuth_deg


def fresnel_iam(
    cos_theta: float,
    n_refr: float = 1.526,
    glazing_panes: int = 2,
) -> float:
    """Computes the multi-pane Fresnel incidence angle modifier (IAM).

    Args:
        cos_theta: Cosine of the solar angle of incidence on the surface.
        n_refr: Glass refractive index (`1.526` for soda-lime float glass).
        glazing_panes: Number of glass panes (`2` for HR++ double glazing,
            `1` for a single PV module glass cover).

    Returns:
        Dimensionless transmittance ratio `tau(theta) / tau(0)` in `[0.0, 1.0]`.
    """
    if not math.isfinite(cos_theta) or cos_theta <= 0.0:
        return 0.0
    c1 = min(1.0, float(cos_theta))
    if c1 >= 1.0 - 1e-12:
        return 1.0

    panes = max(1, int(glazing_panes))
    n = max(1.001, float(n_refr))

    sin2_t1 = max(0.0, 1.0 - c1 * c1)
    sin2_t2 = sin2_t1 / (n * n)
    c2 = math.sqrt(max(0.0, 1.0 - sin2_t2))

    # Unpolarized s- and p-interface reflectances.
    r_perp = ((c1 - n * c2) / (c1 + n * c2)) ** 2
    r_par = ((n * c1 - c2) / (n * c1 + c2)) ** 2

    mult = 2 * panes - 1
    tau_perp = (1.0 - r_perp) / (1.0 + mult * r_perp)
    tau_par = (1.0 - r_par) / (1.0 + mult * r_par)
    tau_theta = 0.5 * (tau_perp + tau_par)

    # Normal incidence reference (c1 = c2 = 1).
    r0 = ((1.0 - n) / (1.0 + n)) ** 2
    tau_0 = (1.0 - r0) / (1.0 + mult * r0)
    if tau_0 <= 0.0:
        return 0.0
    return max(0.0, min(1.0, tau_theta / tau_0))


def _angular_diff_deg(az1_deg: float, az2_deg: float) -> float:
    """Computes the shortest signed angular difference in `[-180, 180]` deg."""
    return ((az1_deg - az2_deg + 180.0) % 360.0) - 180.0


def project_solar_irradiance(
    ts_utc: int,
    p_pv_w: float,
    p_pv_forecast_w: float = 0.0,
    epex_eur_kwh: float = 0.20,
    screen_small_pos: float = 100.0,
    screen_large_pos: float = 100.0,
    pv_stale: bool = False,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
) -> tuple[float, float, float, bool]:
    """Projects roof PV output onto living-room glazing and attic roof planes.

    Decomposes `30°/175°` 5.74 kWp roof plane-of-array irradiance via clear-sky
    index `k_c` into direct beam and diffuse sky components, projects onto the
    `90°/175°` vertical South facade (with HR++ vs PV IAM ratio, horizon taper
    `< 5°`, and `0.35/0.65` exterior screen attenuation) and `38°` attic roof,
    and detects negative-EPEX curtailment or stale inverter telemetry.

    Args:
        ts_utc: Unix timestamp in seconds (UTC).
        p_pv_w: Measured PV inverter AC power [W].
        p_pv_forecast_w: Forecast.Solar clear/cloud power estimate [W].
        epex_eur_kwh: Current day-ahead electricity price [EUR/kWh].
        screen_small_pos: Small exterior screen opening (`0..100`, `100 = open`).
        screen_large_pos: Large exterior screen opening (`0..100`, `100 = open`).
        pv_stale: True when PV telemetry has been stale for more than 20 min.
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.

    Returns:
        A tuple `(i_sol_wk_wm2, i_sol_kt_wm2, tau_screen_wk, pv_fallback_used)`,
        where `i_sol_wk_wm2` is screen-attenuated vertical South window
        irradiance [W/m²], `i_sol_kt_wm2` is `38°` attic roof irradiance [W/m²],
        `tau_screen_wk` is effective screen transmittance in `[0.10, 1.0]`, and
        `pv_fallback_used` indicates `p_pv_forecast_w` substitution.
    """
    pos_small = (
        100.0
        if not _is_valid_number(screen_small_pos)
        else max(0.0, min(100.0, float(screen_small_pos)))
    )
    pos_large = (
        100.0
        if not _is_valid_number(screen_large_pos)
        else max(0.0, min(100.0, float(screen_large_pos)))
    )
    open_frac = (
        priors.SCREEN_SMALL_WEIGHT * pos_small
        + priors.SCREEN_LARGE_WEIGHT * pos_large
    ) / 100.0

    # Closed screens transmit 10% of direct beam and 45% of diffuse daylight.
    eta_scr_dir = 0.10 + 0.90 * open_frac
    eta_scr_diff = 0.45 + 0.55 * open_frac

    pv_missing = not _is_valid_number(p_pv_w)
    p_fc_clean = (
        0.0
        if not _is_valid_number(p_pv_forecast_w)
        else max(0.0, float(p_pv_forecast_w))
    )
    epex_clean = (
        0.20 if not _is_valid_number(epex_eur_kwh) else float(epex_eur_kwh)
    )

    curtailment_detected = (
        not pv_missing
        and epex_clean <= 0.015
        and float(p_pv_w) < 80.0
        and p_fc_clean > 350.0
    )
    if pv_stale or pv_missing or curtailment_detected:
        p_eff_w = p_fc_clean
        pv_fallback_used = True
    else:
        p_eff_w = max(0.0, float(p_pv_w))
        pv_fallback_used = False

    if p_eff_w <= 0.0:
        return 0.0, 0.0, eta_scr_dir, pv_fallback_used

    elev_deg, az_deg = solar_position(ts_utc, lat_deg=lat_deg, lon_deg=lon_deg)
    if elev_deg <= 0.0:
        return 0.0, 0.0, eta_scr_dir, pv_fallback_used

    i_dak_wm2 = p_eff_w / priors.PV_PEAK_KWP

    alpha_rad = math.radians(elev_deg)
    sin_alpha = math.sin(alpha_rad)
    cos_alpha = math.cos(alpha_rad)

    beta_pv_rad = math.radians(priors.PV_TILT_DEG)
    d_az_pv_deg = _angular_diff_deg(az_deg, priors.PV_AZIMUTH_DEG)
    cos_d_az_pv = math.cos(math.radians(d_az_pv_deg))
    cos_dak_pos = max(
        0.0,
        sin_alpha * math.cos(beta_pv_rad)
        + cos_alpha * math.sin(beta_pv_rad) * cos_d_az_pv,
    )

    d_az_wk_deg = _angular_diff_deg(az_deg, priors.WINDOW_WK_AZIMUTH_DEG)
    if abs(d_az_wk_deg) >= 90.0:
        cos_wk_pos = 0.0
    else:
        cos_wk_pos = max(0.0, cos_alpha * math.cos(math.radians(d_az_wk_deg)))

    beta_kt_rad = math.radians(priors.ROOF_KT_TILT_DEG)
    cos_kt_pos = max(
        0.0,
        sin_alpha * math.cos(beta_kt_rad)
        + cos_alpha * math.sin(beta_kt_rad) * cos_d_az_pv,
    )

    # Clear-sky POA reference on 30°/175° roof (Kasten-Young AM + Meinel DNI).
    air_mass = 1.0 / (sin_alpha + 0.50572 * ((elev_deg + 6.07995) ** -1.6364))
    dni_clear = 1361.0 * (0.7 ** (air_mass**0.678))
    dhi_clear = 0.12 * 1361.0 * sin_alpha
    svf_pv = 0.5 * (1.0 + math.cos(beta_pv_rad))
    i_dak_clear = max(25.0, dni_clear * cos_dak_pos + dhi_clear * svf_pv)

    # Smooth clear-sky index decomposition into direct and diffuse irradiance.
    k_c = max(0.0, min(1.15, i_dak_wm2 / i_dak_clear))
    if cos_dak_pos <= 0.0 or k_c <= 0.25:
        f_dir = 0.0
    else:
        f_dir_clear = min(0.90, (dni_clear * cos_dak_pos) / i_dak_clear)
        ramp = min(1.0, (k_c - 0.25) / 0.55)
        smooth_ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        f_dir = f_dir_clear * smooth_ramp

    i_dir_dak = f_dir * i_dak_wm2
    i_diff_dak = (1.0 - f_dir) * i_dak_wm2

    # Horizon taper below 5° solar elevation and low-sun denominator clamp.
    w_hor = 1.0 if elev_deg >= 5.0 else max(0.0, elev_deg / 5.0)
    denom_dak = max(0.15, cos_dak_pos)

    if cos_wk_pos > 0.0 and i_dir_dak > 0.0:
        iam_hrpp = fresnel_iam(cos_wk_pos, glazing_panes=2)
        iam_pv = max(0.05, fresnel_iam(cos_dak_pos, glazing_panes=1))
        geom_wk = w_hor * (cos_wk_pos / denom_dak) * (iam_hrpp / iam_pv)
    else:
        geom_wk = 0.0

    i_wk_dir_open = geom_wk * i_dir_dak
    i_wk_diff_open = 0.54 * i_diff_dak
    i_wk_open = i_wk_dir_open + i_wk_diff_open

    i_sol_wk_wm2 = eta_scr_dir * i_wk_dir_open + eta_scr_diff * i_wk_diff_open
    tau_screen_wk = (
        (i_sol_wk_wm2 / i_wk_open) if i_wk_open > 1e-6 else eta_scr_dir
    )

    svf_kt_ratio = (1.0 + math.cos(beta_kt_rad)) / (1.0 + math.cos(beta_pv_rad))
    i_sol_kt_wm2 = (
        w_hor * (cos_kt_pos / denom_dak) * i_dir_dak
        + svf_kt_ratio * i_diff_dak
    )

    return i_sol_wk_wm2, i_sol_kt_wm2, tau_screen_wk, pv_fallback_used


def select_outdoor_temp(
    ts_utc: int,
    t_dak_c: float | None,
    t_weather_c: float | None,
    t_hm_ema_c: float | None,
    hm_bias_c: float = 0.0,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
) -> float:
    """Selects outdoor air temperature with summer sun-guard and fallbacks.

    When `0 < alpha_s < 18°` and `gamma_s < 130°`, direct low eastern sunlight
    can bias `t_dak_c` upward, so it is clamped against `t_hm_ema_c + hm_bias_c + 1.0`.
    Falls back along `t_dak_c -> t_weather_c -> t_hm_ema_c + hm_bias_c -> 10.0`.

    Args:
        ts_utc: Unix timestamp in seconds (UTC).
        t_dak_c: Primary `0.01 °C` roof/garden thermometer reading [°C].
        t_weather_c: Meteorological forecast outdoor temperature [°C].
        t_hm_ema_c: 30-min EMA-filtered HeishaMon outdoor sensor [°C].
        hm_bias_c: Calibrated offset added to `t_hm_ema_c` [°C].
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.

    Returns:
        Best available outdoor dry-bulb temperature [°C].
    """
    bias = 0.0 if not _is_valid_number(hm_bias_c) else float(hm_bias_c)

    if _is_valid_number(t_dak_c):
        t_dak = float(t_dak_c)
        if _is_valid_number(t_hm_ema_c):
            clamp_ref: float | None = float(t_hm_ema_c) + bias + 1.0
        elif _is_valid_number(t_weather_c):
            clamp_ref = float(t_weather_c) + 1.0
        else:
            clamp_ref = None
        if clamp_ref is not None and t_dak > clamp_ref:
            elev_deg, az_deg = solar_position(
                ts_utc, lat_deg=lat_deg, lon_deg=lon_deg
            )
            if 0.0 < elev_deg < 18.0 and az_deg < 130.0:
                return clamp_ref
        return t_dak

    if _is_valid_number(t_weather_c):
        return float(t_weather_c)
    if _is_valid_number(t_hm_ema_c):
        return float(t_hm_ema_c) + bias
    return 10.0


def compute_interfloor_boundaries(
    room_temps_c: Mapping[str, float | None],
    last_valid_cache: dict[str, tuple[float, int]] | None = None,
    ts_utc: int = 0,
) -> tuple[float, float]:
    """Computes soft 1st-floor boundary temperatures `(t_up_wk_c, t_down_kt_c)`.

    Holds each 1st-floor ZHA sensor value for up to 12 hours (`43200 s`) in
    `last_valid_cache` (mutated in place when provided) and falls back to
    North/South 1st-floor peer averages during longer dropouts.

    Args:
        room_temps_c: Mapping from room name (or `t_air_<room>_c`) to [°C].
        last_valid_cache: Optional mutable cache mapping room name to
            `(last_temp_c, last_ts_utc)`.
        ts_utc: Current Unix timestamp in seconds (UTC).

    Returns:
        A tuple `(t_up_wk_c, t_down_kt_c)`, where `t_up_wk_c` is the weighted
        ceiling boundary above `wk`
        (`(2*slaapkamer + kamer_1e_zuid + werkkamer_1e)/4`) and `t_down_kt_c`
        is the floor boundary below `kt` (`0.5*(kamer_1e_midden + badkamer)`).
    """
    resolved: dict[str, float] = {}

    for room in _ALL_1E_ROOMS:
        raw_val = room_temps_c.get(room)
        if not _is_valid_number(raw_val):
            raw_val = room_temps_c.get(f"t_air_{room}_c")

        if _is_valid_number(raw_val):
            val = float(raw_val)
            resolved[room] = val
            if last_valid_cache is not None:
                last_valid_cache[room] = (val, ts_utc)
        elif last_valid_cache is not None and room in last_valid_cache:
            cached_temp, cached_ts = last_valid_cache[room]
            age_s = ts_utc - cached_ts
            if (
                0 <= age_s <= INTERFLOOR_HOLD_MAX_S
                and _is_valid_number(cached_temp)
            ):
                resolved[room] = float(cached_temp)

    # Group fallback for any room still missing after the 12-hour hold.
    for room in _ALL_1E_ROOMS:
        if room in resolved:
            continue
        own_group = (
            _SOUTH_1E_GROUP if room in _SOUTH_1E_GROUP else _NORTH_1E_GROUP
        )
        group_vals = [resolved[r] for r in own_group if r in resolved]
        if group_vals:
            resolved[room] = sum(group_vals) / len(group_vals)
        elif resolved:
            resolved[room] = sum(resolved.values()) / len(resolved)
        else:
            resolved[room] = 18.0

    t_up_wk_c = 0.25 * (
        2.0 * resolved["slaapkamer"]
        + resolved["kamer_1e_zuid"]
        + resolved["werkkamer_1e"]
    )
    t_down_kt_c = 0.5 * (resolved["kamer_1e_midden"] + resolved["badkamer"])
    return t_up_wk_c, t_down_kt_c


def compute_ac_heating_power_w(
    p_el_ac_w: float,
    ac_mode_code: float,
    ac_defrost: bool,
    t_out_c: float,
    eta_ac: float,
    base_q_ac_w: float | None = None,
) -> float:
    """Computes net split-AC thermal power delivered to the `kt` air node [W].

    Single source of truth for split-AC heating, defrost heat extraction, and
    cooling/dry thermal loads across real-time processing, history replay,
    trajectory calibration, and forward simulation.

    Args:
        p_el_ac_w: Measured split-AC electrical power [W].
        ac_mode_code: Numeric AC mode (`const.AC_MODE_*`).
        ac_defrost: True when split-AC outdoor unit defrost is active.
        t_out_c: Outdoor air temperature [°C].
        eta_ac: Net split-AC heating efficiency at desk height @ 7 °C [-].
        base_q_ac_w: Optional precomputed baseline AC load for non-heating
            modes [W]. When `None`, cooling (`-3.5 * P_net`) and dry
            (`-2.5 * P_net`) modes are computed directly.

    Returns:
        Net convective thermal power delivered to `kt` air node [W] (negative
        during defrost or cooling/dry modes).
    """
    p_ac = 0.0 if not _is_valid_number(p_el_ac_w) else max(0.0, float(p_el_ac_w))
    mode = (
        0
        if not _is_valid_number(ac_mode_code)
        else int(round(float(ac_mode_code)))
    )
    t_out = 7.0 if not _is_valid_number(t_out_c) else float(t_out_c)

    if ac_defrost:
        if base_q_ac_w is not None and base_q_ac_w < 0.0:
            return float(base_q_ac_w)
        return -min(200.0, 50.0 + 0.15 * max(0.0, p_ac - 3.0))
    if mode == int(const.AC_MODE_HEAT) and p_ac > 25.0:
        cop_scale = max(0.60, min(1.35, 1.0 + 0.025 * (t_out - 7.0)))
        return (p_ac - 3.0) * float(eta_ac) * cop_scale
    if base_q_ac_w is not None:
        return float(base_q_ac_w)
    if mode == int(const.AC_MODE_COOL) and p_ac > 25.0:
        return -3.5 * (p_ac - 3.0)
    if mode == int(const.AC_MODE_DRY) and p_ac > 25.0:
        return -2.5 * (p_ac - 3.0)
    return 0.0


def compute_internal_and_ac_loads_kt(
    p_desk_w: float | None,
    pir_occ: float,
    p_growlights_w: float,
    p_el_ac_w: float,
    ac_mode_code: float,
    ac_defrost: bool,
    t_out_c: float,
    eta_ac: float,
    desk_state: dict[str, Any] | None = None,
    dt_s: float = float(const.DT_STEP_S),
) -> tuple[float, float]:
    """Computes internal gains and split-AC thermal load for attic office `kt`.

    Mutates `desk_state` in place when provided to track Schmitt-trigger
    hysteresis and 10-min / 15-min occupancy hold timers.

    Args:
        p_desk_w: Optional smart-plug power on the office desk [W], or `None`
            until installed (activating automatic PIR fallback).
        pir_occ: Time-averaged split-AC occupancy radar fraction in `[0, 1]`.
        p_growlights_w: Measured grow-light electrical power [W].
        p_el_ac_w: Measured MHI split-AC electrical power [W].
        ac_mode_code: Numeric AC mode (`const.AC_MODE_*`).
        ac_defrost: True when split-AC outdoor unit is in defrost.
        t_out_c: Outdoor air temperature [°C].
        eta_ac: Calibrated net heating efficiency at desk height @ 7 °C [-].
        desk_state: Optional mutable dictionary storing hysteresis/hold timers.
        dt_s: Step duration in seconds.

    Returns:
        A tuple `(q_int_kt_w, q_ac_kt_w)` in Watts, representing known internal
        heat gains and net split-AC thermal power (negative during defrost or
        cooling/dry modes).
    """
    state = desk_state if desk_state is not None else {}
    step_s = max(0.0, float(dt_s))

    q_grow_w = (
        0.0
        if not _is_valid_number(p_growlights_w)
        else max(0.0, float(p_growlights_w))
    )

    if _is_valid_number(p_desk_w) and float(p_desk_w) >= 0.0:
        p_desk = float(p_desk_w)
        prev_active = bool(state.get("desk_active", False))
        if p_desk >= DESK_ON_THRESHOLD_W:
            active_now = True
        elif p_desk <= DESK_OFF_THRESHOLD_W:
            active_now = False
        else:
            active_now = prev_active

        if active_now:
            person_present = True
            new_hold_s = DESK_HOLD_S
        else:
            prev_hold_s = float(state.get("desk_hold_s", 0.0))
            if prev_hold_s > 0.0:
                person_present = True
                new_hold_s = max(0.0, prev_hold_s - step_s)
            else:
                person_present = False
                new_hold_s = 0.0

        state["desk_active"] = active_now
        state["desk_hold_s"] = new_hold_s
        q_workplace_w = p_desk + (
            DESK_METABOLIC_GAIN_W if person_present else 0.0
        )
    else:
        pir_val = 0.0 if not _is_valid_number(pir_occ) else float(pir_occ)
        if pir_val > PIR_OCC_THRESHOLD:
            pir_present = True
            new_pir_hold_s = PIR_HOLD_S
        else:
            prev_pir_hold_s = float(state.get("pir_hold_s", 0.0))
            if prev_pir_hold_s > 0.0:
                pir_present = True
                new_pir_hold_s = max(0.0, prev_pir_hold_s - step_s)
            else:
                pir_present = False
                new_pir_hold_s = 0.0

        state["pir_hold_s"] = new_pir_hold_s
        q_workplace_w = PIR_WORKPLACE_GAIN_W if pir_present else 0.0

    q_int_kt_w = q_grow_w + q_workplace_w
    q_ac_kt_w = compute_ac_heating_power_w(
        p_el_ac_w=p_el_ac_w,
        ac_mode_code=ac_mode_code,
        ac_defrost=ac_defrost,
        t_out_c=t_out_c,
        eta_ac=eta_ac,
        base_q_ac_w=None,
    )
    return q_int_kt_w, q_ac_kt_w
