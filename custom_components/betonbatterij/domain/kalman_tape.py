"""Precomputed 1D exogenous tape and unrolled scalar Kalman/ZOH fast paths.

Owns `MeasMode`, `PrecomputedZoneTape`, `build_zone_tape`, `get_zone_tape`,
`run_kalman_on_tape`, `evaluate_windows_on_tape`, `reconstruct_states_on_tape`,
`reconstruct_states_tuples`, and `rollout_open_loop_window_fast`.

This module is co-designed with `zoh_kalman.py` to execute the exact same 3x3
ZOH Kalman recursion and 7-input open-loop rollout using pre-packed 1D arrays
and unrolled scalar math so 90-day (25,920-row) Levenberg-Marquardt parameter
identification runs in seconds on Raspberry Pi 4 and Windows.

Does not own RingBuffer ETL (`preprocessing.py`), continuous matrix
discretization (`zoh_kalman.py`), or multi-regime hindcast scoring
(`evaluation.py`).

Units: temperatures in °C, thermal powers in W, conductances in W/K, time in s.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
import enum
import math

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import solar_weather
from custom_components.betonbatterij.domain import zoh_kalman


class MeasMode(enum.IntEnum):
    """Precomputed room-air measurement update mode per tape step."""

    NONE = 0
    DIRECT = 1
    TOBIT_COARSE = 2
    HEARTBEAT_FINE = 3
    COLD_START = 4


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
    gamma_cal: list[float] = dataclasses.field(default_factory=list)


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
    hist: preprocessing.PreprocessedHistory,
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
    gamma_cal = (
        list(hist.gamma_cal) if len(hist.gamma_cal) == n else [1.0] * n
    )
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

    mode_none = int(MeasMode.NONE)
    mode_direct = int(MeasMode.DIRECT)
    mode_tobit = int(MeasMode.TOBIT_COARSE)
    mode_heartbeat = int(MeasMode.HEARTBEAT_FINE)
    mode_cold = int(MeasMode.COLD_START)

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
    meas_mode: list[int] = [mode_none] * n
    y_obs: list[float] = [0.0] * n
    cold_cap_active: list[bool] = [False] * n

    default_sigma = 110.0 if is_wk else 75.0
    q_idle = zoh_kalman.Q_FLOOR_IDLE_K2_S
    q_pump_span = zoh_kalman.Q_FLOOR_PUMP_K2_S - q_idle
    harvest_mask = const.FLAG_DHW_TANK | const.FLAG_Z1_OFF_HARVEST
    ac_mode_heat_int = int(const.AC_MODE_HEAT)
    tz_name = priors.TIMEZONE_NAME
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
                q_ac_fixed[k] = solar_weather.compute_ac_heating_power_w(
                    p_el_ac_w=pac_k,
                    ac_mode_code=am_k,
                    ac_defrost=True,
                    t_out_c=t_out[k],
                    eta_ac=0.0,
                    base_q_ac_w=qbase_k,
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
            cached_hr_base, _ = preprocessing.local_hour_minute(
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
        meas_mode[0] = mode_cold
        y_obs[0] = t_meas[0]
        last_meas_temp_c: float | None = t_meas[0]
        time_since_change = 0.0
        air_obs_history: collections.deque[tuple[int, float]] = (
            collections.deque(maxlen=12)
        )
        cold_rem = (
            0.0 if u_pump[0] <= 0.5 else zoh_kalman.COLD_START_CAP_DURATION_S
        )
        max_gap_s = const.MAX_RESTART_BRIDGE_GAP_S

        for k in range(1, n):
            dt_k = dt_s[k]
            if dt_k > max_gap_s:
                det.observe(t_meas[k])
                r_v[k] = det.r_meas_k2
                meas_mode[k] = mode_cold
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
                    meas_mode[k] = mode_direct
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
                        meas_mode[k] = mode_tobit
                        r_v[k] = det.r_meas_k2
                    elif (k % 6 == 0) and time_since_change >= 1800.0:
                        meas_mode[k] = mode_heartbeat
                        r_v[k] = zoh_kalman.R_MEAS_HEARTBEAT_FINE_K2
                    else:
                        meas_mode[k] = mode_none
                        r_v[k] = det.r_meas_k2
                air_obs_history.append((ts_utc[k], y_m))
            else:
                time_since_change += step_s
                meas_mode[k] = mode_none
                r_v[k] = det.r_meas_k2

            if cold_rem > 0.0:
                cold_rem = max(0.0, cold_rem - step_s)

    ow_thr = zoh_kalman.OPEN_WINDOW_SLOPE_K_PER_H
    fast_300_anywind = [
        (
            dt_s[k] == 300.0
            and meas_mode[k] == mode_direct
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
        gamma_cal=gamma_cal,
    )


def get_zone_tape(
    hist: preprocessing.PreprocessedHistory,
    zone: contracts.ZoneId,
) -> PrecomputedZoneTape:
    """Returns cached `PrecomputedZoneTape` attached to `hist` for `zone`."""
    if zone == "wk":
        if hist.tape_wk is None:
            hist.tape_wk = build_zone_tape(hist, "wk")
        return hist.tape_wk
    if hist.tape_kt is None:
        hist.tape_kt = build_zone_tape(hist, "kt")
    return hist.tape_kt


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


def run_kalman_on_tape(
    tape: PrecomputedZoneTape,
    params: contracts.ZoneParameters,
    *,
    end_k: int,
    save_k0_set: set[int] | None = None,
    record_all: bool = False,
) -> tuple[
    list[tuple[float, float, float]] | None,
    dict[int, tuple[float, float]] | None,
    zoh_kalman.Mat3,
    zoh_kalman.Mat3x7,
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

    mode_direct = int(MeasMode.DIRECT)
    mode_tobit = int(MeasMode.TOBIT_COARSE)
    mode_heartbeat = int(MeasMode.HEARTBEAT_FINE)
    mode_cold = int(MeasMode.COLD_START)

    # Unpack standard 300s zero-wind ZOH + Q_d basis once outside the k-loop.
    ad_std, bd_std, _, _, _ = zoh_kalman.discretize_ab_cached(
        params, 300.0, 0.0
    )
    g0_std, g1_std, g2_std = zoh_kalman.discretize_qd_basis_cached(
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
    gamma_cal_arr = (
        tape.gamma_cal
        if len(tape.gamma_cal) == tape.n_rows
        else [1.0] * tape.n_rows
    )
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

    surf_gamma = (
        zoh_kalman.surface_temp_divider_gamma(tape.zone, params.r_fh_k_kw)
        if has_surf
        else 0.60
    )

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
                    a00 * x0
                    + a01 * x1
                    + a02 * x2
                    + b00 * t_out_arr[k]
                    + b01 * t_up_arr[k]
                    + b02 * t_down_arr[k]
                    + b03 * i_sol_arr[k]
                    + b04 * u4
                    + b05 * q_int_arr[k]
                    + b06_eta * ac_e
                    + b06 * ac_f
                )
                xm1 = (
                    a10 * x0
                    + a11 * x1
                    + a12 * x2
                    + b10 * t_out_arr[k]
                    + b11 * t_up_arr[k]
                    + b12 * t_down_arr[k]
                    + b13 * i_sol_arr[k]
                    + b14 * u4
                    + b15 * q_int_arr[k]
                    + b16_eta * ac_e
                    + b16 * ac_f
                )
            else:
                xm0 = (
                    a00 * x0
                    + a01 * x1
                    + a02 * x2
                    + b00 * t_out_arr[k]
                    + b01 * t_up_arr[k]
                    + b02 * t_down_arr[k]
                    + b03 * i_sol_arr[k]
                    + b04 * u4
                    + b05 * q_int_arr[k]
                )
                xm1 = (
                    a10 * x0
                    + a11 * x1
                    + a12 * x2
                    + b10 * t_out_arr[k]
                    + b11 * t_up_arr[k]
                    + b12 * t_down_arr[k]
                    + b13 * i_sol_arr[k]
                    + b14 * u4
                    + b15 * q_int_arr[k]
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
            pm00 = (
                ap00 * a00
                + ap01 * a01
                + ap02 * a02
                + (qg0_00 + qc11 * g1_00 + qc22 * g2_00)
            )
            pm01 = (
                ap00 * a10
                + ap01 * a11
                + ap02 * a12
                + (qg0_01 + qc11 * g1_01 + qc22 * g2_01)
            )
            pm02 = ap02 * a22 + (qg0_02 + qc11 * g1_02 + qc22 * g2_02)
            pm11 = (
                ap10 * a10
                + ap11 * a11
                + ap12 * a12
                + (qg0_11 + qc11 * g1_11 + qc22 * g2_11)
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
            q_fh_k = (
                (gamma_cal_arr[k] * up_uz) * khx * (t_buf_top_arr[k] - x1)
            )
            if q_fh_k < 0.0:
                q_fh_k = 0.0
        else:
            q_fh_k = q_fh_hyd_arr[k] * khx_scale_act

        q_ac_k = (
            (ac_eff_arr[k] * eta_ac + q_ac_fixed_arr[k]) if is_kt else 0.0
        )

        mm = meas_mode_arr[k]
        if mm == mode_cold:
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
                ad_k, bd_k, _, _, _ = zoh_kalman.discretize_ab_cached(
                    vent_params, step_s, v_eff_k
                )
                g0_k, g1_k, g2_k = zoh_kalman.discretize_qd_basis_cached(
                    vent_params, step_s, v_eff_k
                )
                ad_nom, bd_nom, _, _, _ = zoh_kalman.discretize_ab_cached(
                    params, step_s, v_eff_k
                )
                af10, af11, af12 = ad_nom[1][0], ad_nom[1][1], ad_nom[1][2]
                bf1 = bd_nom[1]
                qc22 = sd_sq2_arr[k] * inv_tau_d_vent
            else:
                ad_k, bd_k, _, _, _ = zoh_kalman.discretize_ab_cached(
                    params, step_s, v_eff_k
                )
                g0_k, g1_k, g2_k = zoh_kalman.discretize_qd_basis_cached(
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
        pm02 = (
            0.5 * (ap02 * ca22 + (ap20 * ca00 + ap21 * ca01 + ap22 * ca02))
            + q02
        )
        pm11 = ap10 * ca10 + ap11 * ca11 + ap12 * ca12 + q11
        pm12 = (
            0.5 * (ap12 * ca22 + (ap20 * ca10 + ap21 * ca11 + ap22 * ca12))
            + q12
        )
        pm22 = ap22 * ca22 + q22

        do_air_update = False
        e_innov = 0.0
        rv_eff = 0.0

        if mm == mode_direct:
            rv_eff = r_v_arr[k]
            e_innov = y_obs_arr[k] - xm0
            if obs_slope_arr[k] < ow_slope_thr:
                s_k = pm00 + rv_eff
                gate_thresh = max(0.20, 2.5 * math.sqrt(max(1e-12, s_k)))
                if e_innov < -gate_thresh:
                    open_window_hold_rem_s = ow_hold_const
                    open_window_active = True
            do_air_update = True
        elif mm == mode_tobit:
            tobit = zoh_kalman.tobit_equivalent_innovation(
                y_center=y_obs_arr[k],
                half_width_k=0.10,
                mu_minus=xm0,
                p00_minus=pm00,
                sigma_eps_k=0.03,
            )
            if tobit is not None:
                e_innov, rv_eff = tobit
                do_air_update = True
        elif mm == mode_heartbeat:
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
            p12 = (
                0.5 * (-wp10 * gk2 + wp12 - wp20 * gk1 + wp21)
                + rv_eff * gk1 * gk2
            )
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
                x_u, p_u = zoh_kalman.joseph_scalar_update(
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
    _, saved_k0, ad_std, bd_std, khx_ratio, khx_scale_act = run_kalman_on_tape(
        tape,
        params,
        end_k=max_k0,
        save_k0_set=needed_k0,
        record_all=False,
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
    gamma_cal_arr = (
        tape.gamma_cal
        if len(tape.gamma_cal) == tape.n_rows
        else [1.0] * tape.n_rows
    )
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
                q_fh_s = (
                    (gamma_cal_arr[idx] * up_uz)
                    * khx
                    * (t_buf_top_arr[idx] - ox1)
                )
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
                ad_o, bd_o, _, _, _ = zoh_kalman.discretize_ab_cached(
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


def reconstruct_states_on_tape(
    tape: PrecomputedZoneTape,
    params: contracts.ZoneParameters,
    end_k: int | None = None,
) -> list[tuple[float, float, float]]:
    """Runs unrolled scalar Kalman filter on `tape` up to `end_k` inclusive."""
    n = tape.n_rows
    if n == 0:
        return []
    limit_k = (n - 1) if end_k is None else min(end_k, n - 1)
    states_out, _, _, _, _, _ = run_kalman_on_tape(
        tape,
        params,
        end_k=limit_k,
        save_k0_set=None,
        record_all=True,
    )
    assert states_out is not None
    return states_out


def reconstruct_states_tuples(
    hist: preprocessing.PreprocessedHistory,
    zone: contracts.ZoneId,
    params: contracts.ZoneParameters,
) -> list[tuple[float, float, float]]:
    """Fast scalar Kalman state reconstruction returning `(t_air, t_floor, d)`."""
    return reconstruct_states_on_tape(get_zone_tape(hist, zone), params)


def rollout_open_loop_window_fast(
    tape: PrecomputedZoneTape,
    k0: int,
    length_steps: int,
    state_init: tuple[float, float, float],
    params: contracts.ZoneParameters,
    ad_std: zoh_kalman.Mat3,
    bd_std: zoh_kalman.Mat3x7,
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
    gamma_cal_arr = (
        tape.gamma_cal
        if len(tape.gamma_cal) == tape.n_rows
        else [1.0] * tape.n_rows
    )
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
            q_fh_s = (
                (gamma_cal_arr[idx] * up_uz)
                * khx
                * (t_buf_top_arr[idx] - ox1)
            )
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
            ad_o, bd_o, _, _, _ = zoh_kalman.discretize_ab_cached(
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
