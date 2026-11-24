"""Unit tests for Component 3 (`calibration.py`) and `DigitalTwinEngine`.

Covers:
1. Pure-Python Box-Projected Huber Levenberg-Marquardt
   (`solve_b2_box_huber_lm`): multi-scale convergence, IRLS Huber outlier
   damping, active KKT box projection, Fisher `1-sigma` standard deviations,
   and `|rho_jk| > 0.90` collinearity detection.
2. Synthetic multi-day parameter recovery (`A2` + `B2` + `calibrate_k_hx_robust`)
   for both `wk` (4D + `K_hx`) and `kt` (5D incl. `eta_ac` + `K_hx`), preserving
   `aanwezig == off` holiday windows and avoiding the Self-Masking Trap, plus
   `80/20` `auto_weekly` blending.
3. Autonomous seasonal gate (`mean_t_out_7d < 14.0 °C` & 48h cooling guard) and
   step-level cooling + `2.0h` (`24`-step) post-cooling tail exclusion.
4. Choice D2 (`update_posterior_memory` 52w aging + Fisher information gate)
   and Choice D3 (`SeasonalEpisodeBank` 365d reservoir rescuing dark-winter
   `a_sol_m2`).
5. Out-of-Sample Holdout Gate rejection when a distorted candidate degrades
   unseen holdout `J_eval` or any regime by `> +0.03 °C`.
6. `DigitalTwinEngine` 8-deep `snapshot_stack`, `rollback_parameters`,
   `reset_parameters`, and automatic 48-hour (`576`-step) Kalman + hindcast
   replay.
"""
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import datetime
import json
import math
import time
from typing import Any

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import calibration
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import engine
from custom_components.betonbatterij.domain import evaluation
from custom_components.betonbatterij.domain import hydraulics
from custom_components.betonbatterij.domain import kalman_tape
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import solar_weather
from custom_components.betonbatterij.domain import zoh_kalman


def _generate_calibration_dataset(
    n_days: int,
    params_wk: contracts.ZoneParameters,
    params_kt: contracts.ZoneParameters,
    *,
    solar_enabled: bool = True,
    base_t_out_c: float = 2.5,
    start_ts_utc: int | None = None,
) -> dict[str, list[float | int]]:
    """Generates multi-regime synthetic winter telemetry with ground truth."""
    if start_ts_utc is None:
        start_ts_utc = int(
            datetime.datetime(
                2026, 1, 10, 0, 0, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )

    n_steps = n_days * const.STEPS_PER_DAY
    cols: dict[str, list[float | int]] = {
        c: [0] * n_steps for c in const.RINGBUFFER_INT_COLS
    }
    for c in const.RINGBUFFER_FLOAT_COLS:
        cols[c] = [0.0] * n_steps

    flows = const.DEFAULT_FLOW_RATES_L_MIN
    k_hx_map = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
    k_hx_map["wk"] = params_wk.k_hx_w_k
    k_hx_map["gang"] = params_wk.k_hx_w_k * (flows["gang"] / flows["wk"])
    k_hx_map["kantoor"] = params_kt.k_hx_w_k

    hyd = hydraulics.HydraulicModel(
        flow_rates_l_min=flows, k_hx_by_group_w_k=k_hx_map
    )
    st_wk = contracts.ThermalState(t_air_c=19.8, t_floor_c=21.0, d_est_w=0.0)
    st_kt = contracts.ThermalState(t_air_c=18.8, t_floor_c=19.3, d_est_w=0.0)
    hm_ema: float | None = None
    interfloor_cache: dict[str, tuple[float, int]] = {}
    desk_state: dict[str, Any] = {}

    for k in range(n_steps):
        ts_k = start_ts_utc + k * const.DT_STEP_S
        day_idx = k // const.STEPS_PER_DAY
        hr_loc, mn_loc = preprocessing.local_hour_minute(ts_k)
        mins_loc = hr_loc * 60 + mn_loc

        fl = const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
        if day_idx in (3, 6):
            fl &= ~const.FLAG_AANWEZIG  # Clean holiday / absence days

        t_out_raw = base_t_out_c - 2.3 * math.cos(
            2.0 * math.pi * (mins_loc - 210.0) / 1440.0
        )
        alpha_s, _ = solar_weather.solar_position(ts_k)

        is_sunny_day = solar_enabled and (day_idx in (2, 4, 8))
        if alpha_s > 2.0 and solar_enabled:
            peak_pv = 4100.0 if is_sunny_day else 140.0
            p_pv = min(
                peak_pv,
                peak_pv * max(0.0, math.sin(math.radians(alpha_s)) * 2.25),
            )
        else:
            p_pv = 0.0

        u_pump = 0.0
        u_bg = 0.0
        u_kt = 0.0
        f_hz = 0.0
        v_prim = 0.0
        t_top = 23.5
        t_bot_cal = 22.5
        p_ac = 0.0
        ac_mode = float(const.AC_MODE_OFF)

        # 00:30..06:30 is passive every day for clean `night` identification
        if not is_sunny_day:
            if day_idx in (1, 5, 7):
                if 420 <= mins_loc < 690:
                    u_pump = 1.0
                    u_bg = 1.0
                    f_hz = 21.0
                    v_prim = 14.5
                    t_top = 33.5
                    t_bot_cal = 29.5
                elif day_idx in (1, 5) and 810 <= mins_loc < 855:
                    ac_mode = float(const.AC_MODE_HEAT)
                    p_ac = 460.0
            else:
                if 450 <= mins_loc < 495:
                    ac_mode = float(const.AC_MODE_HEAT)
                    p_ac = 460.0
                elif 690 <= mins_loc < 960:
                    u_pump = 1.0
                    u_bg = 1.0
                    u_kt = 1.0
                    f_hz = 22.0
                    v_prim = 15.5
                    t_top = 34.5
                    t_bot_cal = 30.5

        if u_pump > 0.5:
            fl |= const.FLAG_PUMP_SEC_ON

        u_valves: dict[str, float] = {
            "wk": u_bg,
            "gang": u_bg,
            "slaapkamer": 0.0,
            "badkamer": 0.0,
            "werkkamer_1e": 0.0,
            "kamer_1e_midden": 0.0,
            "kamer_1e_zuid": 0.0,
            "kantoor": u_kt,
            "badkamer_boven": 0.0,
            "logeerkamer": 0.0,
        }
        room_temps: dict[str, float | None] = {
            "wk": st_wk.t_air_c,
            "gang": st_wk.t_air_c - 0.8,
            "slaapkamer": 18.0,
            "badkamer": 18.5,
            "werkkamer_1e": 18.0,
            "kamer_1e_midden": 18.0,
            "kamer_1e_zuid": 18.0,
            "kantoor": st_kt.t_air_c,
            "badkamer_boven": 18.0,
            "logeerkamer": 17.5,
        }

        q_exp_wk = (
            u_pump * u_bg * k_hx_map["wk"] * max(0.0, t_top - st_wk.t_floor_c)
        )
        q_exp_gang = (
            u_pump
            * u_bg
            * k_hx_map["gang"]
            * max(0.0, t_top - (st_wk.t_floor_c - 0.5))
        )
        q_exp_kt = (
            u_pump
            * u_kt
            * k_hx_map["kantoor"]
            * max(0.0, t_top - st_kt.t_floor_c)
        )
        p_prim = (q_exp_wk + q_exp_gang + q_exp_kt) if f_hz > 0.0 else 0.0
        t_bot_raw = t_bot_cal - priors.BUF_BOT_SENSOR_OFFSET_C

        hyd_res = hyd.step(
            dt_s=float(const.DT_STEP_S),
            u_pump_sec=u_pump,
            u_valves=u_valves,
            threeway_room=True,
            wp_defrost=False,
            v_prim_l_min=v_prim,
            p_prim_th_w=p_prim,
            f_comp_hz=f_hz,
            t_buf_top_start_c=t_top,
            t_buf_top_end_c=t_top,
            t_buf_bot_start_c=t_bot_raw,
            t_buf_bot_end_c=t_bot_raw,
            t_floor_wk_c=st_wk.t_floor_c,
            t_floor_kt_c=st_kt.t_floor_c,
            room_temps_c=room_temps,
        )
        q_wk = hyd_res.q_fh_wk_w
        q_kt = hyd_res.q_fh_kt_w

        hm_ema = (
            t_out_raw
            if hm_ema is None
            else (t_out_raw / 6.0) + (5.0 / 6.0) * hm_ema
        )
        t_out_k = solar_weather.select_outdoor_temp(
            ts_utc=ts_k,
            t_dak_c=t_out_raw,
            t_weather_c=t_out_raw,
            t_hm_ema_c=hm_ema,
        )
        i_wk, i_kt, _, _ = solar_weather.project_solar_irradiance(
            ts_utc=ts_k,
            p_pv_w=p_pv,
            p_pv_forecast_w=p_pv,
            epex_eur_kwh=0.18,
            screen_small_pos=100.0,
            screen_large_pos=100.0,
            pv_stale=False,
        )
        t_up_wk, t_dn_kt = solar_weather.compute_interfloor_boundaries(
            room_temps_c=room_temps,
            last_valid_cache=interfloor_cache,
            ts_utc=ts_k,
        )
        t_dn_wk = priors.T_GROUND_MEAN_C
        q_int_wk = priors.BASELOAD_WK_W + max(
            0.0, hyd_res.q_leak_1e_to_wk_w
        )
        q_int_kt, q_ac_base = solar_weather.compute_internal_and_ac_loads_kt(
            p_desk_w=None,
            pir_occ=0.0,
            p_growlights_w=0.0,
            p_el_ac_w=p_ac,
            ac_mode_code=ac_mode,
            ac_defrost=False,
            t_out_c=t_out_k,
            eta_ac=params_kt.eta_ac,
            desk_state=desk_state,
            dt_s=float(const.DT_STEP_S),
        )
        q_ac_kt = solar_weather.compute_ac_heating_power_w(
            p_el_ac_w=p_ac,
            ac_mode_code=ac_mode,
            ac_defrost=False,
            t_out_c=t_out_k,
            eta_ac=params_kt.eta_ac,
            base_q_ac_w=q_ac_base,
        )

        exog_wk = contracts.ExogenousStepInput(
            t_out_c=t_out_k,
            v_wind_ms=2.5,
            i_sol_wm2=i_wk,
            t_down_c=t_dn_wk,
            q_int_known_w=q_int_wk,
            t_up_c=t_up_wk,
        )
        exog_kt = contracts.ExogenousStepInput(
            t_out_c=t_out_k,
            v_wind_ms=2.5,
            i_sol_wm2=i_kt,
            t_down_c=t_dn_kt,
            q_int_known_w=q_int_kt,
            t_up_c=0.0,
        )

        if k > 0:
            st_wk = zoh_kalman.propagate_open_loop_step(
                state=st_wk,
                params=params_wk,
                exog=exog_wk,
                q_fh_w=q_wk,
                q_ac_w=0.0,
                dt_s=float(const.DT_STEP_S),
            )
            st_kt = zoh_kalman.propagate_open_loop_step(
                state=st_kt,
                params=params_kt,
                exog=exog_kt,
                q_fh_w=q_kt,
                q_ac_w=q_ac_kt,
                dt_s=float(const.DT_STEP_S),
            )

        cols["ts_utc"][k] = ts_k
        cols["flags"][k] = fl
        cols["u_eff_bg"][k] = u_bg
        cols["u_eff_kantoor"][k] = u_kt
        cols["u_pump_sec"][k] = u_pump
        cols["t_air_wk_c"][k] = st_wk.t_air_c
        cols["t_air_slaapkamer_c"][k] = 18.0
        cols["t_air_badkamer_c"][k] = 18.5
        cols["t_air_werkkamer_1e_c"][k] = 18.0
        cols["t_air_kamer_1e_midden_c"][k] = 18.0
        cols["t_air_kamer_1e_zuid_c"][k] = 18.0
        cols["t_air_kantoor_c"][k] = st_kt.t_air_c
        cols["t_air_badkamer_boven_c"][k] = 18.0
        cols["t_air_logeerkamer_c"][k] = 17.5
        cols["v_prim_l_min"][k] = v_prim
        cols["t_wp_out_c"][k] = t_top + 2.5 if f_hz > 0.0 else t_top
        cols["t_wp_in_c"][k] = t_top - 0.5 if f_hz > 0.0 else t_top
        cols["p_prim_th_w"][k] = p_prim
        cols["f_comp_hz"][k] = f_hz
        cols["p_wp_raw_el_w"][k] = 600.0 if f_hz > 0.0 else 0.0
        cols["t_buf_top_start_c"][k] = t_top
        cols["t_buf_top_end_c"][k] = t_top
        cols["t_buf_bot_start_c"][k] = t_bot_raw
        cols["t_buf_bot_end_c"][k] = t_bot_raw
        cols["t_dhw_top_c"][k] = 48.0
        cols["t_dhw_bot_c"][k] = 42.0
        cols["p_p1_net_w"][k] = (745.0 if f_hz > 0.0 else 250.0) - p_pv
        cols["t_out_dak_c"][k] = t_out_raw
        cols["t_out_weather_c"][k] = t_out_raw
        cols["t_out_hm_c"][k] = t_out_raw
        cols["rh_out_pct"][k] = 80.0
        cols["v_wind_ms"][k] = 2.5
        cols["p_pv_w"][k] = p_pv
        cols["p_pv_forecast_w"][k] = p_pv
        cols["epex_eur_kwh"][k] = 0.18
        cols["screen_small_pos"][k] = 100.0
        cols["screen_large_pos"][k] = 100.0
        cols["p_el_ac_kt_w"][k] = p_ac
        cols["ac_mode_code"][k] = ac_mode
        cols["p_desk_kt_w"][k] = math.nan
        cols["t_bg_sup_c"][k] = math.nan
        cols["t_bg_ret_c"][k] = math.nan
        cols["t_floor_surf_meas_wk_c"][k] = math.nan

    return cols


def _exponential_curve_residuals(
    theta: Sequence[float],
    x_grid: Sequence[float],
    y_noisy: Sequence[float],
) -> list[float]:
    """Computes standardized nonlinear curve residuals for B2 solver test."""
    return [
        ((theta[0] / 1000.0) * (1.0 - math.exp(-x / theta[1])) - ym) / 0.05
        for x, ym in zip(x_grid, y_noisy)
    ]


def test_solve_b2_box_huber_lm_convergence_bounds_outliers_and_collinearity() -> None:
    """Verifies Choice B2 Huber robustness, KKT projection, & collinearity."""
    true_theta = [4800.0, 2.15]
    x_grid = [i * 0.25 for i in range(1, 35)]
    y_clean = [
        (true_theta[0] / 1000.0) * (1.0 - math.exp(-x / true_theta[1]))
        for x in x_grid
    ]
    y_noisy = list(y_clean)
    y_noisy[10] += 25.0  # Gross +30 sigma outlier damped by Huber IRLS

    sol = calibration.solve_b2_box_huber_lm(
        residual_fn=lambda th: _exponential_curve_residuals(
            th, x_grid, y_noisy
        ),
        theta0=[5600.0, 2.80],
        lower=[3500.0, 1.50],
        upper=[8000.0, 4.00],
        prior_mean=[5500.0, 2.60],
        prior_prec_diag=[1.0 / (1100.0**2), 1.0 / (0.60**2)],
        huber_delta=2.5,
    )
    assert sol.success is True
    assert sol.n_iter <= 14
    assert abs(sol.theta[0] - true_theta[0]) / true_theta[0] < 0.02
    assert abs(sol.theta[1] - true_theta[1]) / true_theta[1] < 0.02
    assert all(s > 0.0 for s in sol.posterior_std)

    # Active KKT box bound projection when unconstrained minimum < lower[0]
    sol_box = calibration.solve_b2_box_huber_lm(
        residual_fn=lambda th: [
            (th[0] - 2500.0) / 10.0,
            (th[1] - 2.30) / 0.02,
        ],
        theta0=[5000.0, 3.00],
        lower=[3500.0, 1.80],
        upper=[8000.0, 3.60],
        prior_mean=[5500.0, 2.60],
        prior_prec_diag=[1e-8, 1e-4],
    )
    assert math.isclose(sol_box.theta[0], 3500.0, abs_tol=1e-4)
    assert math.isclose(sol_box.theta[1], 2.30, abs_tol=1e-2)

    # Collinearity detection (`|rho_01| > 0.90`)
    sol_col = calibration.solve_b2_box_huber_lm(
        residual_fn=lambda th: [
            ((th[0] + th[1]) - 10.0) * 50.0 for _ in range(20)
        ],
        theta0=[4.5, 4.5],
        lower=[1.0, 1.0],
        upper=[9.0, 9.0],
        prior_mean=[5.0, 5.0],
        prior_prec_diag=[0.1, 0.1],
    )
    assert len(sol_col.collinearity_pairs) == 1
    assert abs(sol_col.collinearity_pairs[0][2]) > 0.90


def test_synthetic_multiday_parameter_recovery_wk_and_kt_a2_b2() -> None:
    """Verifies Choice A2 + B2 + `calibrate_k_hx_robust` recover `wk` & `kt`."""
    true_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK,
        c_floor_wh_k=4750.0,
        r_fh_k_kw=2.18,
        h_loss_w_k=51.0,
        a_sol_m2=4.15,
        k_hx_w_k=218.0,
    )
    true_kt = dataclasses.replace(
        priors.DAY1_PRIORS_KT,
        c_floor_wh_k=1950.0,
        r_fh_k_kw=4.25,
        h_loss_w_k=36.0,
        a_sol_m2=0.68,
        eta_ac=4.30,
        k_hx_w_k=48.0,
    )

    cols = _generate_calibration_dataset(
        n_days=10, params_wk=true_wk, params_kt=true_kt
    )

    khx_fit = calibration.calibrate_k_hx_robust(
        cols,
        const.DEFAULT_FLOW_RATES_L_MIN,
        priors.DAY1_K_HX_BY_GROUP_W_K,
    )
    assert abs(khx_fit["wk"] - 218.0) / 218.0 < 0.05
    assert abs(khx_fit["kantoor"] - 48.0) / 48.0 < 0.06

    out_wk, mem_wk = calibration.calibrate_zone_a2(
        cols,
        "wk",
        priors.DAY1_PRIORS_WK,
        mode="manual",
        flow_rates_l_min=const.DEFAULT_FLOW_RATES_L_MIN,
        k_hx_by_group=khx_fit,
    )
    assert out_wk.accepted is True
    assert out_wk.holdout_after.overall_6h.j_eval_c is not None
    assert out_wk.holdout_before.overall_6h.j_eval_c is not None
    assert (
        out_wk.holdout_after.overall_6h.j_eval_c
        < out_wk.holdout_before.overall_6h.j_eval_c
    )
    assert abs(out_wk.params_applied.h_loss_w_k - true_wk.h_loss_w_k) < abs(
        priors.DAY1_PRIORS_WK.h_loss_w_k - true_wk.h_loss_w_k
    )
    assert abs(out_wk.params_applied.a_sol_m2 - true_wk.a_sol_m2) < 0.40
    assert "c_floor_wh_k" in mem_wk.mean

    out_kt, _ = calibration.calibrate_zone_a2(
        cols,
        "kt",
        priors.DAY1_PRIORS_KT,
        mode="manual",
        flow_rates_l_min=const.DEFAULT_FLOW_RATES_L_MIN,
        k_hx_by_group=khx_fit,
    )
    assert out_kt.accepted is True
    assert out_kt.holdout_after.overall_6h.j_eval_c is not None
    assert out_kt.holdout_before.overall_6h.j_eval_c is not None
    assert (
        out_kt.holdout_after.overall_6h.j_eval_c
        < out_kt.holdout_before.overall_6h.j_eval_c
    )
    assert abs(out_kt.params_applied.eta_ac - true_kt.eta_ac) < 0.35
    assert abs(out_kt.params_applied.h_loss_w_k - true_kt.h_loss_w_k) < abs(
        priors.DAY1_PRIORS_KT.h_loss_w_k - true_kt.h_loss_w_k
    )

    out_auto, _ = calibration.calibrate_zone_a2(
        cols,
        "wk",
        priors.DAY1_PRIORS_WK,
        mode="auto_weekly",
        flow_rates_l_min=const.DEFAULT_FLOW_RATES_L_MIN,
        k_hx_by_group=khx_fit,
    )
    assert out_auto.accepted is True
    expected_h_loss = (
        0.80 * priors.DAY1_PRIORS_WK.h_loss_w_k
        + 0.20 * out_auto.params_map_raw.h_loss_w_k
    )
    assert math.isclose(
        out_auto.params_candidate.h_loss_w_k, expected_h_loss, abs_tol=1e-9
    )


def test_cooling_exclusion_with_2h_tail_and_seasonal_gate() -> None:
    """Verifies autonomous seasonal gate (`<14 °C`) & +2.0h cooling tail."""
    cols = _generate_calibration_dataset(
        n_days=8,
        params_wk=priors.DAY1_PRIORS_WK,
        params_kt=priors.DAY1_PRIORS_KT,
    )
    ok_winter, _ = calibration.check_seasonal_calibration_gate(cols)
    assert ok_winter is True

    warm_cols = _generate_calibration_dataset(
        n_days=8,
        params_wk=priors.DAY1_PRIORS_WK,
        params_kt=priors.DAY1_PRIORS_KT,
        base_t_out_c=16.5,
    )
    ok_warm, msg_warm = calibration.check_seasonal_calibration_gate(warm_cols)
    assert ok_warm is False
    assert "14.0" in msg_warm

    out_blocked, _ = calibration.calibrate_zone_a2(
        warm_cols, "wk", priors.DAY1_PRIORS_WK, mode="auto_weekly"
    )
    assert out_blocked.accepted is False
    assert "Seasonal gate" in out_blocked.reason

    cool_recent = {k: list(v) for k, v in cols.items()}
    cool_recent["flags"][-100] = (
        int(cool_recent["flags"][-100]) | const.FLAG_FLOOR_COOL
    )
    ok_cool, _ = calibration.check_seasonal_calibration_gate(cool_recent)
    assert ok_cool is False

    ac_cool_recent = {k: list(v) for k, v in cols.items()}
    ac_cool_recent["ac_mode_code"][-50] = float(const.AC_MODE_COOL)
    ok_ac_cool, _ = calibration.check_seasonal_calibration_gate(ac_cool_recent)
    assert ok_ac_cool is False

    cols_mid_cool = {k: list(v) for k, v in cols.items()}
    c_start = 2 * const.STEPS_PER_DAY + 3
    c_end = 2 * const.STEPS_PER_DAY + 12
    for idx in range(c_start, c_end + 1):
        cols_mid_cool["flags"][idx] = (
            int(cols_mid_cool["flags"][idx]) | const.FLAG_FLOOR_COOL
        )

    hist = preprocessing.preprocess_history_columns(cols_mid_cool)
    for idx in range(c_start, c_end + 24 + 1):
        assert hist.cooling_or_tail[idx] is True
    assert hist.cooling_or_tail[c_end + 25] is False

    states, _, _, _ = evaluation.reconstruct_kalman_states(
        hist, "wk", priors.DAY1_PRIORS_WK
    )
    wins = calibration.extract_clean_regime_windows(
        hist, "wk", [s.d_est_w for s in states], min_k0=144
    )
    for w in wins:
        w_steps = set(range(w.k0, w.k0 + w.length_steps + 1))
        forbidden = set(range(c_start, c_end + 24 + 1))
        assert w_steps.isdisjoint(forbidden)


def test_seasonal_precision_memory_d2_and_episode_bank_d3() -> None:
    """Verifies Choice D2 Fisher gate & Choice D3 EpisodeBank across seasons."""
    bounds = {
        k: v for k, v in priors.PARAM_BOUNDS_WK.items() if k != "h_up_w_k"
    }

    _, _, mem_autumn = calibration.update_posterior_memory(
        old_memory=None,
        bounds=bounds,
        fitted_mean={
            "c_floor_wh_k": 4900.0,
            "r_fh_k_kw": 2.25,
            "h_loss_w_k": 54.0,
            "a_sol_m2": 4.40,
        },
        fisher_diag={
            "c_floor_wh_k": 5e-5,
            "r_fh_k_kw": 40.0,
            "h_loss_w_k": 0.25,
            "a_sol_m2": 22.0,
        },
    )
    assert math.isclose(mem_autumn.mean["a_sol_m2"], 4.40, abs_tol=1e-9)
    init_prec_sol = mem_autumn.precision["a_sol_m2"]

    mem_curr = mem_autumn
    gated_mean: dict[str, float] = {}
    for _ in range(13):
        gated_mean, _, mem_curr = calibration.update_posterior_memory(
            old_memory=mem_curr,
            bounds=bounds,
            fitted_mean={
                "c_floor_wh_k": 4850.0,
                "r_fh_k_kw": 2.22,
                "h_loss_w_k": 53.0,
                "a_sol_m2": 3.20,
            },
            fisher_diag={
                "c_floor_wh_k": 8e-5,
                "r_fh_k_kw": 55.0,
                "h_loss_w_k": 0.40,
                "a_sol_m2": 0.0,
            },
        )

    assert math.isclose(gated_mean["a_sol_m2"], 4.40, abs_tol=1e-9)
    assert math.isclose(mem_curr.mean["a_sol_m2"], 4.40, abs_tol=1e-9)
    assert mem_curr.precision["a_sol_m2"] < init_prec_sol
    assert mem_curr.precision["a_sol_m2"] > init_prec_sol * (
        calibration.MEMORY_WEEKLY_GAMMA**15
    )

    true_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK, h_loss_w_k=52.0, a_sol_m2=4.50
    )
    autumn_ts = int(
        datetime.datetime(
            2025, 10, 20, 0, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp()
    )
    autumn_cols = _generate_calibration_dataset(
        n_days=6,
        params_wk=true_wk,
        params_kt=priors.DAY1_PRIORS_KT,
        solar_enabled=True,
        start_ts_utc=autumn_ts,
    )

    bank = calibration.SeasonalEpisodeBank(max_per_regime=8)
    n_added = bank.update_from_columns(autumn_cols)
    assert n_added > 0
    assert len(bank.get_episodes("wk", "solar")) >= 1

    bank_json = json.dumps(bank.to_dict(), allow_nan=False)
    bank_restored = calibration.SeasonalEpisodeBank.from_dict(
        json.loads(bank_json)
    )
    assert len(bank_restored.get_episodes("wk", "solar")) == len(
        bank.get_episodes("wk", "solar")
    )

    winter_ts = int(
        datetime.datetime(
            2026, 1, 15, 0, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp()
    )
    dark_winter_cols = _generate_calibration_dataset(
        n_days=8,
        params_wk=true_wk,
        params_kt=priors.DAY1_PRIORS_KT,
        solar_enabled=False,
        start_ts_utc=winter_ts,
    )

    out_no_bank, _ = calibration.calibrate_zone_a2(
        dark_winter_cols,
        "wk",
        priors.DAY1_PRIORS_WK,
        mode="manual",
        episode_bank=None,
    )
    assert math.isclose(
        out_no_bank.params_map_raw.a_sol_m2,
        priors.DAY1_PRIORS_WK.a_sol_m2,
        abs_tol=0.05,
    )

    out_with_bank, _ = calibration.calibrate_zone_a2(
        dark_winter_cols,
        "wk",
        priors.DAY1_PRIORS_WK,
        mode="manual",
        episode_bank=bank_restored,
    )
    assert abs(out_with_bank.params_map_raw.a_sol_m2 - 4.50) < 0.35


def test_holdout_rejection_gate_when_candidate_degrades() -> None:
    """Verifies holdout gate rejects candidates when unseen `J_eval` degrades."""
    true_wk = priors.DAY1_PRIORS_WK
    cols = _generate_calibration_dataset(
        n_days=9, params_wk=true_wk, params_kt=priors.DAY1_PRIORS_KT
    )

    train_end = preprocessing.compute_train_holdout_split(cols)
    corrupted = {k: list(v) for k, v in cols.items()}
    for k in range(train_end):
        hr, mn = preprocessing.local_hour_minute(int(corrupted["ts_utc"][k]))
        mins = hr * 60 + mn
        if 30 <= mins <= 390:
            corrupted["t_air_wk_c"][k] = float(
                corrupted["t_air_wk_c"][k]
            ) - 0.8 * ((mins - 30.0) / 360.0)

    outcome, mem_out = calibration.calibrate_zone_a2(
        corrupted, "wk", true_wk, mode="manual"
    )
    assert outcome.accepted is False
    assert outcome.params_applied == true_wk
    assert outcome.params_candidate != true_wk
    assert "degraded" in outcome.reason.lower()
    assert mem_out is not None


def test_engine_snapshot_stack_8_deep_rollback_reset_and_48h_replay() -> None:
    """Verifies 8-deep snapshot stack, rollback, reset, & 48h state replay."""
    true_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK,
        c_floor_wh_k=4700.0,
        r_fh_k_kw=2.18,
        h_loss_w_k=51.5,
        a_sol_m2=4.10,
        k_hx_w_k=220.0,
    )
    true_kt = dataclasses.replace(
        priors.DAY1_PRIORS_KT,
        c_floor_wh_k=2000.0,
        r_fh_k_kw=4.30,
        h_loss_w_k=36.5,
        eta_ac=4.25,
        k_hx_w_k=57.0,
    )
    cols = _generate_calibration_dataset(
        n_days=10, params_wk=true_wk, params_kt=true_kt
    )

    eng = engine.DigitalTwinEngine()
    eng.replay_48h_from_columns(cols)
    floor_before_cal = eng.get_zone_snapshot("wk").state.t_floor_c
    assert len(eng.hindcast_wk.queue) == evaluation.HORIZON_6H_STEPS
    assert eng.get_zone_snapshot("wk").t_hindcast_6h_c is not None

    cal_res = eng.run_calibration(cols, mode="manual")
    assert cal_res["accepted_any"] is True
    assert cal_res["accepted_wk"] is True
    assert len(eng.snapshot_stack) == 1

    # Strict RFC-8259 / orjson compliance (`allow_nan=False`) + episode_bank reload
    strict_state_json = json.dumps(eng.to_state_dict(), allow_nan=False)
    eng_reloaded = engine.DigitalTwinEngine()
    eng_reloaded.load_state_dict(json.loads(strict_state_json))
    assert len(eng_reloaded.episode_bank.get_episodes("wk")) == len(
        eng.episode_bank.get_episodes("wk")
    )
    assert len(eng_reloaded.episode_bank.get_episodes("wk")) > 0

    learned_wk = eng.get_zone_snapshot("wk").params
    assert learned_wk != priors.DAY1_PRIORS_WK
    floor_after_cal = eng.get_zone_snapshot("wk").state.t_floor_c
    assert abs(floor_after_cal - floor_before_cal) > 0.02
    assert len(eng.hindcast_wk.queue) == evaluation.HORIZON_6H_STEPS

    eng.reset_parameters(cols)
    assert len(eng.snapshot_stack) == 2
    assert eng.get_zone_snapshot("wk").params == priors.DAY1_PRIORS_WK
    assert math.isclose(
        eng.get_zone_snapshot("wk").state.t_floor_c,
        floor_before_cal,
        abs_tol=1e-6,
    )

    ok_rb1 = eng.rollback_parameters(cols)
    assert ok_rb1 is True
    assert len(eng.snapshot_stack) == 1
    assert eng.get_zone_snapshot("wk").params == learned_wk
    assert math.isclose(
        eng.get_zone_snapshot("wk").state.t_floor_c,
        floor_after_cal,
        abs_tol=1e-6,
    )

    ok_rb2 = eng.rollback_parameters(cols)
    assert ok_rb2 is True
    assert len(eng.snapshot_stack) == 0
    assert eng.rollback_parameters(cols) is False

    for _ in range(11):
        eng.reset_parameters(columns=None)
    assert len(eng.snapshot_stack) == 8


def _reference_window_residuals(
    hist: preprocessing.PreprocessedHistory,
    zone: contracts.ZoneId,
    params: contracts.ZoneParameters,
    window_specs: Sequence[tuple[int, int, Sequence[int], float]],
) -> list[float]:
    """Reference evaluator via `reconstruct_kalman_states` + `propagate_open_loop_step`."""
    exogs = preprocessing.build_zone_exog_list(hist, zone)
    states, _, _, _ = evaluation.reconstruct_kalman_states(
        hist, zone, params, exogs
    )
    t_meas = hist.t_air_wk_c if zone == "wk" else hist.t_air_kt_c
    out: list[float] = []
    for k0_w, len_w, sample_pts, w_factor in window_specs:
        st0 = states[k0_w]
        st = contracts.ThermalState(
            t_air_c=st0.t_air_c,
            t_floor_c=st0.t_floor_c,
            d_est_w=0.0,
        )
        ptr = 0
        for s in range(1, len_w + 1):
            idx = k0_w + s
            q_fh_s = preprocessing.step_q_fh_for_params(
                hist, zone, idx, st.t_floor_c, params
            )
            q_ac_s = preprocessing.step_q_ac_for_params(hist, zone, idx, params)
            st = zoh_kalman.propagate_open_loop_step(
                state=st,
                params=params,
                exog=exogs[idx],
                q_fh_w=q_fh_s,
                q_ac_w=q_ac_s,
                dt_s=hist.dt_s[idx],
            )
            if ptr < len(sample_pts) and s == sample_pts[ptr]:
                out.append((st.t_air_c - t_meas[idx]) * w_factor)
                ptr += 1
    return out


def test_fast_tape_evaluator_matches_reference_kalman_and_zoh_rollout() -> None:
    """Proves fast-tape residuals match reference Kalman + ZOH within `< 1e-11 °C`."""
    cand_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK,
        c_floor_wh_k=4820.0,
        r_fh_k_kw=2.21,
        h_loss_w_k=54.3,
        a_sol_m2=4.22,
        k_hx_w_k=212.0,
        c_wind=0.07,
    )
    cand_kt = dataclasses.replace(
        priors.DAY1_PRIORS_KT,
        c_floor_wh_k=1980.0,
        r_fh_k_kw=4.35,
        h_loss_w_k=35.2,
        a_sol_m2=0.71,
        eta_ac=4.18,
        k_hx_w_k=55.0,
        c_wind=0.05,
    )

    cols = _generate_calibration_dataset(
        n_days=6, params_wk=cand_wk, params_kt=cand_kt
    )
    n_rows = len(cols["ts_utc"])
    # Exercise variable wind (>3 m/s), Tobit coarse plateau, heartbeat,
    # harvest flag, and open-window cooling drop.
    for k in range(n_rows):
        if k % 5 == 0:
            cols["v_wind_ms"][k] = 6.4
        if 180 <= k < 205:
            cols["t_air_wk_c"][k] = round(float(cols["t_air_wk_c"][180]), 1)
        if 420 <= k < 430:
            cols["t_air_kt_c" if "t_air_kt_c" in cols else "t_air_kantoor_c"][
                k
            ] = 19.13
        if 520 <= k < 528:
            cols["flags"][k] = int(cols["flags"][k]) | const.FLAG_Z1_OFF_HARVEST
            cols["u_pump_sec"][k] = 1.0
            cols["u_eff_bg"][k] = 1.0
    cols["t_air_wk_c"][700] = float(cols["t_air_wk_c"][699]) - 0.85

    hist = preprocessing.preprocess_history_columns(cols)
    for zone, cand_p in (("wk", cand_wk), ("kt", cand_kt)):
        z_id: contracts.ZoneId = "wk" if zone == "wk" else "kt"
        st_init = kalman_tape.reconstruct_states_tuples(hist, z_id, cand_p)
        wins = calibration.extract_clean_regime_windows(
            hist, z_id, [s[2] for s in st_init], min_k0=144
        )
        assert len(wins) >= 4
        specs = [
            (
                w.k0,
                w.length_steps,
                calibration._sample_rollout_steps(w.regime, w.length_steps),
                1.0 / calibration.RESIDUAL_SIGMA_C,
            )
            for w in wins
        ]
        tape = kalman_tape.build_zone_tape(hist, z_id)
        res_fast = kalman_tape.evaluate_windows_on_tape(tape, cand_p, specs)
        res_ref = _reference_window_residuals(hist, z_id, cand_p, specs)
        assert len(res_fast) == len(res_ref) > 25
        max_diff_c = max(
            abs(rf - rr) * calibration.RESIDUAL_SIGMA_C
            for rf, rr in zip(res_fast, res_ref)
        )
        assert max_diff_c < 1e-11


def test_lts_detection_and_filtering_in_khx_windows_episode_bank_and_cop() -> None:
    """Verifies LTS (`FLAG_LTS_BACKFILL` & legacy `v0.1.0`) filtering rules."""
    true_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK, k_hx_w_k=222.0, h_loss_w_k=52.0
    )
    true_kt = dataclasses.replace(
        priors.DAY1_PRIORS_KT, k_hx_w_k=58.0, eta_ac=4.20
    )
    raw_cols = _generate_calibration_dataset(
        n_days=10, params_wk=true_wk, params_kt=true_kt
    )

    # 1. Synthetic RC + cosine dataset must have ZERO false-positive LTS flags.
    is_lts_clean, miss_clean = preprocessing.detect_lts_and_missing_prim_rows(
        raw_cols
    )
    assert not any(is_lts_clean)
    assert not any(miss_clean)

    # Synthetic minimal dict without `t_wp_out_c`/`t_wp_in_c` but finite `p_prim_th_w`
    _, miss_synth = preprocessing.detect_lts_and_missing_prim_rows(
        {"ts_utc": [1_760_000_000, 1_760_000_300], "p_prim_th_w": [0.0, 1800.0]}
    )
    assert miss_synth == [False, False]

    _, miss_wp_nan = preprocessing.detect_lts_and_missing_prim_rows(
        {
            "ts_utc": [1_760_000_000, 1_760_000_300],
            "p_prim_th_w": [0.0, 0.0],
            "t_wp_out_c": [28.0, math.nan],
            "t_wp_in_c": [25.0, 25.0],
        }
    )
    assert miss_wp_nan == [False, True]

    # 2. Inject legacy `v0.1.0` hourly linearly interpolated LTS on Days 0..4
    #    (including a flat sandwiched hour at hour 2) + `FLAG_LTS_BACKFILL` on Day 5,
    #    with poisoned `p_prim_th_w` that would distort `K_hx` and `"heat"` fits.
    mixed_cols = {k: list(v) for k, v in raw_cols.items()}
    lts_hours = 5 * 24  # 120 hours = 1440 steps (`0..1440`)
    for h_idx in range(lts_hours):
        k0 = h_idx * 12
        for m in range(k0, k0 + 13):
            mixed_cols["flags"][m] = (
                const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
            )
            mixed_cols["t_out_weather_c"][m] = math.nan
            for vc in (
                "u_eff_slaapkamer",
                "u_eff_badkamer",
                "u_eff_werkkamer_1e",
                "u_eff_kamer_1e_midden",
                "u_eff_kamer_1e_zuid",
                "u_eff_badkamer_boven",
                "u_eff_logeerkamer",
            ):
                mixed_cols[vc][m] = 0.0

        for ch, base_v, amp_v in (
            ("t_out_dak_c", 2.0, 1.4),
            ("t_out_hm_c", 2.2, 1.4),
            ("t_air_wk_c", 19.6, 0.35),
            ("t_air_kantoor_c", 18.8, 0.25),
            ("p_p1_net_w", 240.0, 90.0),
        ):
            if h_idx == 2:
                # Flat sandwiched hour between hour 1 and hour 3
                v_start = float(mixed_cols[ch][k0])
                v_end = v_start
            else:
                v_start = (
                    float(mixed_cols[ch][k0])
                    if h_idx > 0
                    else (base_v + 0.1 * h_idx)
                )
                v_end = base_v + amp_v * math.sin(0.37 * (h_idx + 1))
            for j in range(13):
                mixed_cols[ch][k0 + j] = v_start + (v_end - v_start) * (
                    j / 12.0
                )

    # Day 5 (`1440..1728`) flagged explicitly with `FLAG_LTS_BACKFILL`
    for m in range(1440, 1728):
        mixed_cols["flags"][m] = (
            int(mixed_cols["flags"][m]) | const.FLAG_LTS_BACKFILL
        )

    # Poison heating steps inside `0..1728` with absurd `p_prim_th_w = 9500 W`
    for m in range(0, 1728):
        if float(mixed_cols["f_comp_hz"][m]) > 5.0:
            mixed_cols["p_prim_th_w"][m] = 9500.0

    is_lts_mix, miss_mix = preprocessing.detect_lts_and_missing_prim_rows(
        mixed_cols
    )
    assert all(is_lts_mix[:1728])
    assert all(miss_mix[:1728])
    assert not any(is_lts_mix[1729:])

    # `calibrate_k_hx_robust` skips all LTS rows and recovers clean `222 W/K`
    khx_mix = calibration.calibrate_k_hx_robust(
        mixed_cols,
        const.DEFAULT_FLOW_RATES_L_MIN,
        priors.DAY1_K_HX_BY_GROUP_W_K,
    )
    assert abs(khx_mix["wk"] - 222.0) / 222.0 < 0.06

    # `extract_clean_regime_windows`: ZERO `"heat"` / `"ac"` windows in `0..1728`,
    # passive `"night"` allowed on LTS only when WP & AC were off on `[k0-24, k1]`.
    hist_mix = preprocessing.preprocess_history_columns(mixed_cols)
    st_mix = kalman_tape.reconstruct_states_tuples(hist_mix, "wk", true_wk)
    wins_wk = calibration.extract_clean_regime_windows(
        hist_mix, "wk", [s[2] for s in st_mix], min_k0=144
    )
    for w in wins_wk:
        if w.regime == "heat":
            assert w.k0 > 1728
    lts_night = [
        w for w in wins_wk if w.regime == "night" and w.k0 + w.length_steps < 1728
    ]
    assert len(lts_night) >= 1

    # Contaminating LTS night with `f_comp_hz = 20 Hz` (without `q_fh`) must block it
    contam_night_cols = {k: list(v) for k, v in mixed_cols.items()}
    for w_n in lts_night:
        contam_night_cols["f_comp_hz"][w_n.k0 + 5] = 20.0
        contam_night_cols["p_wp_raw_el_w"][w_n.k0 + 5] = 600.0
        contam_night_cols["p_prim_th_w"][w_n.k0 + 5] = 0.0
    hist_cn = preprocessing.preprocess_history_columns(contam_night_cols)
    st_cn = kalman_tape.reconstruct_states_tuples(hist_cn, "wk", true_wk)
    wins_cn = calibration.extract_clean_regime_windows(
        hist_cn, "wk", [s[2] for s in st_cn], min_k0=144
    )
    assert not [
        w for w in wins_cn if w.regime == "night" and w.k0 == lts_night[0].k0
    ]

    # `SeasonalEpisodeBank` filters persisted `"heat"` / `"airco_heat"` LTS episodes
    bank = calibration.SeasonalEpisodeBank()
    bank.update_from_columns(raw_cols)
    raw_bank_dict = bank.to_dict()
    assert len(raw_bank_dict["buckets"]["wk"]["heat"]) >= 1
    for ep_d in raw_bank_dict["buckets"]["wk"]["heat"]:
        ep_d["columns"]["flags"][10] = (
            int(ep_d["columns"]["flags"][10]) | const.FLAG_LTS_BACKFILL
        )
    for ep_d in raw_bank_dict["buckets"]["kt"]["ac"]:
        ep_d["regime"] = "airco_heat"
        ep_d["columns"]["flags"][5] = (
            int(ep_d["columns"]["flags"][5]) | const.FLAG_LTS_BACKFILL
        )
    cleaned_bank = calibration.SeasonalEpisodeBank.from_dict(raw_bank_dict)
    assert cleaned_bank.get_episodes("wk", "heat") == []
    assert cleaned_bank.get_episodes("kt", "ac") == []
    assert len(cleaned_bank.get_episodes("wk", "night")) >= 1

    # Cold-start COP fallback: 7d LTS + 3d raw (< 7d non-LTS in `train_slice`)
    # must train COP on `columns` and fit the raw >=60m CV runs!
    cold_cop_cols = {k: list(v) for k, v in raw_cols.items()}
    for m in range(0, 7 * const.STEPS_PER_DAY):
        cold_cop_cols["flags"][m] = (
            int(cold_cop_cols["flags"][m]) | const.FLAG_LTS_BACKFILL
        )
    true_cop = contracts.CopParameters(
        eta_carnot_cv=0.55,
        beta_defrost=0.20,
        eta_carnot_dhw=0.40,
        delta_p_wp_bias_w=145.0,
    )
    for m in range(7 * const.STEPS_PER_DAY, 10 * const.STEPS_PER_DAY):
        if float(cold_cop_cols["f_comp_hz"][m]) > 5.0:
            t_o = float(cold_cop_cols["t_out_dak_c"][m])
            t_s = float(cold_cop_cols["t_wp_out_c"][m])
            cop_v = (
                true_cop.eta_carnot_cv
                * ((t_s + 276.15) / max(14.0, (t_s + 3.0) - (t_o - 4.0)))
            )
            cold_cop_cols["p_prim_th_w"][m] = cop_v * (600.0 + 145.0)

    eng_cold = engine.DigitalTwinEngine()
    eng_cold.run_calibration(cold_cop_cols, mode="manual")
    assert eng_cold.cop_diagnostics["n_cv_runs"] >= 2
    assert abs(eng_cold.cop_params.eta_carnot_cv - 0.55) < 0.03


def test_three_phase_lock_free_calibration_zero_mutation_live_ticks_and_epoch() -> None:
    """Verifies 3-phase calibration API (`ctx -> plan -> apply`) & epoch guard."""
    true_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK,
        c_floor_wh_k=4720.0,
        r_fh_k_kw=2.17,
        h_loss_w_k=51.0,
        a_sol_m2=4.12,
        k_hx_w_k=220.0,
    )
    true_kt = dataclasses.replace(
        priors.DAY1_PRIORS_KT,
        c_floor_wh_k=1980.0,
        r_fh_k_kw=4.28,
        h_loss_w_k=36.0,
        eta_ac=4.25,
        k_hx_w_k=56.5,
    )
    cols = _generate_calibration_dataset(
        n_days=10, params_wk=true_wk, params_kt=true_kt
    )

    eng = engine.DigitalTwinEngine()
    eng.replay_48h_from_columns(cols)
    epoch_0 = eng.param_epoch
    state_json_before = json.dumps(eng.to_state_dict(), sort_keys=True)

    # Phase 1: capture context (<1 ms)
    t_ctx0 = time.perf_counter()
    ctx = eng.capture_calibration_context()
    ctx_ms = (time.perf_counter() - t_ctx0) * 1000.0
    assert ctx_ms < 50.0
    assert ctx["param_epoch"] == epoch_0

    # Phase 2: pure `compute_calibration_plan` must mutate `eng` 0%
    plan = eng.compute_calibration_plan(cols, ctx, mode="manual")
    assert eng.param_epoch == epoch_0
    assert json.dumps(eng.to_state_dict(), sort_keys=True) == state_json_before

    # Simulate 2 live 5-minute ticks arriving while executor was computing `plan`
    recent_cols = {k: list(v) for k, v in cols.items()}
    for extra_idx in (1, 2):
        new_ts = int(cols["ts_utc"][-1]) + extra_idx * const.DT_STEP_S
        raw_tick: dict[str, float | int] = {
            k: cols[k][-1] for k in cols
        }
        raw_tick["ts_utc"] = new_ts
        raw_tick["t_air_wk_c"] = float(cols["t_air_wk_c"][-1]) - 0.01 * extra_idx
        traj = eng.process_5m_window(raw_tick)
        for k in recent_cols:
            if k in traj:
                recent_cols[k].append(traj[k])
            elif k in raw_tick:
                recent_cols[k].append(raw_tick[k])
            else:
                recent_cols[k].append(recent_cols[k][-1])

    # Live 5m ticks must NOT bump `param_epoch`
    assert eng.param_epoch == epoch_0

    # Phase 3: `apply_calibration_plan` applies learned params & preserves new ticks
    res = eng.apply_calibration_plan(plan, recent_cols)
    assert res["accepted_any"] is True
    assert eng.param_epoch == epoch_0 + 1
    assert eng.get_zone_snapshot("wk").ts_utc == int(recent_cols["ts_utc"][-1])
    assert eng.eval_report_wk is not None
    assert eng.eval_report_kt is not None

    # Stale epoch guard: concurrent `reset_parameters` between Phase 1 and Phase 3
    ctx_stale = eng.capture_calibration_context()
    plan_stale = eng.compute_calibration_plan(recent_cols, ctx_stale, mode="manual")
    eng.reset_parameters(columns=None)
    epoch_after_reset = eng.param_epoch
    assert epoch_after_reset == epoch_0 + 2

    aborted = eng.apply_calibration_plan(plan_stale, recent_cols)
    assert aborted["aborted_stale_epoch"] is True
    assert aborted["accepted_wk"] is False
    assert aborted["accepted_kt"] is False
    assert eng.param_epoch == epoch_after_reset
    assert eng.get_zone_snapshot("wk").params == priors.DAY1_PRIORS_WK

    # Empty plan safety
    empty_plan = eng.compute_calibration_plan(
        {"ts_utc": []}, eng.capture_calibration_context(), mode="manual"
    )
    empty_res = eng.apply_calibration_plan(empty_plan, {"ts_utc": []})
    assert empty_res["empty"] is True
    assert empty_res["accepted_any"] is False


def test_90_day_25920_row_calibration_speed_benchmark() -> None:
    """Verifies full 90-day (`25,920`-row) calibration completes in seconds."""
    true_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK,
        c_floor_wh_k=4760.0,
        r_fh_k_kw=2.19,
        h_loss_w_k=52.0,
        a_sol_m2=4.18,
        k_hx_w_k=219.0,
    )
    true_kt = dataclasses.replace(
        priors.DAY1_PRIORS_KT,
        c_floor_wh_k=1960.0,
        r_fh_k_kw=4.26,
        h_loss_w_k=36.2,
        a_sol_m2=0.69,
        eta_ac=4.28,
        k_hx_w_k=56.5,
    )
    base_9d = _generate_calibration_dataset(
        n_days=9, params_wk=true_wk, params_kt=true_kt
    )
    chunk_len = len(base_9d["ts_utc"])
    ts0 = int(base_9d["ts_utc"][0])
    cols_90d: dict[str, list[float | int]] = {}
    for col_name, seq in base_9d.items():
        if col_name == "ts_utc":
            cols_90d[col_name] = [
                ts0 + k * const.DT_STEP_S for k in range(10 * chunk_len)
            ]
        else:
            cols_90d[col_name] = list(seq) * 10

    assert len(cols_90d["ts_utc"]) == 25_920

    eng = engine.DigitalTwinEngine()
    t0 = time.perf_counter()
    res = eng.run_calibration(cols_90d, mode="manual")
    elapsed_s = time.perf_counter() - t0

    assert res["accepted_any"] is True
    assert res["accepted_wk"] is True
    assert res["accepted_kt"] is True
    assert elapsed_s < 10.0


def test_f41_freeze_eta_ac_and_a_sol_f43_bound_warnings_and_f33_heat_guard() -> None:
    """Verifies F4.1 freezing, F4.3 bound_warnings, F3.3 heat guard, & K_hx cap."""
    # 1. F4.1: Freeze `eta_ac` when < 4 AC windows (remove AC runs on kt)
    true_kt_no_ac = dataclasses.replace(
        priors.DAY1_PRIORS_KT,
        h_loss_w_k=35.0,
        eta_ac=4.70,
        k_hx_w_k=48.0,
    )
    cols_no_ac = _generate_calibration_dataset(
        n_days=8,
        params_wk=priors.DAY1_PRIORS_WK,
        params_kt=true_kt_no_ac,
        solar_enabled=False,
    )
    # Strip AC heating runs so kt has 0 (< 4) AC windows and 0 (< 2) solar windows
    cols_no_ac["p_el_ac_kt_w"] = [0.0] * len(cols_no_ac["ts_utc"])
    cols_no_ac["ac_mode_code"] = [float(const.AC_MODE_OFF)] * len(
        cols_no_ac["ts_utc"]
    )
    # Re-generate clean kt temperatures without AC so calibration converges cleanly
    cols_clean = _generate_calibration_dataset(
        n_days=8,
        params_wk=priors.DAY1_PRIORS_WK,
        params_kt=dataclasses.replace(true_kt_no_ac, eta_ac=0.0),
        solar_enabled=False,
    )
    cols_clean["p_el_ac_kt_w"] = [0.0] * len(cols_clean["ts_utc"])
    cols_clean["ac_mode_code"] = [float(const.AC_MODE_OFF)] * len(
        cols_clean["ts_utc"]
    )

    out_frozen, _ = calibration.calibrate_zone_a2(
        cols_clean,
        "kt",
        dataclasses.replace(priors.DAY1_PRIORS_KT, eta_ac=4.20),
        mode="manual",
    )
    # `eta_ac` and `a_sol_m2` must be frozen at aged prior (`3.60` / `0.40`)
    assert math.isclose(
        out_frozen.params_map_raw.eta_ac,
        priors.DAY1_PRIORS_KT.eta_ac,
        abs_tol=1e-9,
    )
    assert math.isclose(
        out_frozen.params_map_raw.a_sol_m2,
        priors.DAY1_PRIORS_KT.a_sol_m2,
        abs_tol=1e-9,
    )

    # 2. F4.3: Parameter hitting lower bound emits `bound_warnings` & persists
    true_kt_low_loss = dataclasses.replace(
        priors.DAY1_PRIORS_KT,
        h_loss_w_k=11.0,  # Below PARAM_BOUNDS_KT["h_loss_w_k"].lower == 12.0
        k_hx_w_k=48.0,
    )
    cols_bound = _generate_calibration_dataset(
        n_days=10,
        params_wk=priors.DAY1_PRIORS_WK,
        params_kt=true_kt_low_loss,
    )
    eng_b = engine.DigitalTwinEngine()
    res_b = eng_b.run_calibration(cols_bound, mode="manual")
    out_kt_b: contracts.CalibrationZoneOutcome = res_b["outcome_kt"]
    assert out_kt_b.accepted is True
    assert any("h_loss_w_k" in w for w in out_kt_b.bound_warnings)
    diag_kt = eng_b.get_zone_fit_diagnostics("kt")
    assert diag_kt["bound_warnings"] == out_kt_b.bound_warnings

    # Survives state serialization round-trip
    eng_b2 = engine.DigitalTwinEngine()
    eng_b2.load_state_dict(json.loads(json.dumps(eng_b.to_state_dict())))
    assert eng_b2.get_zone_fit_diagnostics("kt")["bound_warnings"] == (
        out_kt_b.bound_warnings
    )

    # 3. K_hx physical epsilon ceiling (<= 0.75 * flow * 69.73) & other_u <= 1.5
    cols_high_khx = _generate_calibration_dataset(
        n_days=8,
        params_wk=dataclasses.replace(
            priors.DAY1_PRIORS_WK, k_hx_w_k=245.0
        ),
        params_kt=priors.DAY1_PRIORS_KT,
    )
    # Artificially inflate primary thermal power 1.6x -> must clamp at eps<=0.75
    cols_high_khx["p_prim_th_w"] = [
        float(v) * 1.6 for v in cols_high_khx["p_prim_th_w"]
    ]
    khx_capped = calibration.calibrate_k_hx_robust(
        cols_high_khx,
        const.DEFAULT_FLOW_RATES_L_MIN,
        priors.DAY1_K_HX_BY_GROUP_W_K,
    )
    wk_cap = min(
        1.45 * priors.DAY1_K_HX_BY_GROUP_W_K["wk"],
        0.75
        * const.DEFAULT_FLOW_RATES_L_MIN["wk"]
        * priors.WATER_POWER_FACTOR_W_PER_LMIN_K,
    )
    assert khx_capped["wk"] <= wk_cap + 1e-6

    # When other_u > 1.5 during all steady steps, wk_mix_vals must be ignored
    cols_multi = {k: list(v) for k, v in cols_high_khx.items()}
    cols_multi["u_eff_slaapkamer"] = [1.0] * len(cols_multi["ts_utc"])
    cols_multi["u_eff_badkamer"] = [1.0] * len(cols_multi["ts_utc"])
    khx_ignored = calibration.calibrate_k_hx_robust(
        cols_multi,
        const.DEFAULT_FLOW_RATES_L_MIN,
        priors.DAY1_K_HX_BY_GROUP_W_K,
    )
    assert khx_ignored["wk"] == priors.DAY1_K_HX_BY_GROUP_W_K["wk"]

    # 4. F3.3: Relative `heating_and_coast` holdout guard
    # Rejects when heating_and_coast degrades even if overall J_eval improves,
    # and never blocks improvements from large negative bias (-0.85 -> -0.24).
    true_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK,
        c_floor_wh_k=4700.0,
        r_fh_k_kw=2.20,
        h_loss_w_k=52.0,
    )
    cols_hc = _generate_calibration_dataset(
        n_days=10, params_wk=true_wk, params_kt=priors.DAY1_PRIORS_KT
    )
    # Start from biased prior (`h_loss=95`, producing large negative bias):
    # improving toward truth must NOT be blocked by relative bias guard.
    bad_start = dataclasses.replace(priors.DAY1_PRIORS_WK, h_loss_w_k=95.0)
    out_imp, _ = calibration.calibrate_zone_a2(
        cols_hc, "wk", bad_start, mode="manual"
    )
    assert out_imp.accepted is True

    # Corrupt training heating runs only -> worsens holdout heating_and_coast
    train_end_hc = preprocessing.compute_train_holdout_split(cols_hc)
    cols_bad_heat = {k: list(v) for k, v in cols_hc.items()}
    for k in range(train_end_hc):
        if float(cols_bad_heat["u_pump_sec"][k]) > 0.5:
            cols_bad_heat["t_air_wk_c"][k] = (
                float(cols_bad_heat["t_air_wk_c"][k]) + 1.4
            )
    out_rej_hc, _ = calibration.calibrate_zone_a2(
        cols_bad_heat, "wk", true_wk, mode="manual"
    )
    assert out_rej_hc.accepted is False
    assert "degraded" in out_rej_hc.reason.lower()


def test_calibrate_k_hx_robust_10_lmin_flow_cap_and_buffer_offsets() -> None:
    """F3.1 & F3.2: 10 L/min `K_hx` reaches up to 523 W/K & honors offsets."""
    flows_10 = dict(const.DEFAULT_FLOW_RATES_L_MIN, wk=10.0, gang=2.0)
    true_khx_wk = 455.0  # Above old 355.25 cap, below 522.975 physical cap
    true_wk = dataclasses.replace(priors.DAY1_PRIORS_WK, k_hx_w_k=true_khx_wk)
    cols = _generate_calibration_dataset(
        n_days=6, params_wk=true_wk, params_kt=priors.DAY1_PRIORS_KT
    )
    init_khx_10 = {
        g: hydraulics.scale_k_hx_eps_ntu(
            priors.DAY1_K_HX_BY_GROUP_W_K[g],
            const.DEFAULT_FLOW_RATES_L_MIN[g],
            flows_10[g],
        )
        for g in const.ALL_HYDRAULIC_GROUPS
    }
    # Reconstruct `tf_wk` on the 10 L/min hydraulic pass and synthesize exact
    # BG-solo `p_prim_th_w` matching `true_khx_wk = 455.0` and `gang/wk = 0.2`.
    st_wk = kalman_tape.reconstruct_states_tuples(
        preprocessing.preprocess_history_columns(
            cols,
            flow_rates_l_min=flows_10,
            k_hx_by_group=init_khx_10,
        ),
        "wk",
        dataclasses.replace(priors.DAY1_PRIORS_WK, k_hx_w_k=init_khx_10["wk"]),
    )
    for k in range(len(cols["ts_utc"])):
        if (
            float(cols["f_comp_hz"][k]) >= 18.0
            and float(cols["u_eff_bg"][k]) > 0.9
            and float(cols["u_eff_kantoor"][k]) < 0.05
        ):
            t_top = float(cols["t_buf_top_end_c"][k])
            tf_k = st_wk[k][1]
            q_wk = true_khx_wk * max(0.0, t_top - tf_k)
            q_gang = (true_khx_wk * 0.2) * max(0.0, t_top - (tf_k - 0.5))
            cols["p_prim_th_w"][k] = q_wk + q_gang
            cols["v_prim_l_min"][k] = 16.0

    fit_10 = calibration.calibrate_k_hx_robust(cols, flows_10, init_khx_10)
    assert fit_10["wk"] > 355.25
    assert abs(fit_10["wk"] - true_khx_wk) / true_khx_wk < 0.03

    # Inflating primary power reaches `0.75 * 10.0 * 69.73 = 522.975` cap.
    cols_cap = {k: list(v) for k, v in cols.items()}
    cols_cap["p_prim_th_w"] = [float(v) * 1.45 for v in cols["p_prim_th_w"]]
    fit_cap = calibration.calibrate_k_hx_robust(cols_cap, flows_10, init_khx_10)
    phys_cap_10 = 0.75 * 10.0 * priors.WATER_POWER_FACTOR_W_PER_LMIN_K
    assert 515.0 < fit_cap["wk"] <= phys_cap_10 + 1e-6

    # Non-zero BufferSensorOffsets: shift raw sensors by (-top_off, -bot_off)
    # and verify `calibrate_k_hx_robust(..., buffer_offsets=...)` matches.
    custom_off = contracts.BufferSensorOffsets(
        buf_top_offset_c=-1.35,
        buf_bot_offset_c=0.45,
    )
    cols_shifted = {k: list(v) for k, v in cols.items()}
    cols_shifted["t_buf_top_start_c"] = [
        float(v) - custom_off.buf_top_offset_c
        for v in cols["t_buf_top_start_c"]
    ]
    cols_shifted["t_buf_top_end_c"] = [
        float(v) - custom_off.buf_top_offset_c for v in cols["t_buf_top_end_c"]
    ]
    cols_shifted["t_buf_bot_start_c"] = [
        float(v) + priors.BUF_BOT_SENSOR_OFFSET_C - custom_off.buf_bot_offset_c
        for v in cols["t_buf_bot_start_c"]
    ]
    cols_shifted["t_buf_bot_end_c"] = [
        float(v) + priors.BUF_BOT_SENSOR_OFFSET_C - custom_off.buf_bot_offset_c
        for v in cols["t_buf_bot_end_c"]
    ]
    fit_with_off = calibration.calibrate_k_hx_robust(
        cols_shifted,
        flows_10,
        init_khx_10,
        buffer_offsets=custom_off,
    )
    fit_without_off = calibration.calibrate_k_hx_robust(
        cols_shifted,
        flows_10,
        init_khx_10,
    )
    assert math.isclose(fit_with_off["wk"], fit_10["wk"], rel_tol=1e-6)
    assert abs(fit_without_off["wk"] - fit_10["wk"]) > 15.0


def test_subregime_min_veto_windows_f33_and_pure_holdout_before(
    monkeypatch: Any,
) -> None:
    """F3.3: `n_windows < 5` cannot veto overall improvement; `>= 5` vetoes."""
    assert calibration.MIN_REGIME_VETO_WINDOWS == 5
    true_wk = dataclasses.replace(priors.DAY1_PRIORS_WK, h_loss_w_k=51.0)
    cols = _generate_calibration_dataset(
        n_days=8, params_wk=true_wk, params_kt=priors.DAY1_PRIORS_KT
    )

    def _metrics(
        reg: str, n_win: int, j_val: float
    ) -> contracts.RegimeHindcastMetrics:
        return contracts.RegimeHindcastMetrics(
            regime=reg,
            horizon_steps=72,
            n_windows=n_win,
            rmse_end_c=j_val,
            rmse_path_c=j_val,
            mae_end_c=j_val * 0.8,
            bias_end_c=0.0,
            j_eval_c=j_val,
            prior_rmse_end_c=0.60,
            skill_score_pct=25.0,
        )

    seen_khx_calls: list[float] = []
    sunny_n_windows = 2

    def _fake_eval(
        columns: Any,
        zone: contracts.ZoneId,
        params_eval: contracts.ZoneParameters,
        params_prior: contracts.ZoneParameters,
        flow_rates_l_min: Any = None,
        k_hx_by_group: Any = None,
        **kwargs: Any,
    ) -> contracts.ZoneEvaluationReport:
        del columns, params_prior, flow_rates_l_min, kwargs
        seen_khx_calls.append(
            float((k_hx_by_group or {}).get("wk", params_eval.k_hx_w_k))
        )
        is_before = math.isclose(
            params_eval.h_loss_w_k, priors.DAY1_PRIORS_WK.h_loss_w_k
        )
        j_ov = 0.52 if is_before else 0.31
        j_sun = 0.20 if is_before else 0.29  # +0.09 C (> +0.03 C threshold)
        return contracts.ZoneEvaluationReport(
            zone=zone,
            ts_evaluated_utc=1_760_000_000,
            overall_6h=_metrics("overall_6h", 18, j_ov),
            by_regime={
                "heating_and_coast": _metrics(
                    "heating_and_coast", 8, 0.45 if is_before else 0.28
                ),
                "night_cooling": _metrics("night_cooling", 2, j_sun),
                "sunny_afternoon": _metrics(
                    "sunny_afternoon", sunny_n_windows, j_sun
                ),
            },
        )

    monkeypatch.setattr(evaluation, "evaluate_14d_hindcast", _fake_eval)

    # 1. With `n_windows = 2` (< 5), noisy sub-regimes do not veto `-40%` J.
    out_acc, _ = calibration.calibrate_zone_a2(
        cols,
        "wk",
        priors.DAY1_PRIORS_WK,
        mode="manual",
        k_hx_by_group={"wk": 310.0},
        old_k_hx_by_group={"wk": 245.0},
    )
    assert out_acc.accepted is True
    assert seen_khx_calls == [245.0, 310.0]

    # 2. With `n_windows = 5` (>= 5), +0.09 C on sunny_afternoon vetoes!
    sunny_n_windows = 5
    out_veto, _ = calibration.calibrate_zone_a2(
        cols,
        "wk",
        priors.DAY1_PRIORS_WK,
        mode="manual",
        k_hx_by_group={"wk": 310.0},
        old_k_hx_by_group={"wk": 245.0},
    )
    assert out_veto.accepted is False
    assert "sunny_afternoon" in out_veto.reason


def test_engine_buffer_offsets_calibration_damping_rollback_and_reset() -> None:
    """F3.2 & F4: 0 Hz sniffing cal, 80/20 damping, rollback, & reset."""
    true_wk = dataclasses.replace(
        priors.DAY1_PRIORS_WK,
        c_floor_wh_k=4750.0,
        r_fh_k_kw=2.20,
        h_loss_w_k=52.0,
        k_hx_w_k=222.0,
    )
    cols = _generate_calibration_dataset(
        n_days=9, params_wk=true_wk, params_kt=priors.DAY1_PRIORS_KT
    )
    # Inject 10 consecutive stationary 0 Hz primary sniffing steps on Day 1
    # night (`01:00..01:50` -> indices `300..309` in training split):
    # `t_wp_out == t_wp_in == 24.0`, raw top = 23.2 (+0.8), raw bot = 24.6.
    for k in range(300, 310):
        cols["flags"][k] = const.FLAG_VALID_WK | const.FLAG_VALID_KT
        cols["f_comp_hz"][k] = 0.0
        cols["u_pump_sec"][k] = 0.0
        cols["v_prim_l_min"][k] = 14.5
        cols["t_wp_out_c"][k] = 24.0
        cols["t_wp_in_c"][k] = 24.0
        cols["t_buf_top_start_c"][k] = 23.2
        cols["t_buf_top_end_c"][k] = 23.2
        cols["t_buf_bot_start_c"][k] = 24.6
        cols["t_buf_bot_end_c"][k] = 24.6

    eng = engine.DigitalTwinEngine()
    assert eng.buffer_offsets == priors.DAY1_BUFFER_OFFSETS

    # 1. `auto_weekly` blends 80% old + 20% raw on buffer offsets
    eng.run_calibration(cols, mode="auto_weekly")
    assert math.isclose(
        eng.buffer_offsets.buf_top_offset_c, 0.20 * 0.80, abs_tol=1e-6
    )
    assert math.isclose(
        eng.buffer_offsets.buf_bot_offset_c,
        0.80 * priors.BUF_BOT_SENSOR_OFFSET_C + 0.20 * (-0.60),
        abs_tol=1e-6,
    )
    assert eng.hydraulics.buffer_offsets == eng.buffer_offsets
    assert eng.prev_buffer_offsets == priors.DAY1_BUFFER_OFFSETS

    # 2. `manual` applies 100% raw offsets (+0.80, -0.60) & persists in state
    eng.run_calibration(cols, mode="manual")
    assert math.isclose(
        eng.buffer_offsets.buf_top_offset_c, 0.80, abs_tol=1e-6
    )
    assert math.isclose(
        eng.buffer_offsets.buf_bot_offset_c, -0.60, abs_tol=1e-6
    )

    # Engine `predict_live_expected_cop` forwards `self.buffer_offsets` at 0 Hz
    live_cop = eng.predict_live_expected_cop(
        {
            "t_out_dak_c": 5.0,
            "rh_out_pct": 80.0,
            "f_comp_hz": 0.0,
            "t_wp_out_c": 36.0,
            "t_buf_top_end_c": 26.2,  # 26.2 + 0.80 = 27.0 C
        }
    )
    assert math.isclose(
        live_cop, eng.predict_cop_cv(5.0, 27.0, 80.0), rel_tol=1e-9
    )

    saved = json.loads(json.dumps(eng.to_state_dict(), allow_nan=False))
    eng_reloaded = engine.DigitalTwinEngine()
    eng_reloaded.load_state_dict(saved)
    assert eng_reloaded.buffer_offsets == eng.buffer_offsets
    assert eng_reloaded.hydraulics.buffer_offsets == eng.buffer_offsets

    # 3. `reset_parameters_to_day1()` resets to DAY1; rollback restores
    manual_offsets = eng.buffer_offsets
    eng.reset_parameters_to_day1(columns=None)
    assert eng.buffer_offsets == priors.DAY1_BUFFER_OFFSETS
    assert eng.hydraulics.buffer_offsets == priors.DAY1_BUFFER_OFFSETS
    assert eng.prev_buffer_offsets == manual_offsets

    assert eng.rollback_parameters(columns=None) is True
    assert eng.buffer_offsets == manual_offsets
    assert eng.hydraulics.buffer_offsets == manual_offsets


def test_heating_and_coast_bias_veto_respects_substantial_j_gain(
    monkeypatch: Any,
) -> None:
    """Allows moderate |bias_end| (<=0.35 C) when J_eval drops >=0.05 C."""
    true_wk = dataclasses.replace(priors.DAY1_PRIORS_WK, h_loss_w_k=52.0)
    cols = _generate_calibration_dataset(
        n_days=8, params_wk=true_wk, params_kt=priors.DAY1_PRIORS_KT
    )

    scenario = {
        "ov_before": 0.51,
        "ov_after": 0.35,
        "hc_before": 0.45,
        "hc_after": 0.34,
        "bias_before": 0.10,
        "bias_after": 0.31,
    }

    def _metrics(
        reg: str, n_win: int, j_val: float, bias_val: float
    ) -> contracts.RegimeHindcastMetrics:
        return contracts.RegimeHindcastMetrics(
            regime=reg,
            horizon_steps=72,
            n_windows=n_win,
            rmse_end_c=j_val,
            rmse_path_c=j_val,
            mae_end_c=j_val,
            bias_end_c=bias_val,
            j_eval_c=j_val,
            prior_rmse_end_c=0.55,
            skill_score_pct=20.0,
        )

    call_idx = 0

    def _fake_eval(
        _cols: Any,
        zone: contracts.ZoneId,
        *_args: Any,
        **_kwargs: Any,
    ) -> contracts.ZoneEvaluationReport:
        nonlocal call_idx
        is_before = (call_idx % 2) == 0
        call_idx += 1
        j_ov = scenario["ov_before"] if is_before else scenario["ov_after"]
        j_hc = scenario["hc_before"] if is_before else scenario["hc_after"]
        b_hc = scenario["bias_before"] if is_before else scenario["bias_after"]
        return contracts.ZoneEvaluationReport(
            zone=zone,
            ts_evaluated_utc=1_760_000_000,
            overall_6h=_metrics("overall_6h", 40, j_ov, b_hc),
            by_regime={
                "heating_and_coast": _metrics(
                    "heating_and_coast", 20, j_hc, b_hc
                ),
            },
        )

    monkeypatch.setattr(evaluation, "evaluate_14d_hindcast", _fake_eval)

    # 1. Large J_eval improvement (-0.16 C) + |bias_end| 0.10 -> 0.31 C: ACCEPT
    out_acc, _ = calibration.calibrate_zone_a2(
        cols, "wk", priors.DAY1_PRIORS_WK, mode="manual"
    )
    assert out_acc.accepted is True

    # 2. Marginal J_eval gain (-0.01 C) + |bias_end| 0.10 -> 0.22 C: REJECT
    scenario.update(
        ov_before=0.35,
        ov_after=0.34,
        hc_before=0.34,
        hc_after=0.33,
        bias_before=0.10,
        bias_after=0.22,
    )
    out_marg, _ = calibration.calibrate_zone_a2(
        cols, "wk", priors.DAY1_PRIORS_WK, mode="manual"
    )
    assert out_marg.accepted is False
    assert "|bias_end| degraded" in out_marg.reason

    # 3. Severe |bias_end| (> 0.35 C) still REJECTED even with large J_eval gain
    scenario.update(
        ov_before=0.55,
        ov_after=0.38,
        hc_before=0.50,
        hc_after=0.38,
        bias_before=0.10,
        bias_after=0.38,
    )
    out_sev, _ = calibration.calibrate_zone_a2(
        cols, "wk", priors.DAY1_PRIORS_WK, mode="manual"
    )
    assert out_sev.accepted is False
    assert "|bias_end| degraded" in out_sev.reason
