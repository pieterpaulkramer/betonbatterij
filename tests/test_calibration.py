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
from typing import Any

from custom_components.betonbatterij import const
from custom_components.betonbatterij import house_config
from custom_components.betonbatterij.domain import calibration
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import engine
from custom_components.betonbatterij.domain import evaluation
from custom_components.betonbatterij.domain import hydraulics
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
    k_hx_map = dict(house_config.DAY1_K_HX_BY_GROUP_W_K)
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
        hr_loc, mn_loc = solar_weather.local_hour_minute(ts_k)
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
                elif day_idx == 5 and 810 <= mins_loc < 855:
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
        t_bot_raw = t_bot_cal - house_config.BUF_BOT_SENSOR_OFFSET_C

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
        t_dn_wk = house_config.T_GROUND_MEAN_C
        q_int_wk = house_config.BASELOAD_WK_W + max(
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
        q_ac_kt = evaluation.compute_ac_heating_power_w(
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
        house_config.DAY1_PRIORS_WK,
        c_floor_wh_k=4750.0,
        r_fh_k_kw=2.18,
        h_loss_w_k=51.0,
        a_sol_m2=4.15,
        k_hx_w_k=218.0,
    )
    true_kt = dataclasses.replace(
        house_config.DAY1_PRIORS_KT,
        c_floor_wh_k=1950.0,
        r_fh_k_kw=4.25,
        h_loss_w_k=36.0,
        a_sol_m2=0.68,
        eta_ac=4.30,
        k_hx_w_k=56.0,
    )

    cols = _generate_calibration_dataset(
        n_days=10, params_wk=true_wk, params_kt=true_kt
    )

    khx_fit = calibration.calibrate_k_hx_robust(
        cols,
        const.DEFAULT_FLOW_RATES_L_MIN,
        house_config.DAY1_K_HX_BY_GROUP_W_K,
    )
    assert abs(khx_fit["wk"] - 218.0) / 218.0 < 0.05
    assert abs(khx_fit["kantoor"] - 56.0) / 56.0 < 0.06

    out_wk, mem_wk = calibration.calibrate_zone_a2(
        cols,
        "wk",
        house_config.DAY1_PRIORS_WK,
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
        house_config.DAY1_PRIORS_WK.h_loss_w_k - true_wk.h_loss_w_k
    )
    assert abs(out_wk.params_applied.a_sol_m2 - true_wk.a_sol_m2) < 0.40
    assert "c_floor_wh_k" in mem_wk.mean

    out_kt, _ = calibration.calibrate_zone_a2(
        cols,
        "kt",
        house_config.DAY1_PRIORS_KT,
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
        house_config.DAY1_PRIORS_KT.h_loss_w_k - true_kt.h_loss_w_k
    )

    out_auto, _ = calibration.calibrate_zone_a2(
        cols,
        "wk",
        house_config.DAY1_PRIORS_WK,
        mode="auto_weekly",
        flow_rates_l_min=const.DEFAULT_FLOW_RATES_L_MIN,
        k_hx_by_group=khx_fit,
    )
    assert out_auto.accepted is True
    expected_h_loss = (
        0.80 * house_config.DAY1_PRIORS_WK.h_loss_w_k
        + 0.20 * out_auto.params_map_raw.h_loss_w_k
    )
    assert math.isclose(
        out_auto.params_candidate.h_loss_w_k, expected_h_loss, abs_tol=1e-9
    )


def test_cooling_exclusion_with_2h_tail_and_seasonal_gate() -> None:
    """Verifies autonomous seasonal gate (`<14 °C`) & +2.0h cooling tail."""
    cols = _generate_calibration_dataset(
        n_days=8,
        params_wk=house_config.DAY1_PRIORS_WK,
        params_kt=house_config.DAY1_PRIORS_KT,
    )
    ok_winter, _ = calibration.check_seasonal_calibration_gate(cols)
    assert ok_winter is True

    warm_cols = _generate_calibration_dataset(
        n_days=8,
        params_wk=house_config.DAY1_PRIORS_WK,
        params_kt=house_config.DAY1_PRIORS_KT,
        base_t_out_c=16.5,
    )
    ok_warm, msg_warm = calibration.check_seasonal_calibration_gate(warm_cols)
    assert ok_warm is False
    assert "14.0" in msg_warm

    out_blocked, _ = calibration.calibrate_zone_a2(
        warm_cols, "wk", house_config.DAY1_PRIORS_WK, mode="auto_weekly"
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

    hist = evaluation.preprocess_history_columns(cols_mid_cool)
    for idx in range(c_start, c_end + 24 + 1):
        assert hist.cooling_or_tail[idx] is True
    assert hist.cooling_or_tail[c_end + 25] is False

    states, _, _, _ = evaluation.reconstruct_kalman_states(
        hist, "wk", house_config.DAY1_PRIORS_WK
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
        k: v for k, v in house_config.PARAM_BOUNDS_WK.items() if k != "h_up_w_k"
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
        house_config.DAY1_PRIORS_WK, h_loss_w_k=52.0, a_sol_m2=4.50
    )
    autumn_ts = int(
        datetime.datetime(
            2025, 10, 20, 0, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp()
    )
    autumn_cols = _generate_calibration_dataset(
        n_days=6,
        params_wk=true_wk,
        params_kt=house_config.DAY1_PRIORS_KT,
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
        params_kt=house_config.DAY1_PRIORS_KT,
        solar_enabled=False,
        start_ts_utc=winter_ts,
    )

    out_no_bank, _ = calibration.calibrate_zone_a2(
        dark_winter_cols,
        "wk",
        house_config.DAY1_PRIORS_WK,
        mode="manual",
        episode_bank=None,
    )
    assert math.isclose(
        out_no_bank.params_map_raw.a_sol_m2,
        house_config.DAY1_PRIORS_WK.a_sol_m2,
        abs_tol=0.05,
    )

    out_with_bank, _ = calibration.calibrate_zone_a2(
        dark_winter_cols,
        "wk",
        house_config.DAY1_PRIORS_WK,
        mode="manual",
        episode_bank=bank_restored,
    )
    assert abs(out_with_bank.params_map_raw.a_sol_m2 - 4.50) < 0.35


def test_holdout_rejection_gate_when_candidate_degrades() -> None:
    """Verifies holdout gate rejects candidates when unseen `J_eval` degrades."""
    true_wk = house_config.DAY1_PRIORS_WK
    cols = _generate_calibration_dataset(
        n_days=9, params_wk=true_wk, params_kt=house_config.DAY1_PRIORS_KT
    )

    n_total = len(cols["ts_utc"])
    train_end = int(0.70 * n_total)
    corrupted = {k: list(v) for k, v in cols.items()}
    for k in range(train_end):
        hr, mn = solar_weather.local_hour_minute(int(corrupted["ts_utc"][k]))
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
        house_config.DAY1_PRIORS_WK,
        c_floor_wh_k=4700.0,
        r_fh_k_kw=2.18,
        h_loss_w_k=51.5,
        a_sol_m2=4.10,
        k_hx_w_k=220.0,
    )
    true_kt = dataclasses.replace(
        house_config.DAY1_PRIORS_KT,
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
    assert learned_wk != house_config.DAY1_PRIORS_WK
    floor_after_cal = eng.get_zone_snapshot("wk").state.t_floor_c
    assert abs(floor_after_cal - floor_before_cal) > 0.02
    assert len(eng.hindcast_wk.queue) == evaluation.HORIZON_6H_STEPS

    eng.reset_parameters(cols)
    assert len(eng.snapshot_stack) == 2
    assert eng.get_zone_snapshot("wk").params == house_config.DAY1_PRIORS_WK
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
