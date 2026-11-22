"""Unit tests for Component 4 (`evaluation.py`) and `DigitalTwinEngine`.

Covers:
1. Bit-exact `LiveHindcastTracker` vs offline 72-step ZOH rollout, warmup
   countdown, `bridge_gap` (`<= 3h` vs `> 3h`), `reseed_from_history`, and JSON
   persistence.
2. Fair parameter-consistent `x_eval(k0)` & `x_prior(k0)` Kalman reconstruction
   (zero initial-state enthalpy jump, independent of stale `t_floor_*_c`).
3. Causal mid-window truncation (`truncate_on_unmodeled_event`, `>= 4.0h` clean)
   preserving afternoon heating runs while stripping `18:15` cooking spikes.
4. `kt` 1-hour split-AC hindcast (`H=12`, stride 2 + `>80 W` rising flank)
   sensitivity to `eta_ac`.
5. Atomic 31-column `telemetry_5m.csv` + `fit_report.json` export.
6. `DigitalTwinEngine` Layer-2 API (`process_5m_window`, `simulate_step`,
   `simulate_horizon`, `estimate_coast_duration`, `estimate_preheat_lead_time`,
   `export_solver_model`, `to_state_dict` / `load_state_dict`).
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import math
import pathlib
from typing import Any

from custom_components.betonbatterij import const
from custom_components.betonbatterij import house_config
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import engine
from custom_components.betonbatterij.domain import evaluation
from custom_components.betonbatterij.domain import hydraulics
from custom_components.betonbatterij.domain import solar_weather
from custom_components.betonbatterij.domain import zoh_kalman
from custom_components.betonbatterij.ha_bridge import storage_ring


def _generate_synthetic_columns(
    n_days: int,
    params_wk: contracts.ZoneParameters,
    params_kt: contracts.ZoneParameters,
    *,
    inject_evening_cooking_wk: bool = False,
    start_ts_utc: int | None = None,
) -> dict[str, list[float | int]]:
    """Generates physically exact multi-day 5-minute RingBuffer columns."""
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
    st_wk = contracts.ThermalState(t_air_c=19.8, t_floor_c=21.2, d_est_w=0.0)
    st_kt = contracts.ThermalState(t_air_c=18.8, t_floor_c=19.4, d_est_w=0.0)
    hm_ema: float | None = None
    interfloor_cache: dict[str, tuple[float, int]] = {}
    desk_state: dict[str, Any] = {}

    for k in range(n_steps):
        ts_k = start_ts_utc + k * const.DT_STEP_S
        day_idx = k // const.STEPS_PER_DAY
        hr_loc, mn_loc = solar_weather.local_hour_minute(ts_k)
        mins_loc = hr_loc * 60 + mn_loc

        fl = const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
        if day_idx == 3:
            fl &= ~const.FLAG_AANWEZIG  # Holiday day (`aanwezig == off`)

        # Diurnal winter outdoor temp (~0.8 °C night .. ~5.2 °C afternoon)
        t_out_raw = 3.0 - 2.2 * math.cos(
            2.0 * math.pi * (mins_loc - 210.0) / 1440.0
        )
        alpha_s, _ = solar_weather.solar_position(ts_k)

        is_sunny_day = (day_idx % 3) == 2
        if alpha_s > 2.0:
            peak_pv = 3900.0 if is_sunny_day else 180.0
            p_pv = min(
                peak_pv,
                peak_pv * max(0.0, math.sin(math.radians(alpha_s)) * 2.2),
            )
        else:
            p_pv = 0.0

        u_pump = 0.0
        u_bg = 0.0
        u_kt = 0.0
        f_hz = 0.0
        v_prim = 0.0
        t_top = 24.0
        t_bot_cal = 23.0
        p_ac = 0.0
        ac_mode = float(const.AC_MODE_OFF)

        if inject_evening_cooking_wk:
            # Afternoon heating run 13:00..16:30 coasting past 18:15 cooking
            if 780 <= mins_loc < 990 and not is_sunny_day:
                u_pump = 1.0
                u_bg = 1.0
                f_hz = 21.0
                v_prim = 14.5
                t_top = 33.0
                t_bot_cal = 29.0
        else:
            if not is_sunny_day and (day_idx % 2 == 0) and 450 <= mins_loc < 720:
                u_pump = 1.0
                u_bg = 1.0
                u_kt = 1.0
                f_hz = 21.0
                v_prim = 15.0
                t_top = 33.5
                t_bot_cal = 29.5
            elif not is_sunny_day and (day_idx % 2 == 1):
                if 450 <= mins_loc < 495:
                    ac_mode = float(const.AC_MODE_HEAT)
                    p_ac = 420.0
                elif 750 <= mins_loc < 990:
                    u_pump = 1.0
                    u_bg = 1.0
                    f_hz = 20.0
                    v_prim = 14.0
                    t_top = 32.5
                    t_bot_cal = 28.5

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

        # Optional unmodeled evening cooking spike at 18:15..19:00 (`1095..1140`)
        q_unmodeled_wk = 0.0
        if (
            inject_evening_cooking_wk
            and (fl & const.FLAG_AANWEZIG)
            and 1095 <= mins_loc < 1140
        ):
            q_unmodeled_wk = 950.0

        exog_wk_true = contracts.ExogenousStepInput(
            t_out_c=t_out_k,
            v_wind_ms=2.5,
            i_sol_wm2=i_wk,
            t_down_c=t_dn_wk,
            q_int_known_w=q_int_wk + q_unmodeled_wk,
            t_up_c=t_up_wk,
        )
        exog_kt_true = contracts.ExogenousStepInput(
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
                exog=exog_wk_true,
                q_fh_w=q_wk,
                q_ac_w=0.0,
                dt_s=float(const.DT_STEP_S),
            )
            st_kt = zoh_kalman.propagate_open_loop_step(
                state=st_kt,
                params=params_kt,
                exog=exog_kt_true,
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
        cols["p_p1_net_w"][k] = (750.0 if f_hz > 0.0 else 250.0) - p_pv
        cols["t_out_dak_c"][k] = t_out_raw
        cols["t_out_weather_c"][k] = t_out_raw
        cols["t_out_hm_c"][k] = t_out_raw
        cols["rh_out_pct"][k] = 82.0
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
        cols["t_floor_wk_c"][k] = st_wk.t_floor_c
        cols["q_fh_wk_w"][k] = q_wk
        cols["t_floor_kt_c"][k] = st_kt.t_floor_c
        cols["q_fh_kt_w"][k] = q_kt

    return cols


def test_live_hindcast_tracker_warmup_and_bit_exact_rollout() -> None:
    """Verifies `LiveHindcastTracker` matches offline 72-step ZOH bit-for-bit."""
    params = house_config.DAY1_PRIORS_WK
    tracker = evaluation.LiveHindcastTracker(
        horizon_steps=evaluation.HORIZON_6H_STEPS
    )

    n_steps = 110
    posteriors: list[contracts.ThermalState] = []
    exogs: list[contracts.ExogenousStepInput] = []
    q_fhs: list[float] = []

    st = contracts.ThermalState(t_air_c=19.5, t_floor_c=21.0, d_est_w=120.0)
    for k in range(n_steps):
        ex = contracts.ExogenousStepInput(
            t_out_c=2.0 + 3.0 * math.sin(k * 0.07),
            v_wind_ms=3.5,
            i_sol_wm2=180.0 if 30 <= k <= 65 else 0.0,
            t_down_c=10.5,
            q_int_known_w=65.0,
            t_up_c=18.0,
        )
        q_fh = 2400.0 if (10 <= k < 35) else 0.0
        if k > 0:
            st = zoh_kalman.propagate_open_loop_step(
                state=st,
                params=params,
                exog=ex,
                q_fh_w=q_fh,
                q_ac_w=0.0,
                dt_s=300.0,
            )
            st = contracts.ThermalState(
                t_air_c=st.t_air_c + 0.01 * math.cos(k),
                t_floor_c=st.t_floor_c + 0.003 * math.sin(k),
                d_est_w=st.d_est_w + 5.0 * math.sin(k * 0.3),
            )
        posteriors.append(st)
        exogs.append(ex)
        q_fhs.append(q_fh)

    for k in range(n_steps):
        pub_c, warmup_min = tracker.step(
            posteriors[k], params, exogs[k], q_fhs[k], 0.0, dt_s=300.0
        )
        if k < evaluation.HORIZON_6H_STEPS:
            assert pub_c is None
            assert warmup_min == (evaluation.HORIZON_6H_STEPS - k) * 5
        else:
            assert warmup_min == 0
            assert pub_c is not None
            assert len(tracker.queue) == evaluation.HORIZON_6H_STEPS

            k0 = k - evaluation.HORIZON_6H_STEPS
            st_off = posteriors[k0]
            for s in range(1, evaluation.HORIZON_6H_STEPS + 1):
                st_off = zoh_kalman.propagate_open_loop_step(
                    state=st_off,
                    params=params,
                    exog=exogs[k0 + s],
                    q_fh_w=q_fhs[k0 + s],
                    q_ac_w=0.0,
                    dt_s=300.0,
                )

            assert tracker.last_published_exact_c == st_off.t_air_c
            assert pub_c == round(st_off.t_air_c, 2)


def test_live_hindcast_tracker_bridge_gap_reseed_and_persistence() -> None:
    """Verifies `to_dict`/`from_dict`, `bridge_gap`, and `reseed_from_history`."""
    params = house_config.DAY1_PRIORS_WK
    tracker = evaluation.LiveHindcastTracker()
    ex = contracts.ExogenousStepInput(
        t_out_c=4.0,
        v_wind_ms=2.0,
        i_sol_wm2=0.0,
        t_down_c=10.5,
        q_int_known_w=50.0,
        t_up_c=18.0,
    )
    st = contracts.ThermalState(t_air_c=20.0, t_floor_c=21.5, d_est_w=0.0)

    hist_records: list[dict[str, Any]] = []
    pub: float | None = None
    rem = 0
    for _ in range(75):
        st = zoh_kalman.propagate_open_loop_step(
            state=st,
            params=params,
            exog=ex,
            q_fh_w=0.0,
            q_ac_w=0.0,
            dt_s=300.0,
        )
        pub, rem = tracker.step(st, params, ex, 0.0, 0.0, 300.0)
        hist_records.append(
            {
                "posterior_state": st,
                "exog": ex,
                "q_fh_w": 0.0,
                "q_ac_w": 0.0,
                "dt_s": 300.0,
            }
        )

    assert pub is not None and rem == 0

    restored = evaluation.LiveHindcastTracker.from_dict(tracker.to_dict())
    assert len(restored.queue) == evaluation.HORIZON_6H_STEPS
    assert restored.last_published_c == tracker.last_published_c

    # Short restart gap (15 min <= 3h) preserves the 6h comparison line
    restored.bridge_gap(900.0, params, ex, q_fh_avg_w=0.0)
    next_pub, next_rem = restored.step(st, params, ex, 0.0, 0.0, 300.0)
    assert next_pub is not None and next_rem == 0

    # Long outage (> 3h = 14400s) clears the queue safely
    restored.bridge_gap(14400.0, params, ex, q_fh_avg_w=0.0)
    assert len(restored.queue) == 0
    assert restored.last_published_c is None

    # Reseed from 75-step history under updated parameters restores output
    params_new = dataclasses.replace(params, h_loss_w_k=85.0)
    reseeded_val = restored.reseed_from_history(hist_records, params_new)
    assert reseeded_val is not None
    assert len(restored.queue) == evaluation.HORIZON_6H_STEPS
    assert reseeded_val < pub


def test_fair_x_prior_reconstruction_zero_enthalpy_jump() -> None:
    """Verifies `evaluate_14d_hindcast` reconstructs `x_prior(k0)` cleanly."""
    true_wk = dataclasses.replace(
        house_config.DAY1_PRIORS_WK,
        c_floor_wh_k=4600.0,
        r_fh_k_kw=2.15,
        h_loss_w_k=49.0,
        a_sol_m2=4.10,
        k_hx_w_k=218.0,
    )
    cols = _generate_synthetic_columns(
        n_days=7, params_wk=true_wk, params_kt=house_config.DAY1_PRIORS_KT
    )

    rep_clean = evaluation.evaluate_14d_hindcast(
        cols, "wk", true_wk, house_config.DAY1_PRIORS_WK
    )
    assert rep_clean.overall_6h.n_windows > 20
    assert rep_clean.overall_6h.rmse_end_c is not None
    assert rep_clean.overall_6h.prior_rmse_end_c is not None
    assert rep_clean.overall_6h.rmse_end_c < 0.06
    assert (
        rep_clean.overall_6h.prior_rmse_end_c > rep_clean.overall_6h.rmse_end_c
    )
    assert (rep_clean.overall_6h.skill_score_pct or 0.0) > 50.0

    # Poison `t_floor_wk_c` with +8.0 °C offset: evaluation must be 100% immune
    cols_poisoned = {k: list(v) for k, v in cols.items()}
    cols_poisoned["t_floor_wk_c"] = [
        float(v) + 8.0 for v in cols["t_floor_wk_c"]
    ]
    rep_poisoned = evaluation.evaluate_14d_hindcast(
        cols_poisoned, "wk", true_wk, house_config.DAY1_PRIORS_WK
    )

    assert math.isclose(
        rep_clean.overall_6h.rmse_end_c,
        rep_poisoned.overall_6h.rmse_end_c or 0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        rep_clean.overall_6h.prior_rmse_end_c,
        rep_poisoned.overall_6h.prior_rmse_end_c or 0.0,
        abs_tol=1e-9,
    )

    # Short history < 48h gracefully returns empty metrics without error
    short_cols = {k: v[:400] for k, v in cols.items()}
    rep_short = evaluation.evaluate_14d_hindcast(
        short_cols, "wk", true_wk, house_config.DAY1_PRIORS_WK
    )
    assert rep_short.overall_6h.n_windows == 0
    assert rep_short.overall_6h.rmse_end_c is None


def test_causal_mid_window_truncation_on_evening_cooking_spike() -> None:
    """Verifies afternoon heating windows truncate causally before cooking."""
    true_wk = house_config.DAY1_PRIORS_WK
    cols_cook = _generate_synthetic_columns(
        n_days=6,
        params_wk=true_wk,
        params_kt=house_config.DAY1_PRIORS_KT,
        inject_evening_cooking_wk=True,
    )
    cols_clean_ref = _generate_synthetic_columns(
        n_days=6,
        params_wk=true_wk,
        params_kt=house_config.DAY1_PRIORS_KT,
        inject_evening_cooking_wk=False,
    )

    rep = evaluation.evaluate_14d_hindcast(
        cols_cook,
        "wk",
        true_wk,
        dataclasses.replace(true_wk, h_loss_w_k=95.0),
    )
    heat_metrics = rep.by_regime["heating_and_coast"]

    assert heat_metrics.n_windows >= 2
    assert heat_metrics.rmse_end_c is not None
    assert heat_metrics.rmse_end_c < 0.24

    cols_win = {k: list(v) for k, v in cols_clean_ref.items()}
    cols_win["flags"][900] = (
        int(cols_win["flags"][900]) | const.FLAG_OPEN_WIN_WK
    )
    rep_win = evaluation.evaluate_14d_hindcast(
        cols_win, "wk", true_wk, house_config.DAY1_PRIORS_WK
    )
    assert rep_win.overall_6h.n_windows >= 10


def test_airco_heating_1h_hindcast_sensitivity_and_flank_trigger() -> None:
    """Verifies `kt` 1h AC hindcast (`H=12`) captures `eta_ac` sensitivity."""
    true_kt = dataclasses.replace(house_config.DAY1_PRIORS_KT, eta_ac=4.35)
    wrong_kt = dataclasses.replace(house_config.DAY1_PRIORS_KT, eta_ac=2.40)

    cols = _generate_synthetic_columns(
        n_days=6,
        params_wk=house_config.DAY1_PRIORS_WK,
        params_kt=true_kt,
    )

    rep_true = evaluation.evaluate_14d_hindcast(cols, "kt", true_kt, wrong_kt)
    rep_wrong = evaluation.evaluate_14d_hindcast(
        cols, "kt", wrong_kt, house_config.DAY1_PRIORS_KT
    )

    ac_true = rep_true.by_regime["airco_heating_1h"]
    ac_wrong = rep_wrong.by_regime["airco_heating_1h"]

    assert ac_true.horizon_steps == evaluation.HORIZON_AC_1H_STEPS == 12
    assert ac_true.n_windows >= 2
    assert ac_true.rmse_end_c is not None and ac_wrong.rmse_end_c is not None
    assert ac_true.rmse_end_c < 0.08
    assert ac_wrong.rmse_end_c > 0.25
    assert (ac_wrong.bias_end_c or 0.0) < -0.20
    assert (ac_true.skill_score_pct or 0.0) > 60.0


def test_atomic_export_csv_31_columns_and_fit_report_json(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies atomic export writes 31 CSV columns + `fit_report.json`."""
    cols = _generate_synthetic_columns(
        n_days=3,
        params_wk=house_config.DAY1_PRIORS_WK,
        params_kt=house_config.DAY1_PRIORS_KT,
    )
    eng = engine.DigitalTwinEngine()
    eng.replay_48h_from_columns(cols)

    csv_path, json_path = eng.export_diagnostics(tmp_path, cols)
    assert pathlib.Path(csv_path).is_file()
    assert pathlib.Path(json_path).is_file()
    assert not (tmp_path / "telemetry_5m.csv.tmp").exists()
    assert not (tmp_path / "fit_report.json.tmp").exists()

    with open(csv_path, encoding="utf-8", newline="") as f_csv:
        rows = list(csv.reader(f_csv))

    assert len(rows[0]) == 31
    assert tuple(rows[0]) == evaluation.CSV_EXPORT_COLUMNS
    assert "t_surf_wk_c" in rows[0]
    assert "t_surf_kt_c" in rows[0]
    assert "p_wp_raw_el_w" in rows[0]
    assert "f_comp_hz" in rows[0]
    assert len(rows) == 1 + len(cols["ts_utc"])

    with open(json_path, encoding="utf-8") as f_js:
        report = json.load(f_js)

    assert report["ringbuffer"]["n_rows"] == len(cols["ts_utc"])
    assert "wk" in report["zones"] and "kt" in report["zones"]
    assert report["zones"]["wk"]["evaluation_14d"] is not None

    csv2, json2 = evaluation.export_csv_and_report(
        tmp_path / "sub", {k: v[:20] for k, v in cols.items()}, report
    )
    assert pathlib.Path(csv2).is_file() and pathlib.Path(json2).is_file()


def test_digital_twin_engine_layer2_api_methods() -> None:
    """Verifies `DigitalTwinEngine` online pipeline, coast/preheat, & solver."""
    eng = engine.DigitalTwinEngine()

    base_ts = int(
        datetime.datetime(
            2026, 1, 15, 6, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp()
    )
    out: dict[str, float] = {}
    for step_i in range(80):
        out = eng.process_5m_window(
            {
                "ts_utc": base_ts + step_i * 300,
                "flags": (
                    const.FLAG_VALID_WK
                    | const.FLAG_VALID_KT
                    | const.FLAG_AANWEZIG
                ),
                "t_out_dak_c": 2.0,
                "v_wind_ms": 3.0,
                "p_pv_w": 0.0,
                "t_air_wk_c": 19.6 + 0.005 * step_i,
                "t_air_kantoor_c": 18.8,
                "u_pump_sec": 1.0 if step_i < 40 else 0.0,
                "u_eff_bg": 1.0 if step_i < 40 else 0.0,
                "v_prim_l_min": 14.0 if step_i < 40 else 0.0,
                "p_prim_th_w": 3000.0 if step_i < 40 else 0.0,
                "f_comp_hz": 20.0 if step_i < 40 else 0.0,
                "t_buf_top_start_c": 31.0,
                "t_buf_top_end_c": 31.0,
                "t_buf_bot_start_c": 28.0,
                "t_buf_bot_end_c": 28.0,
            }
        )

    assert set(out.keys()) == {
        "t_floor_wk_c",
        "d_est_wk_w",
        "q_fh_wk_w",
        "t_hind6h_wk_c",
        "t_floor_kt_c",
        "d_est_kt_w",
        "q_fh_kt_w",
        "t_hind6h_kt_c",
    }
    assert not math.isnan(out["t_hind6h_wk_c"])

    snap_wk = eng.get_zone_snapshot("wk")
    assert snap_wk.t_hindcast_6h_c is not None

    forecast_6h = [
        contracts.ExogenousStepInput(
            t_out_c=4.0,
            v_wind_ms=2.5,
            i_sol_wm2=0.0,
            t_down_c=10.5,
            q_int_known_w=50.0,
            t_up_c=18.5,
        )
        for _ in range(72)
    ]
    coast_charged = eng.estimate_coast_duration(
        "wk",
        floor_temp_min_air_c=19.5,
        exog_forecast=forecast_6h,
        initial_state=contracts.ThermalState(
            t_air_c=19.8, t_floor_c=25.5, d_est_w=0.0
        ),
    )
    assert coast_charged.peak_t_air_overshoot_c > 19.95
    assert coast_charged.coast_duration_min == 360

    coast_cooking_illusion = eng.estimate_coast_duration(
        "wk",
        floor_temp_min_air_c=19.5,
        exog_forecast=forecast_6h,
        initial_state=contracts.ThermalState(
            t_air_c=20.1, t_floor_c=19.0, d_est_w=750.0
        ),
    )
    assert (
        coast_cooking_illusion.coast_duration_min
        < coast_charged.coast_duration_min
    )

    fc_kt = [
        contracts.ExogenousStepInput(
            t_out_c=3.0,
            v_wind_ms=2.5,
            i_sol_wm2=0.0,
            t_down_c=18.0,
            q_int_known_w=0.0,
            t_up_c=0.0,
        )
        for _ in range(72)
    ]
    st_cold_kt = contracts.ThermalState(
        t_air_c=17.5, t_floor_c=18.0, d_est_w=0.0
    )
    pre_floor = eng.estimate_preheat_lead_time(
        "kt",
        target_temp_c=19.5,
        exog_forecast=fc_kt,
        t_supply_sec_c=35.0,
        p_el_ac_w=0.0,
        initial_state=st_cold_kt,
    )
    pre_hybrid = eng.estimate_preheat_lead_time(
        "kt",
        target_temp_c=19.5,
        exog_forecast=fc_kt,
        t_supply_sec_c=35.0,
        p_el_ac_w=550.0,
        initial_state=st_cold_kt,
    )
    assert pre_hybrid.reachable is True
    assert pre_hybrid.lead_time_min % 5 == 0
    assert pre_hybrid.lead_time_min < pre_floor.lead_time_min
    assert pre_hybrid.predicted_t_air_end_c >= 19.5

    # Side-effect-free check: snapshot must remain identical after simulations
    assert eng.get_zone_snapshot("wk") == snap_wk

    lin_wk = eng.export_solver_model("wk", dt_s=1800.0)
    lin_kt = eng.export_solver_model("kt", dt_s=1800.0)
    assert lin_wk.gamma_fh[1] > lin_wk.gamma_fh[0] > 0.0
    assert lin_wk.gamma_ac_el_ref == (0.0, 0.0)
    assert lin_kt.gamma_ac_el_ref[0] > 0.0
    assert 15.0 < lin_wk.effective_tau_coast_h < 80.0
    assert lin_wk.steady_loss_total_w_k > house_config.DAY1_PRIORS_WK.h_loss_w_k

    ex_zero = contracts.ExogenousStepInput(
        t_out_c=0.0,
        v_wind_ms=2.0,
        i_sol_wm2=0.0,
        t_down_c=0.0,
        q_int_known_w=0.0,
        t_up_c=0.0,
    )
    s0 = eng.simulate_step(
        "wk",
        contracts.ThermalState(20.0, 20.0, 0.0),
        ex_zero,
        contracts.ActuatorStepInput(q_fh_w=0.0),
        dt_s=1800.0,
    )
    s1 = eng.simulate_step(
        "wk",
        contracts.ThermalState(20.0, 20.0, 0.0),
        ex_zero,
        contracts.ActuatorStepInput(q_fh_w=1000.0),
        dt_s=1800.0,
    )
    assert math.isclose(
        (s1.state.t_air_c - s0.state.t_air_c) / 1000.0,
        lin_wk.gamma_fh[0],
        rel_tol=1e-9,
    )
    assert math.isclose(
        (s1.state.t_floor_c - s0.state.t_floor_c) / 1000.0,
        lin_wk.gamma_fh[1],
        rel_tol=1e-9,
    )

    sink_w = eng.estimate_total_sink_capacity_w(
        32.0, ["wk", "gang", "kantoor"]
    )
    assert sink_w > 1500.0
    assert 2.5 < eng.predict_cop_cv(7.0, 28.0, 80.0) < 6.5
    assert 1.0 <= eng.predict_cop_dhw(5.0, 40.0, 55.0, 85.0) < 5.0

    eng.update_flow_rates({"wk": 5.5})
    saved = eng.to_state_dict()
    eng2 = engine.DigitalTwinEngine()
    eng2.load_state_dict(saved)
    assert math.isclose(
        eng2.get_zone_snapshot("wk").state.t_floor_c,
        snap_wk.state.t_floor_c,
        abs_tol=1e-9,
    )


def test_ringbuffer_float32_roundtrip_legacy_lts_detection_and_flat_bridging() -> None:
    """Verifies float32-safe legacy v0.1.0 LTS detection & multi-hour flat bridge."""
    rb = storage_ring.ColumnarRingBuffer(chunk_rows=128, num_chunks=2)
    t0 = int(
        datetime.datetime(
            2026, 1, 14, 0, 0, 0, tzinfo=datetime.timezone.utc
        ).timestamp()
    )

    # 4 legacy LTS hours (0..48): hour 0 linear with ~1200 W P1, hours 1 & 2
    # flat sandwiched, hour 3 linear; followed by 24 rows of real 5m Recorder.
    n_lts = 49
    n_rec = 24
    n_tot = n_lts + n_rec

    def _interp_4h(
        v0: float, v1: float, v2: float, v3: float, v4: float
    ) -> list[float]:
        out = [0.0] * n_lts
        knots = (v0, v1, v2, v3, v4)
        for h in range(4):
            a, b = knots[h], knots[h + 1]
            for j in range(13):
                out[h * 12 + j] = a + (b - a) * (j / 12.0)
        return out

    t_dak_lts = _interp_4h(2.1, 2.7, 2.7, 2.7, 3.3)
    t_hm_lts = _interp_4h(2.3, 2.9, 2.9, 2.9, 3.5)
    t_wk_lts = _interp_4h(19.7, 19.4, 19.4, 19.4, 19.1)
    t_kt_lts = _interp_4h(18.9, 18.6, 18.6, 18.6, 18.4)
    p1_lts = _interp_4h(950.0, 1450.0, 1450.0, 1450.0, 1080.0)

    upstairs_zero = {
        c: 0.0
        for c in (
            "u_eff_slaapkamer",
            "u_eff_badkamer",
            "u_eff_werkkamer_1e",
            "u_eff_kamer_1e_midden",
            "u_eff_kamer_1e_zuid",
            "u_eff_badkamer_boven",
            "u_eff_logeerkamer",
        )
    }

    for k in range(n_tot):
        row: dict[str, float | int] = {
            "ts_utc": t0 + k * const.DT_STEP_S,
            "flags": const.FLAG_VALID_WK | const.FLAG_VALID_KT,
            "p_prim_th_w": 2400.0 if k >= 12 else 0.0,
            "t_wp_out_c": 29.0,
            "t_wp_in_c": 26.0,
            **upstairs_zero,
        }
        if k < n_lts:
            row["t_out_weather_c"] = math.nan
            row["t_out_dak_c"] = t_dak_lts[k]
            row["t_out_hm_c"] = t_hm_lts[k]
            row["t_air_wk_c"] = t_wk_lts[k]
            row["t_air_kantoor_c"] = t_kt_lts[k]
            row["p_p1_net_w"] = p1_lts[k]
        else:
            # Real 5-min Recorder measurements with nonlinear fluctuations
            dt_i = k - n_lts
            row["t_out_weather_c"] = 3.4 + 0.12 * math.sin(0.9 * dt_i)
            row["t_out_dak_c"] = 3.3 + 0.15 * math.cos(0.7 * dt_i)
            row["t_out_hm_c"] = 3.5 + 0.10 * math.sin(0.5 * dt_i)
            row["t_air_wk_c"] = 19.1 + 0.08 * math.sin(0.6 * dt_i)
            row["t_air_kantoor_c"] = 18.4 + 0.06 * math.cos(0.8 * dt_i)
            row["p_p1_net_w"] = 1150.0 + 180.0 * math.sin(1.1 * dt_i)
        rb.append_row(row)

    cols_f32 = rb.get_columns()
    is_lts, miss_prim = evaluation.detect_lts_and_missing_prim_rows(cols_f32)
    assert all(is_lts[:n_lts])
    assert all(miss_prim[:n_lts])
    assert not any(is_lts[n_lts:])
    assert not any(miss_prim[n_lts:])


def test_compute_train_holdout_split_80d_lts_plus_10d_raw() -> None:
    """Verifies F1 raw-aware split on 80d LTS + 10d raw (and Cases A & C)."""
    n_lts = 80 * const.STEPS_PER_DAY
    n_raw = 10 * const.STEPS_PER_DAY
    n_tot = n_lts + n_raw
    t0 = 1_760_000_000
    cols: dict[str, list[float | int]] = {
        "ts_utc": [t0 + k * const.DT_STEP_S for k in range(n_tot)],
        "flags": [const.FLAG_LTS_BACKFILL] * n_lts + [0] * n_raw,
    }
    train_end = evaluation.compute_train_holdout_split(cols)
    # 65% of 10d raw (= 6.5d = 1872 rows) in train, 35% (= 3.5d = 1008) holdout
    assert train_end == n_tot - int(round(0.35 * n_raw))
    assert train_end == n_lts + (n_raw - 1008)
    assert sum(1 for f in cols["flags"][:train_end] if f == 0) == 1872
    assert all(f == 0 for f in cols["flags"][train_end:])

    # Case A: >= 28d raw -> standard 14-day holdout
    n_35d = 35 * const.STEPS_PER_DAY
    cols_a = {
        "ts_utc": [t0 + k * const.DT_STEP_S for k in range(n_35d)],
        "flags": [0] * n_35d,
    }
    assert (
        evaluation.compute_train_holdout_split(cols_a)
        == n_35d - evaluation.EVAL_14D_STEPS
    )

    # Case C: < 4d raw (7d LTS + 3d raw = 10d) -> 70/30 fallback
    n_7d = 7 * const.STEPS_PER_DAY
    n_3d = 3 * const.STEPS_PER_DAY
    cols_c = {
        "ts_utc": [t0 + k * const.DT_STEP_S for k in range(n_7d + n_3d)],
        "flags": [const.FLAG_LTS_BACKFILL] * n_7d + [0] * n_3d,
    }
    assert evaluation.compute_train_holdout_split(cols_c) == int(
        0.70 * (n_7d + n_3d)
    )


def test_simulate_step_gamma_cal_f5a_without_scaling_solver_gamma_fh() -> None:
    """Verifies F5a (`gamma_cal` in `simulate_step`) & untouched `gamma_fh`."""
    eng = engine.DigitalTwinEngine()
    st0 = contracts.ThermalState(t_air_c=19.5, t_floor_c=21.0, d_est_w=0.0)
    ex = contracts.ExogenousStepInput(t_out_c=3.0)

    eng.hydraulics.gamma_cal = 1.0
    lin_1 = eng.export_solver_model("wk", dt_s=1800.0)
    res_sup_1 = eng.simulate_step(
        "wk",
        st0,
        ex,
        contracts.ActuatorStepInput(t_supply_sec_c=31.0, u_eff=1.0),
    )
    res_qfh_1 = eng.simulate_step(
        "wk", st0, ex, contracts.ActuatorStepInput(q_fh_w=1500.0)
    )

    eng.hydraulics.gamma_cal = 1.20
    lin_12 = eng.export_solver_model("wk", dt_s=1800.0)
    res_sup_12 = eng.simulate_step(
        "wk",
        st0,
        ex,
        contracts.ActuatorStepInput(t_supply_sec_c=31.0, u_eff=1.0),
    )
    res_qfh_12 = eng.simulate_step(
        "wk", st0, ex, contracts.ActuatorStepInput(q_fh_w=1500.0)
    )

    # F5a scales t_supply_sec_c power by gamma_cal; explicit q_fh_w unchanged
    assert math.isclose(
        res_sup_12.q_fh_applied_w,
        1.20 * res_sup_1.q_fh_applied_w,
        rel_tol=1e-9,
    )
    assert res_qfh_12.q_fh_applied_w == res_qfh_1.q_fh_applied_w == 1500.0
    # NOT F5b: linearized solver gamma_fh [K/W] must remain identical
    assert lin_12.gamma_fh == lin_1.gamma_fh
    assert lin_12.phi_2x2 == lin_1.phi_2x2

    # Public pure wrapper compute_14d_evaluation_reports
    cols = _generate_synthetic_columns(
        n_days=4,
        params_wk=house_config.DAY1_PRIORS_WK,
        params_kt=house_config.DAY1_PRIORS_KT,
    )
    ctx = eng.capture_calibration_context()
    rep_wk, rep_kt = eng.compute_14d_evaluation_reports(cols, ctx)
    assert rep_wk.zone == "wk" and rep_kt.zone == "kt"
    assert rep_wk.overall_6h.n_windows > 0
    assert eng.eval_report_wk is None  # Pure: does not mutate engine state
