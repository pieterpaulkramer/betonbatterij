"""Offline verification suite for Component 2 ZOH discretization and Kalman.

Covers 65-digit `decimal.Decimal` 6x6 Van Loan reference verification,
semigroup consistency (`36 * 300s == 10800s`), adiabatic enthalpy conservation,
change-conditioned `0.01 °C` / `0.1 °C` resolution detection, Tobit deadband,
`+900 W` cooking spike isolation, `K_floor == 0` (`0.0000 °C` false slab pull)
open-window gate, nocturnal anchoring, 48h replay, and restart gap bridging.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal

import pytest

from custom_components.betonbatterij import house_config
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import solar_weather
from custom_components.betonbatterij.domain import zoh_kalman


def _utc_ts(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int = 0,
) -> int:
    """Creates a UTC Unix timestamp in seconds."""
    return int(
        datetime.datetime(
            year, month, day, hour, minute, tzinfo=datetime.timezone.utc
        ).timestamp()
    )


def _to_dec(val: float | int) -> decimal.Decimal:
    """Converts a numeric literal or float into an exact `Decimal`."""
    return decimal.Decimal(repr(val))


def _matmul6_dec(
    x: list[list[decimal.Decimal]],
    y: list[list[decimal.Decimal]],
) -> list[list[decimal.Decimal]]:
    """Multiplies two `6x6` `Decimal` matrices."""
    return [
        [sum(x[i][k] * y[k][j] for k in range(6)) for j in range(6)]
        for i in range(6)
    ]


def _decimal_reference_zoh(
    params: contracts.ZoneParameters,
    dt_s: float,
    sigma_d_w: float,
    u_pump: float,
    q_fh_w: float,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Computes an independent 65-digit `Decimal` 6x6 Van Loan reference."""
    with decimal.localcontext() as ctx:
        ctx.prec = 65
        zero = decimal.Decimal(0)
        one = decimal.Decimal(1)

        c_air = _to_dec(3600.0) * _to_dec(params.c_air_wh_k)
        c_floor = _to_dec(3600.0) * _to_dec(params.c_floor_wh_k)
        h_fh = _to_dec(1000.0) / _to_dec(params.r_fh_k_kw)
        tau_d = _to_dec(3600.0) * _to_dec(params.tau_d_h)
        h_oa = _to_dec(params.h_loss_w_k)
        h_up = _to_dec(params.h_up_w_k)
        h_down = _to_dec(params.h_down_w_k)
        dt = _to_dec(dt_s)

        ac = [
            [-(h_fh + h_oa + h_up) / c_air, h_fh / c_air, one / c_air],
            [h_fh / c_floor, -(h_fh + h_down) / c_floor, zero],
            [zero, zero, -one / tau_d],
        ]
        bc = [
            [
                h_oa / c_air,
                h_up / c_air,
                zero,
                (_to_dec(params.alpha_sol) * _to_dec(params.a_sol_m2)) / c_air,
                zero,
                one / c_air,
                one / c_air,
            ],
            [
                zero,
                zero,
                h_down / c_floor,
                ((one - _to_dec(params.alpha_sol)) * _to_dec(params.a_sol_m2))
                / c_floor,
                one / c_floor,
                zero,
                zero,
            ],
            [zero] * 7,
        ]

        qc00 = _to_dec(zoh_kalman.Q_AIR_BASE_K2_S)
        qc11 = (
            _to_dec(zoh_kalman.Q_FLOOR_IDLE_K2_S)
            + _to_dec(u_pump)
            * (
                _to_dec(zoh_kalman.Q_FLOOR_PUMP_K2_S)
                - _to_dec(zoh_kalman.Q_FLOOR_IDLE_K2_S)
            )
            + ((_to_dec(0.08) * abs(_to_dec(q_fh_w))) ** 2)
            * _to_dec(zoh_kalman.TAU_HYDR_S)
            / (c_floor * c_floor)
        )
        qc22 = (
            decimal.Decimal(2) * _to_dec(sigma_d_w) * _to_dec(sigma_d_w)
        ) / tau_d
        qc = [
            [qc00, zero, zero],
            [zero, qc11, zero],
            [zero, zero, qc22],
        ]

        # 6x6 Van Loan matrix M = [[-A_c, Q_c], [0, A_c^T]] * dt.
        m6 = [[zero] * 6 for _ in range(6)]
        for i in range(3):
            for j in range(3):
                m6[i][j] = -ac[i][j] * dt
                m6[i][j + 3] = qc[i][j] * dt
                m6[i + 3][j + 3] = ac[j][i] * dt

        exp6 = [[one if i == j else zero for j in range(6)] for i in range(6)]
        term6 = [[one if i == j else zero for j in range(6)] for i in range(6)]
        for k in range(1, 50):
            term6 = _matmul6_dec(term6, m6)
            inv_k = one / decimal.Decimal(k)
            term6 = [[term6[i][j] * inv_k for j in range(6)] for i in range(6)]
            exp6 = [
                [exp6[i][j] + term6[i][j] for j in range(6)] for i in range(6)
            ]

        ad_dec = [[exp6[j + 3][i + 3] for j in range(3)] for i in range(3)]
        ur_dec = [[exp6[i][j + 3] for j in range(3)] for i in range(3)]
        qd_dec = [
            [sum(ad_dec[i][k] * ur_dec[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]

        psi_dec = [[dt if i == j else zero for j in range(3)] for i in range(3)]
        t3 = [[one if i == j else zero for j in range(3)] for i in range(3)]
        ac_dt = [[ac[i][j] * dt for j in range(3)] for i in range(3)]
        for k in range(1, 50):
            inv_k = one / decimal.Decimal(k)
            t3 = [
                [
                    sum(t3[i][m] * ac_dt[m][j] for m in range(3)) * inv_k
                    for j in range(3)
                ]
                for i in range(3)
            ]
            w_psi = dt / decimal.Decimal(k + 1)
            psi_dec = [
                [psi_dec[i][j] + w_psi * t3[i][j] for j in range(3)]
                for i in range(3)
            ]

        bd_dec = [
            [sum(psi_dec[i][k] * bc[k][j] for k in range(3)) for j in range(7)]
            for i in range(3)
        ]

        return (
            [[float(ad_dec[i][j]) for j in range(3)] for i in range(3)],
            [[float(bd_dec[i][j]) for j in range(7)] for i in range(3)],
            [[float(qd_dec[i][j]) for j in range(3)] for i in range(3)],
        )


class TestZohDiscretizationAndPhysics:
    """Verifies machine precision, semigroup exactness, and energy balance."""

    @pytest.mark.parametrize("dt_s", [300.0, 10800.0])
    def test_machine_precision_against_60_digit_decimal_van_loan(
        self,
        dt_s: float,
    ) -> None:
        priors = house_config.DAY1_PRIORS_WK
        ad, bd, qd = zoh_kalman.discretize_zoh_3x3(
            priors,
            dt_s=dt_s,
            v_wind_ms=2.5,
            sigma_d_w=110.0,
            u_pump=1.0,
            q_fh_w=1850.0,
        )
        ad_ref, bd_ref, qd_ref = _decimal_reference_zoh(
            priors,
            dt_s=dt_s,
            sigma_d_w=110.0,
            u_pump=1.0,
            q_fh_w=1850.0,
        )

        for i in range(3):
            for j in range(3):
                assert abs(ad[i][j] - ad_ref[i][j]) < 1e-14
                assert qd[i][j] == qd[j][i]
                denom = max(1e-20, abs(qd_ref[i][j]))
                assert abs(qd[i][j] - qd_ref[i][j]) / denom < 1e-13

        for i in range(3):
            for j in range(7):
                assert abs(bd[i][j] - bd_ref[i][j]) < 1e-14

    def test_semigroup_consistency_36_steps_vs_3_hours(self) -> None:
        """36 steps of 300 s match 1 step of 10,800 s for state and `Q_d`."""
        priors = house_config.DAY1_PRIORS_WK
        exog = contracts.ExogenousStepInput(
            t_out_c=3.0,
            v_wind_ms=2.0,
            i_sol_wm2=180.0,
            t_down_c=10.5,
            q_int_known_w=75.0,
            t_up_c=17.5,
        )
        x0 = contracts.ThermalState(
            t_air_c=19.2, t_floor_c=21.8, d_est_w=140.0
        )

        x_roll = x0
        for _ in range(36):
            x_roll = zoh_kalman.propagate_open_loop_step(
                x_roll, priors, exog, q_fh_w=1600.0, dt_s=300.0
            )
        x_single = zoh_kalman.propagate_open_loop_step(
            x0, priors, exog, q_fh_w=1600.0, dt_s=10800.0
        )

        assert x_roll.t_air_c == pytest.approx(x_single.t_air_c, abs=1e-12)
        assert x_roll.t_floor_c == pytest.approx(x_single.t_floor_c, abs=1e-12)
        assert x_roll.d_est_w == pytest.approx(x_single.d_est_w, abs=1e-11)

        ad_300, _, qd_300 = zoh_kalman.discretize_zoh_3x3(
            priors, dt_s=300.0, sigma_d_w=110.0
        )
        _, _, qd_3h = zoh_kalman.discretize_zoh_3x3(
            priors, dt_s=10800.0, sigma_d_w=110.0
        )

        p_acc = [[0.0, 0.0, 0.0] for _ in range(3)]
        for _ in range(36):
            ap = [
                [sum(ad_300[i][k] * p_acc[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)
            ]
            p_acc = [
                [
                    sum(ap[i][k] * ad_300[j][k] for k in range(3)) + qd_300[i][j]
                    for j in range(3)
                ]
                for i in range(3)
            ]
        for i in range(3):
            for j in range(3):
                assert p_acc[i][j] == pytest.approx(qd_3h[i][j], rel=1e-12)

    def test_exact_energy_conservation_for_adiabatic_envelope(self) -> None:
        """With `H_loss = H_up = H_down = 0`, enthalpy equals integrated power."""
        adiabatic = dataclasses.replace(
            house_config.DAY1_PRIORS_WK,
            h_loss_w_k=0.0,
            h_up_w_k=0.0,
            h_down_w_k=0.0,
        )
        state = contracts.ThermalState(
            t_air_c=19.0, t_floor_c=19.0, d_est_w=0.0
        )
        e0_wh = (
            adiabatic.c_air_wh_k * state.t_air_c
            + adiabatic.c_floor_wh_k * state.t_floor_c
        )

        injected_wh = 0.0
        for step in range(144):
            q_fh = 1500.0 if step < 72 else 0.0
            i_sol = 200.0 if 36 <= step < 96 else 0.0
            q_int = 60.0
            exog = contracts.ExogenousStepInput(
                t_out_c=-5.0,
                i_sol_wm2=i_sol,
                t_down_c=5.0,
                q_int_known_w=q_int,
                t_up_c=12.0,
            )
            state = zoh_kalman.propagate_open_loop_step(
                state, adiabatic, exog, q_fh_w=q_fh, dt_s=300.0
            )
            injected_wh += (
                q_fh + q_int + adiabatic.a_sol_m2 * i_sol
            ) * (300.0 / 3600.0)

        e1_wh = (
            adiabatic.c_air_wh_k * state.t_air_c
            + adiabatic.c_floor_wh_k * state.t_floor_c
        )
        assert (e1_wh - e0_wh) == pytest.approx(injected_wh, abs=1e-8)


class TestResolutionTobitAndKalmanGuards:
    """Verifies resolution detector, Tobit, cooking spike, and open-window."""

    def test_resolution_detector_night_plateau_immunity_and_fallback(
        self,
    ) -> None:
        det = zoh_kalman.ResolutionDetector()
        assert det.is_fine_resolution is False

        assert det.observe(19.84) is True
        det.observe(19.00)
        for _ in range(25):
            assert det.observe(19.00) is True

        # `19.00` above was distinct coarse sample #1; `step == 11` (`20.1`)
        # hits 12 distinct `0.1 °C` samples and triggers coarse fallback.
        for step in range(1, 13):
            res = det.observe(19.0 + 0.1 * step)
            if step < 11:
                assert res is True
            else:
                assert res is False

    def test_tobit_deadband_on_01c_sensor_and_001c_humidity_heartbeat(
        self,
    ) -> None:
        priors = house_config.DAY1_PRIORS_WK
        exog = contracts.ExogenousStepInput(
            t_out_c=2.0, t_up_c=18.0, t_down_c=10.5, q_int_known_w=50.0
        )

        est_tobit = zoh_kalman.ZoneKalmanEstimator("wk", priors)
        est_stale = zoh_kalman.ZoneKalmanEstimator("wk", priors)
        est_tobit.cold_start(20.0, exog)
        est_stale.cold_start(20.0, exog)
        est_tobit.state = dataclasses.replace(est_tobit.state, t_floor_c=19.8)
        est_stale.state = dataclasses.replace(est_stale.state, t_floor_c=19.8)

        ts = _utc_ts(2026, 1, 20, 12, 0)
        for k in range(1, 10):
            ts_k = ts + k * 300
            est_tobit.step(
                ts_k,
                300.0,
                exog,
                0.0,
                0.0,
                0.0,
                20.0,
                temp_changed=False,
                sensor_stale=False,
            )
            est_stale.step(
                ts_k,
                300.0,
                exog,
                0.0,
                0.0,
                0.0,
                20.0,
                temp_changed=False,
                sensor_stale=True,
            )

        assert abs(est_tobit.state.t_air_c - 20.0) < abs(
            est_stale.state.t_air_c - 20.0
        )
        assert est_tobit.cov_p[0][0] < est_stale.cov_p[0][0]

        est_fine = zoh_kalman.ZoneKalmanEstimator("wk", priors)
        est_fine.cold_start(20.03, exog)
        est_fine.state = dataclasses.replace(est_fine.state, t_floor_c=19.8)
        assert est_fine.resolution_detector.is_fine_resolution is True

        for k in range(1, 6):
            est_fine.step(
                ts + k * 300,
                300.0,
                exog,
                0.0,
                0.0,
                0.0,
                20.03,
                temp_changed=False,
                hum_changed=True,
            )
        p00_before = est_fine.cov_p[0][0]
        est_fine.step(
            ts + 6 * 300,
            300.0,
            exog,
            0.0,
            0.0,
            0.0,
            20.03,
            temp_changed=False,
            hum_changed=True,
        )
        assert est_fine.cov_p[0][0] < p00_before

    def test_cooking_spike_900w_isolated_from_concrete_slab(self) -> None:
        """A `+900 W` evening cooking spike routes to `d_est`, not `T_floor`."""
        priors = house_config.DAY1_PRIORS_WK
        exog = contracts.ExogenousStepInput(
            t_out_c=4.0, t_up_c=18.0, t_down_c=10.5, q_int_known_w=50.0
        )
        est = zoh_kalman.ZoneKalmanEstimator("wk", priors)
        truth = contracts.ThermalState(
            t_air_c=19.5, t_floor_c=21.6, d_est_w=0.0
        )
        est.cold_start(truth.t_air_c, exog)

        ts0 = _utc_ts(2026, 1, 14, 23, 0)
        for step in range(1, 217):
            heating_on = 84 <= step < 120
            q_fh_step = 2200.0 if heating_on else 0.0
            u_pump_step = 1.0 if heating_on else 0.0
            truth = zoh_kalman.propagate_open_loop_step(
                truth, priors, exog, q_fh_w=q_fh_step, dt_s=300.0
            )
            est.step(
                ts0 + step * 300,
                300.0,
                exog,
                q_fh_step,
                0.0,
                u_pump_step,
                round(truth.t_air_c, 2),
            )

        ts_cook_start = ts0 + 216 * 300
        exog_cook_truth = dataclasses.replace(exog, q_int_known_w=50.0 + 900.0)
        max_floor_err = 0.0
        for step in range(1, 8):
            for _ in range(30):
                truth = zoh_kalman.propagate_open_loop_step(
                    truth, priors, exog_cook_truth, q_fh_w=0.0, dt_s=10.0
                )
            est_state = est.step(
                ts_cook_start + step * 300,
                300.0,
                exog,
                0.0,
                0.0,
                0.0,
                round(truth.t_air_c, 2),
                aanwezig=True,
            )
            max_floor_err = max(
                max_floor_err, abs(est_state.t_floor_c - truth.t_floor_c)
            )

        assert est.state.d_est_w > 550.0
        assert max_floor_err < 0.04

    def test_open_window_gate_zero_false_slab_pull_spd_and_rebound(
        self,
    ) -> None:
        """Open window holds `K_floor == 0` (`0.0000 °C` pull) for 35 min."""
        priors = house_config.DAY1_PRIORS_WK
        exog = contracts.ExogenousStepInput(
            t_out_c=3.0, t_up_c=18.0, t_down_c=10.5, q_int_known_w=50.0
        )
        est = zoh_kalman.ZoneKalmanEstimator("wk", priors)
        est.cold_start(20.02, exog)
        ts0 = _utc_ts(2026, 1, 15, 10, 0)

        truth = est.state
        for k in range(1, 5):
            truth = zoh_kalman.propagate_open_loop_step(
                truth, priors, exog, q_fh_w=0.0, dt_s=300.0
            )
            est.step(
                ts0 + k * 300, 300.0, exog, 0.0, 0.0, 0.0, round(truth.t_air_c, 2)
            )

        est_tick = zoh_kalman.ZoneKalmanEstimator.from_dict(est.to_dict())
        est_tick.step(
            ts0 + 5 * 300, 300.0, exog, 0.0, 0.0, 0.0, truth.t_air_c - 0.10
        )
        assert est_tick.open_window_active is False

        est_hold = zoh_kalman.ZoneKalmanEstimator.from_dict(est.to_dict())
        est_hold.step(
            ts0 + 5 * 300, 300.0, exog, 0.0, 0.0, 0.0, truth.t_air_c - 0.60
        )
        assert est_hold.open_window_active is True
        for s_idx in range(1, 8):
            est_hold.step(
                ts0 + (5 + s_idx) * 300,
                300.0,
                exog,
                0.0,
                0.0,
                0.0,
                truth.t_air_c,
            )
            if s_idx <= 6:
                assert est_hold.open_window_active is True
            else:
                assert est_hold.open_window_active is False

        drops = [-0.40, -0.75, -1.00, -1.10]
        ts_w = ts0 + 4 * 300
        for idx, d_air in enumerate(drops, start=1):
            truth = zoh_kalman.propagate_open_loop_step(
                truth, priors, exog, q_fh_w=0.0, dt_s=300.0
            )
            prior = zoh_kalman.propagate_open_loop_step(
                est.state, priors, exog, q_fh_w=0.0, dt_s=300.0
            )
            y_open = truth.t_air_c + d_air
            post = est.step(
                ts_w + idx * 300, 300.0, exog, 0.0, 0.0, 0.0, y_open
            )
            assert est.open_window_active is True
            assert abs(post.t_floor_c - prior.t_floor_c) == pytest.approx(
                0.0, abs=1e-14
            )
            for r in range(3):
                assert est.cov_p[r][r] > 0.0
                for c in range(3):
                    assert est.cov_p[r][c] == pytest.approx(
                        est.cov_p[c][r], abs=1e-15
                    )

        ts_close = ts_w + 4 * 300
        rebound_offsets = [-0.45, -0.15] + ([0.0] * 12)
        for idx, d_reb in enumerate(rebound_offsets, start=1):
            truth = zoh_kalman.propagate_open_loop_step(
                truth, priors, exog, q_fh_w=0.0, dt_s=300.0
            )
            y_reb = truth.t_air_c + d_reb
            est.step(ts_close + idx * 300, 300.0, exog, 0.0, 0.0, 0.0, y_reb)
        assert est.open_window_active is False
        assert abs(est.state.d_est_w) < 90.0

    def test_nocturnal_anchoring_48h_replay_surface_and_gap_bridge(
        self,
    ) -> None:
        priors = house_config.DAY1_PRIORS_WK
        exog = contracts.ExogenousStepInput(
            t_out_c=2.0, t_up_c=17.5, t_down_c=10.5, q_int_known_w=50.0
        )
        truth = contracts.ThermalState(
            t_air_c=19.5, t_floor_c=21.7, d_est_w=0.0
        )

        est_night = zoh_kalman.ZoneKalmanEstimator("wk", priors)
        est_night.cold_start(truth.t_air_c, exog)
        est_night.state = dataclasses.replace(
            truth, t_floor_c=truth.t_floor_c + 0.40
        )
        ts_night = _utc_ts(2026, 1, 15, 23, 30)
        t_cur = truth
        for k in range(1, 67):
            t_cur = zoh_kalman.propagate_open_loop_step(
                t_cur, priors, exog, q_fh_w=0.0, dt_s=300.0
            )
            est_night.step(
                ts_night + k * 300, 300.0, exog, 0.0, 0.0, 0.0, t_cur.t_air_c
            )
        assert abs(est_night.state.t_floor_c - t_cur.t_floor_c) < 0.05

        p11_no_surf = est_night.cov_p[1][1]
        t_surf_true = solar_weather.estimate_surface_temp(
            t_cur.t_air_c,
            t_cur.t_floor_c,
            priors.r_fh_k_kw,
            house_config.AREA_WK_M2,
        )
        est_night.step(
            ts_night + 67 * 300,
            300.0,
            exog,
            0.0,
            0.0,
            0.0,
            t_cur.t_air_c,
            t_surf_meas_c=t_surf_true,
        )
        assert est_night.cov_p[1][1] < p11_no_surf

        new_params = dataclasses.replace(
            priors, r_fh_k_kw=2.15, h_loss_w_k=54.0
        )
        history: list[tuple[int, float]] = []
        t_sim = contracts.ThermalState(
            t_air_c=19.8, t_floor_c=21.4, d_est_w=0.0
        )
        ts_rep0 = _utc_ts(2026, 1, 10, 0, 0)
        for k in range(1, 577):
            q_in = 1400.0 if (k % 144) < 36 else 0.0
            t_sim = zoh_kalman.propagate_open_loop_step(
                t_sim, new_params, exog, q_fh_w=q_in, dt_s=300.0
            )
            history.append((ts_rep0 + k * 300, t_sim.t_air_c))

        est_replay = zoh_kalman.ZoneKalmanEstimator("wk", new_params)
        est_replay.cold_start(history[0][1], exog)
        for idx, (ts_k, y_k) in enumerate(history, start=1):
            q_in = 1400.0 if (idx % 144) < 36 else 0.0
            est_replay.step(
                ts_k,
                300.0,
                exog,
                q_in,
                0.0,
                1.0 if q_in > 0 else 0.0,
                y_k,
                aanwezig=False,
            )
        assert abs(est_replay.state.t_floor_c - t_sim.t_floor_c) < 0.02

        est_gap = zoh_kalman.ZoneKalmanEstimator.from_dict(est_replay.to_dict())
        t_after_2h = zoh_kalman.propagate_open_loop_step(
            t_sim, new_params, exog, q_fh_w=2000.0, dt_s=7200.0
        )
        bridged = est_gap.bridge_restart_gap(
            ts_utc=history[-1][0] + 7200,
            dt_gap_s=7200.0,
            exog_saved=exog,
            exog_now=exog,
            q_fh_saved_w=2000.0,
            q_fh_now_w=2000.0,
            q_ac_saved_w=0.0,
            q_ac_now_w=0.0,
            u_pump_avg=1.0,
            t_air_meas_c=t_after_2h.t_air_c,
        )
        assert abs(bridged.t_floor_c - t_after_2h.t_floor_c) < 0.05

    def test_qd_basis_cached_linearity_equivalence_and_zero_qfh_misses(
        self,
    ) -> None:
        """`_discretize_qd_basis_cached` matches Van Loan (<1e-12) & caches."""
        params = dataclasses.replace(
            house_config.DAY1_PRIORS_WK,
            c_wind=0.08,
            h_loss_w_k=53.5,
            r_fh_k_kw=2.12,
        )
        for dt_s, v_wind, sigma_d, u_pump, q_fh, q_vent in [
            (300.0, 2.0, 110.0, 1.0, 2350.0, 0.0),
            (300.0, 7.5, 280.0, 0.7, 940.0, zoh_kalman.Q_AIR_OPEN_WINDOW_K2_S),
            (10800.0, 1.5, 20.0, 0.0, 0.0, 0.0),
            (900.0, 5.2, 75.0, 1.0, 1825.0, 0.0),
        ]:
            v_exc = max(0.0, v_wind - 3.0) if params.c_wind != 0.0 else 0.0
            g0, g1, g2 = zoh_kalman._discretize_qd_basis_cached(
                params, dt_s, v_exc
            )
            for gm in (g0, g1, g2):
                for i in range(3):
                    for j in range(3):
                        assert gm[i][j] == gm[j][i]

            c_floor = 3600.0 * params.c_floor_wh_k
            tau_d = 3600.0 * params.tau_d_h
            qc00 = zoh_kalman.Q_AIR_BASE_K2_S + q_vent
            qc11 = (
                zoh_kalman.Q_FLOOR_IDLE_K2_S
                + u_pump
                * (
                    zoh_kalman.Q_FLOOR_PUMP_K2_S
                    - zoh_kalman.Q_FLOOR_IDLE_K2_S
                )
                + ((0.08 * abs(q_fh)) ** 2)
                * zoh_kalman.TAU_HYDR_S
                / (c_floor * c_floor)
            )
            qc22 = (2.0 * sigma_d * sigma_d) / tau_d

            _, _, qd_fn = zoh_kalman.discretize_zoh_3x3(
                params,
                dt_s=dt_s,
                v_wind_ms=v_wind,
                sigma_d_w=sigma_d,
                u_pump=u_pump,
                q_fh_w=q_fh,
                q_air_vent=q_vent,
            )
            for i in range(3):
                for j in range(3):
                    qd_lin = (
                        qc00 * g0[i][j] + qc11 * g1[i][j] + qc22 * g2[i][j]
                    )
                    assert abs(qd_lin - qd_fn[i][j]) < 1e-12

            if v_wind <= 3.0 and q_vent == 0.0:
                _, _, qd_dec = _decimal_reference_zoh(
                    params, dt_s, sigma_d, u_pump, q_fh
                )
                for i in range(3):
                    for j in range(3):
                        assert abs(qd_fn[i][j] - qd_dec[i][j]) < 1e-12

        zoh_kalman._discretize_qd_basis_cached.cache_clear()
        for step_idx in range(50):
            zoh_kalman.discretize_zoh_3x3(
                params,
                dt_s=300.0,
                v_wind_ms=2.2,
                sigma_d_w=110.0,
                u_pump=1.0,
                q_fh_w=200.0 + 41.3 * step_idx,
            )
        info = zoh_kalman._discretize_qd_basis_cached.cache_info()
        assert info.misses == 1
        assert info.hits == 49
