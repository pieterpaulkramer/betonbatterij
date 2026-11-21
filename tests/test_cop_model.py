"""Offline verification suite for Component 6 heat-pump COP and P1 bias model.

Verifies thermodynamic monotonicity, frost penalty bell around `+1 °C`, the
two-stage `51.0 °C` DHW backup-heater kink, nocturnal P1 flank bias recovery,
and Choice D2 frost-memory retention across warm spring weeks.
"""

from __future__ import annotations

import datetime

import pytest

from custom_components.betonbatterij import const
from custom_components.betonbatterij import house_config
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import cop_model


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


class TestCopPhysicsAndMonotonicity:
    """Verifies Carnot curves, frost bell, and `51.0 °C` DHW heater kink."""

    def test_frost_penalty_factor_bounds_and_peak(self) -> None:
        assert cop_model.frost_penalty_factor(1.0, 96.0) == pytest.approx(1.0)
        assert cop_model.frost_penalty_factor(-5.5, 98.0) == 0.0
        assert cop_model.frost_penalty_factor(6.0, 98.0) == 0.0
        assert cop_model.frost_penalty_factor(1.0, 75.0) == 0.0
        assert 0.3 < cop_model.frost_penalty_factor(0.0, 88.0) < 0.7

    def test_predict_cop_cv_monotonicity_and_frost_dip(self) -> None:
        day1 = house_config.DAY1_COP_PARAMS
        sups = [20.0, 24.0, 28.0, 32.0, 38.0, 45.0]
        cops_sup = [
            cop_model.predict_cop_cv(
                day1, t_out_c=7.0, t_sup_c=s, rh_out_pct=75.0
            )
            for s in sups
        ]
        for c_low, c_high in zip(cops_sup, cops_sup[1:], strict=False):
            assert c_low >= c_high

        t_outs = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
        cops_out = [
            cop_model.predict_cop_cv(
                day1, t_out_c=t, t_sup_c=30.0, rh_out_pct=70.0
            )
            for t in t_outs
        ]
        for c_cold, c_warm in zip(cops_out, cops_out[1:], strict=False):
            assert c_cold < c_warm

        cop_dry = cop_model.predict_cop_cv(
            day1, t_out_c=1.0, t_sup_c=29.0, rh_out_pct=70.0
        )
        cop_fog = cop_model.predict_cop_cv(
            day1, t_out_c=1.0, t_sup_c=29.0, rh_out_pct=96.0
        )
        assert cop_fog == pytest.approx(
            cop_dry * (1.0 - day1.beta_defrost), rel=1e-6
        )

    def test_predict_cop_dhw_two_stage_51c_kink(self) -> None:
        day1 = house_config.DAY1_COP_PARAMS
        cop_50 = cop_model.predict_cop_dhw(
            day1, t_out_c=7.0, t_dhw_start_c=40.0, t_dhw_target_c=50.0
        )
        cop_51 = cop_model.predict_cop_dhw(
            day1, t_out_c=7.0, t_dhw_start_c=40.0, t_dhw_target_c=51.0
        )
        cop_55 = cop_model.predict_cop_dhw(
            day1, t_out_c=7.0, t_dhw_start_c=40.0, t_dhw_target_c=55.0
        )
        cop_62 = cop_model.predict_cop_dhw(
            day1, t_out_c=7.0, t_dhw_start_c=40.0, t_dhw_target_c=62.0
        )
        cop_above_51 = cop_model.predict_cop_dhw(
            day1, t_out_c=7.0, t_dhw_start_c=52.0, t_dhw_target_c=60.0
        )

        assert cop_50 > 2.0
        slope_below = cop_50 - cop_51
        slope_above = (cop_51 - cop_55) / 4.0
        assert slope_above > 3.5 * slope_below
        assert cop_51 > cop_55 > cop_62 > 1.0
        assert cop_above_51 == pytest.approx(1.0)

    def test_corrected_electrical_power(self) -> None:
        day1 = house_config.DAY1_COP_PARAMS
        assert (
            cop_model.corrected_wp_electrical_power_w(day1, 0.0, f_comp_hz=0.0)
            == 0.0
        )
        assert cop_model.corrected_wp_electrical_power_w(
            day1, 600.0, f_comp_hz=20.0
        ) == pytest.approx(745.0)
        assert cop_model.corrected_wp_electrical_power_w(
            day1, 800.0, f_comp_hz=35.0, internal_heater_on=True
        ) == pytest.approx(800.0 + 145.0 + 3000.0)


class TestP1FlankAndCopCalibration:
    """Verifies nocturnal P1 flank recovery and Choice D2 COP calibration."""

    def test_calibrate_p1_power_bias_recovers_clean_night_flanks(self) -> None:
        true_bias_w = 165.0
        t0 = _utc_ts(2026, 1, 15, 1, 0)  # 02:00 CET inside 00:30..06:00.
        ts_utc: list[int] = []
        f_comp: list[float] = []
        p_raw: list[float] = []
        p_p1: list[float] = []

        for k in range(14):
            ts_utc.append(t0 + k * const.DT_STEP_S)
            if 4 <= k < 10:
                f_comp.append(21.0)
                p_raw.append(600.0)
                p_p1.append(130.0 + 600.0 + true_bias_w + (k % 2) * 2.0)
            else:
                f_comp.append(0.0)
                p_raw.append(0.0)
                p_p1.append(130.0 + (k % 2) * 2.0)

        # Daytime run at 12:00 UTC must be ignored.
        t_day = _utc_ts(2026, 1, 15, 12, 0)
        for k in range(10):
            ts_utc.append(t_day + k * const.DT_STEP_S)
            if 3 <= k < 8:
                f_comp.append(25.0)
                p_raw.append(800.0)
                p_p1.append(3500.0)
            else:
                f_comp.append(0.0)
                p_raw.append(0.0)
                p_p1.append(400.0)

        # Contaminated night flank (+1800 W appliance jump) must be rejected.
        t_n2 = _utc_ts(2026, 1, 16, 2, 0)
        for k in range(10):
            ts_utc.append(t_n2 + k * const.DT_STEP_S)
            if k < 3:
                f_comp.append(0.0)
                p_raw.append(0.0)
                p_p1.append(120.0)
            elif k < 7:
                f_comp.append(22.0)
                p_raw.append(600.0)
                p_p1.append(1920.0 + 600.0 + true_bias_w)
            else:
                f_comp.append(0.0)
                p_raw.append(0.0)
                p_p1.append(1920.0)

        est_bias = cop_model.calibrate_p1_power_bias(
            {
                "ts_utc": ts_utc,
                "f_comp_hz": f_comp,
                "p_wp_raw_el_w": p_raw,
                "p_p1_net_w": p_p1,
            }
        )
        assert est_bias is not None
        assert est_bias == pytest.approx(true_bias_w, abs=3.0)

    def test_fit_cop_parameters_recovery_and_d2_frost_memory(self) -> None:
        true_params = contracts.CopParameters(
            eta_carnot_cv=0.53,
            beta_defrost=0.25,
            eta_carnot_dhw=0.39,
            delta_p_wp_bias_w=160.0,
        )

        ts_utc: list[int] = []
        flags: list[int] = []
        f_comp: list[float] = []
        p_raw: list[float] = []
        p_prim: list[float] = []
        t_out: list[float] = []
        rh_out: list[float] = []
        t_sup: list[float] = []
        t_dhw_top: list[float] = []
        p_p1: list[float] = []

        t_cursor = _utc_ts(2026, 1, 10, 1, 30)

        def append_idle(steps: int, t_o: float = 4.0) -> None:
            nonlocal t_cursor
            for _ in range(steps):
                ts_utc.append(t_cursor)
                flags.append(0)
                f_comp.append(0.0)
                p_raw.append(0.0)
                p_prim.append(0.0)
                t_out.append(t_o)
                rh_out.append(80.0)
                t_sup.append(25.0)
                t_dhw_top.append(44.0)
                p_p1.append(140.0)
                t_cursor += const.DT_STEP_S

        cv_conditions = [
            (8.0, 28.0, 72.0),
            (1.0, 30.0, 96.0),
            (6.0, 27.0, 75.0),
            (1.0, 29.0, 95.0),
            (10.0, 26.5, 70.0),
            (0.5, 30.5, 96.0),
            (5.0, 28.5, 74.0),
            (1.2, 29.5, 95.0),
        ]
        for t_o, t_s, rh in cv_conditions:
            append_idle(4, t_o)
            cop_true = cop_model.predict_cop_cv(true_params, t_o, t_s, rh)
            for step in range(12):
                ts_utc.append(t_cursor)
                flags.append(
                    const.FLAG_DEFROST_WP if rh > 90.0 and step == 7 else 0
                )
                f_comp.append(22.0)
                p_el_true = 800.0 + true_params.delta_p_wp_bias_w
                p_raw.append(800.0)
                p_prim.append(cop_true * p_el_true)
                t_out.append(t_o)
                rh_out.append(rh)
                t_sup.append(t_s)
                t_dhw_top.append(44.0)
                p_p1.append(140.0 + p_el_true)
                t_cursor += const.DT_STEP_S
        append_idle(4, 4.0)

        for dhw_idx in range(4):
            t_o = 6.0 + dhw_idx
            cop_d = cop_model.predict_cop_dhw(
                true_params, t_o, 40.0, 50.0, 75.0
            )
            for step in range(8):
                ts_utc.append(t_cursor)
                heater = step >= 6
                fl = const.FLAG_DHW_TANK | (
                    const.FLAG_INTERNAL_HEATER if heater else 0
                )
                flags.append(fl)
                f_comp.append(35.0)
                p_raw.append(1200.0)
                p_el_true = (
                    1200.0
                    + true_params.delta_p_wp_bias_w
                    + (3000.0 if heater else 0.0)
                )
                t_tank = 40.0 + 2.0 * step
                p_prim.append((1.0 if heater else cop_d) * p_el_true)
                t_out.append(t_o)
                rh_out.append(75.0)
                t_sup.append(t_tank + 8.5)
                t_dhw_top.append(t_tank)
                p_p1.append(300.0 + p_el_true)
                t_cursor += const.DT_STEP_S
            append_idle(3, t_o)

        cols_winter = {
            "ts_utc": ts_utc,
            "flags": flags,
            "f_comp_hz": f_comp,
            "p_wp_raw_el_w": p_raw,
            "p_prim_th_w": p_prim,
            "t_out_dak_c": t_out,
            "rh_out_pct": rh_out,
            "t_wp_out_c": t_sup,
            "t_dhw_top_c": t_dhw_top,
            "p_p1_net_w": p_p1,
        }
        fitted_w, mem_w, diag_w = cop_model.fit_cop_parameters(
            cols_winter, house_config.DAY1_COP_PARAMS, memory=None
        )

        assert diag_w["n_cv_runs"] == 8
        assert diag_w["n_dhw_runs"] == 4
        assert diag_w["rmse_cop_14d"] < diag_w["prior_rmse_cop_14d"]
        assert fitted_w.eta_carnot_cv == pytest.approx(
            true_params.eta_carnot_cv, abs=0.02
        )
        assert fitted_w.beta_defrost == pytest.approx(
            true_params.beta_defrost, abs=0.03
        )
        assert fitted_w.eta_carnot_dhw == pytest.approx(
            true_params.eta_carnot_dhw, abs=0.025
        )
        assert fitted_w.delta_p_wp_bias_w == pytest.approx(
            true_params.delta_p_wp_bias_w, abs=15.0
        )

        # Warm spring week (`T_out = 12°C`) must retain learned `beta_defrost`.
        spring_cols = dict(
            cols_winter,
            t_out_dak_c=[12.0] * len(ts_utc),
            rh_out_pct=[65.0] * len(ts_utc),
        )
        spring_p_prim = [
            cop_model.predict_cop_cv(true_params, 12.0, t_sup[i], 65.0)
            * (p_raw[i] + fitted_w.delta_p_wp_bias_w)
            if f_comp[i] > 0 and (flags[i] & const.FLAG_DHW_TANK) == 0
            else p_prim[i]
            for i in range(len(ts_utc))
        ]
        spring_cols["p_prim_th_w"] = spring_p_prim
        fitted_s, _, _ = cop_model.fit_cop_parameters(
            spring_cols, fitted_w, memory=mem_w
        )
        assert fitted_s.beta_defrost == pytest.approx(
            fitted_w.beta_defrost, abs=1e-6
        )
