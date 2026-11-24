"""Offline verification suite for Component 1 solar, weather, and C3 hydraulics.

Verifies NOAA solar angles, Fresnel IAM ratios, seasonal 30°->90° direct/diffuse
split, exterior screen attenuation, negative-EPEX curtailment fallback, summer
morning sun-guard, 12h soft interfloor hold, and Choice C3 hydraulics across
Regimes 1, 2, and 3.
"""

from __future__ import annotations

from collections.abc import Sequence
import datetime
import math

import pytest

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import engine
from custom_components.betonbatterij.domain import evaluation
from custom_components.betonbatterij.domain import hydraulics
from custom_components.betonbatterij.domain import kalman_tape
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
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


class TestSolarGeometryAndProjection:
    """Verifies NOAA geometry, Fresnel IAM, seasonal ratios, and fallbacks."""

    def test_solar_position_cardinal_and_seasons(self) -> None:
        ts_dec_noon = _utc_ts(2026, 12, 21, 11, 36)
        ts_mar_noon = _utc_ts(2026, 3, 20, 11, 45)
        ts_jun_noon = _utc_ts(2026, 6, 21, 11, 39)
        ts_midnight = _utc_ts(2026, 12, 21, 0, 0)

        el_dec, az_dec = solar_weather.solar_position(ts_dec_noon)
        el_mar, az_mar = solar_weather.solar_position(ts_mar_noon)
        el_jun, az_jun = solar_weather.solar_position(ts_jun_noon)
        el_mid, _ = solar_weather.solar_position(ts_midnight)

        assert 14.5 < el_dec < 16.0
        assert 38.0 < el_mar < 39.5
        assert 61.5 < el_jun < 62.6
        assert el_mid < -50.0

        assert 177.0 < az_dec < 183.0
        assert 177.0 < az_mar < 183.0
        assert 177.0 < az_jun < 183.0

        # Morning is East (~90°); afternoon is West (~270°).
        _, az_morn = solar_weather.solar_position(_utc_ts(2026, 3, 20, 6, 0))
        _, az_even = solar_weather.solar_position(_utc_ts(2026, 3, 20, 17, 30))
        assert 80.0 < az_morn < 105.0
        assert 255.0 < az_even < 280.0

    def test_fresnel_iam_properties(self) -> None:
        assert solar_weather.fresnel_iam(-0.2) == 0.0
        assert solar_weather.fresnel_iam(0.0) == 0.0
        assert solar_weather.fresnel_iam(1.0, glazing_panes=1) == pytest.approx(
            1.0
        )
        assert solar_weather.fresnel_iam(1.0, glazing_panes=2) == pytest.approx(
            1.0
        )

        grid = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        vals_2 = [solar_weather.fresnel_iam(c, glazing_panes=2) for c in grid]
        for low, high in zip(vals_2, vals_2[1:], strict=False):
            assert low < high

        # HR++ 2-pane has stronger oblique reflection loss than 1-pane PV glass.
        assert solar_weather.fresnel_iam(
            0.4, glazing_panes=2
        ) < solar_weather.fresnel_iam(0.4, glazing_panes=1)

    def test_seasonal_clear_sky_vertical_to_roof_ratios(self) -> None:
        """Winter low sun boosts 90° South glazing relative to 30° PV roof."""
        ts_dec = _utc_ts(2026, 12, 21, 11, 36)
        ts_mar = _utc_ts(2026, 3, 20, 11, 45)
        ts_jun = _utc_ts(2026, 6, 21, 11, 39)

        i_wk_dec, i_kt_dec, tau_dec, fb_dec = (
            solar_weather.project_solar_irradiance(ts_dec, p_pv_w=2600.0)
        )
        i_wk_mar, i_kt_mar, tau_mar, fb_mar = (
            solar_weather.project_solar_irradiance(ts_mar, p_pv_w=4600.0)
        )
        i_wk_jun, i_kt_jun, tau_jun, fb_jun = (
            solar_weather.project_solar_irradiance(ts_jun, p_pv_w=5100.0)
        )

        kwp = priors.PV_PEAK_KWP
        r_dec = i_wk_dec / (2600.0 / kwp)
        r_mar = i_wk_mar / (4600.0 / kwp)
        r_jun = i_wk_jun / (5100.0 / kwp)

        assert not (fb_dec or fb_mar or fb_jun)
        assert tau_dec == pytest.approx(1.0)
        assert tau_mar == pytest.approx(1.0)
        assert tau_jun == pytest.approx(1.0)

        assert 1.22 < r_dec < 1.42
        assert 0.74 < r_mar < 0.88
        assert 0.38 < r_jun < 0.52
        assert r_dec > r_mar > r_jun
        assert i_kt_dec > 0.0 and i_kt_mar > 0.0 and i_kt_jun > 0.0

    def test_cloudy_winter_avoids_direct_beam_overestimate(self) -> None:
        """Heavy winter cloud cover (`k_c <= 0.25`) uses `0.54` diffuse factor."""
        ts_dec = _utc_ts(2026, 12, 21, 11, 36)
        p_pv_cloudy = 350.0
        i_wk, _, _, _ = solar_weather.project_solar_irradiance(
            ts_dec, p_pv_w=p_pv_cloudy
        )
        i_dak = p_pv_cloudy / priors.PV_PEAK_KWP
        assert (i_wk / i_dak) == pytest.approx(0.54, abs=0.01)

    def test_sun_behind_south_facade_gives_zero_direct_beam(self) -> None:
        """On June mornings (`|az - 175°| > 90°`), direct beam on `wk` is 0."""
        ts_jun_morn = _utc_ts(2026, 6, 21, 5, 30)
        elev, az = solar_weather.solar_position(ts_jun_morn)
        assert elev > 10.0
        assert abs(az - 175.0) > 90.0

        i_wk_open, i_kt, _, _ = solar_weather.project_solar_irradiance(
            ts_jun_morn,
            p_pv_w=2000.0,
            screen_small_pos=100.0,
            screen_large_pos=100.0,
        )
        i_wk_closed, _, tau_closed, _ = solar_weather.project_solar_irradiance(
            ts_jun_morn,
            p_pv_w=2000.0,
            screen_small_pos=0.0,
            screen_large_pos=0.0,
        )
        assert i_wk_open > 0.0
        assert i_kt > i_wk_open
        assert tau_closed == pytest.approx(0.45, abs=1e-6)
        assert i_wk_closed == pytest.approx(0.45 * i_wk_open, rel=1e-6)

    def test_exterior_screens_attenuation_weights(self) -> None:
        ts_mar = _utc_ts(2026, 3, 20, 11, 45)
        i_open, _, tau_open, _ = solar_weather.project_solar_irradiance(
            ts_mar,
            p_pv_w=4600.0,
            screen_small_pos=100.0,
            screen_large_pos=100.0,
        )
        i_half, _, tau_half, _ = solar_weather.project_solar_irradiance(
            ts_mar,
            p_pv_w=4600.0,
            screen_small_pos=100.0,
            screen_large_pos=0.0,
        )
        i_closed, _, tau_closed, _ = solar_weather.project_solar_irradiance(
            ts_mar,
            p_pv_w=4600.0,
            screen_small_pos=0.0,
            screen_large_pos=0.0,
        )
        assert tau_open == pytest.approx(1.0)
        assert 0.10 <= tau_closed <= 0.18
        assert i_closed < i_half < i_open
        assert 0.35 < tau_half < 0.50

    def test_curtailment_and_stale_pv_fallback(self) -> None:
        ts_may_noon = _utc_ts(2026, 5, 1, 11, 30)
        i_wk_curt, _, _, fb_curt = solar_weather.project_solar_irradiance(
            ts_may_noon,
            p_pv_w=15.0,
            p_pv_forecast_w=3800.0,
            epex_eur_kwh=-0.12,
        )
        i_wk_direct, _, _, fb_direct = solar_weather.project_solar_irradiance(
            ts_may_noon,
            p_pv_w=3800.0,
            p_pv_forecast_w=3800.0,
            epex_eur_kwh=0.25,
        )
        assert fb_curt is True
        assert fb_direct is False
        assert i_wk_curt == pytest.approx(i_wk_direct, rel=1e-9)

        i_wk_nan, _, _, fb_nan = solar_weather.project_solar_irradiance(
            ts_may_noon, p_pv_w=math.nan, p_pv_forecast_w=2500.0
        )
        assert fb_nan is True
        assert i_wk_nan > 150.0


class TestOutdoorInterfloorAndInternalLoads:
    """Verifies sun-guard, 12h soft interfloor hold, desk/PIR, and AC modes."""

    def test_summer_morning_sun_guard_and_fallbacks(self) -> None:
        ts_jun_early = _utc_ts(2026, 6, 21, 5, 15)
        elev, az = solar_weather.solar_position(ts_jun_early)
        assert 0.0 < elev < 18.0 and az < 130.0

        t_guarded = solar_weather.select_outdoor_temp(
            ts_jun_early,
            t_dak_c=22.0,
            t_weather_c=15.0,
            t_hm_ema_c=14.5,
            hm_bias_c=0.3,
        )
        assert t_guarded == pytest.approx(15.8)

        ts_jun_noon = _utc_ts(2026, 6, 21, 11, 39)
        assert solar_weather.select_outdoor_temp(
            ts_jun_noon, 22.0, 19.0, 18.0
        ) == pytest.approx(22.0)

        assert solar_weather.select_outdoor_temp(
            ts_jun_noon, None, 12.4, 11.0
        ) == pytest.approx(12.4)
        assert solar_weather.select_outdoor_temp(
            ts_jun_noon, math.nan, None, 8.5, hm_bias_c=-0.5
        ) == pytest.approx(8.0)
        assert solar_weather.select_outdoor_temp(
            ts_jun_noon, None, None, None
        ) == pytest.approx(10.0)

    def test_interfloor_12h_soft_hold_and_group_fallback(self) -> None:
        cache: dict[str, tuple[float, int]] = {}
        rooms: dict[str, float | None] = {
            "slaapkamer": 17.0,
            "kamer_1e_zuid": 18.0,
            "werkkamer_1e": 19.0,
            "kamer_1e_midden": 18.5,
            "badkamer": 20.5,
        }
        t_up, t_down = solar_weather.compute_interfloor_boundaries(
            rooms, cache, ts_utc=10_000
        )
        assert t_up == pytest.approx((2.0 * 17.0 + 18.0 + 19.0) / 4.0)
        assert t_down == pytest.approx(0.5 * (18.5 + 20.5))

        rooms_drop = dict(rooms, kamer_1e_midden=None)
        _, t_down_8h = solar_weather.compute_interfloor_boundaries(
            rooms_drop, cache, ts_utc=10_000 + 28_800
        )
        assert t_down_8h == pytest.approx(0.5 * (18.5 + 20.5))

        rooms_drop2: dict[str, float | None] = {
            "slaapkamer": 17.0,
            "kamer_1e_zuid": 16.0,
            "werkkamer_1e": 18.0,
            "kamer_1e_midden": None,
            "badkamer": 20.0,
        }
        _, t_down_13h = solar_weather.compute_interfloor_boundaries(
            rooms_drop2, cache, ts_utc=10_000 + 46_800
        )
        assert t_down_13h == pytest.approx(0.5 * (17.0 + 20.0))

    def test_kt_workplace_schmitt_pir_and_ac_modes(self) -> None:
        desk_state: dict[str, object] = {}
        q_int, q_ac = solar_weather.compute_internal_and_ac_loads_kt(
            p_desk_w=65.0,
            pir_occ=0.0,
            p_growlights_w=23.0,
            p_el_ac_w=403.0,
            ac_mode_code=const.AC_MODE_HEAT,
            ac_defrost=False,
            t_out_c=7.0,
            eta_ac=3.6,
            desk_state=desk_state,
            dt_s=300.0,
        )
        assert q_int == pytest.approx(23.0 + 65.0 + 85.0)
        assert q_ac == pytest.approx(400.0 * 3.6)

        q_int_hyst, _ = solar_weather.compute_internal_and_ac_loads_kt(
            28.0, 0.0, 0.0, 0.0, const.AC_MODE_OFF, False, 7.0, 3.6, desk_state
        )
        assert q_int_hyst == pytest.approx(28.0 + 85.0)

        q_hold1, _ = solar_weather.compute_internal_and_ac_loads_kt(
            10.0, 0.0, 0.0, 0.0, const.AC_MODE_OFF, False, 7.0, 3.6, desk_state
        )
        q_hold2, _ = solar_weather.compute_internal_and_ac_loads_kt(
            10.0, 0.0, 0.0, 0.0, const.AC_MODE_OFF, False, 7.0, 3.6, desk_state
        )
        q_expired, _ = solar_weather.compute_internal_and_ac_loads_kt(
            10.0, 0.0, 0.0, 0.0, const.AC_MODE_OFF, False, 7.0, 3.6, desk_state
        )
        assert q_hold1 == pytest.approx(10.0 + 85.0)
        assert q_hold2 == pytest.approx(10.0 + 85.0)
        assert q_expired == pytest.approx(10.0)

        pir_state: dict[str, object] = {}
        q_pir0, q_ac_def = solar_weather.compute_internal_and_ac_loads_kt(
            None,
            1.0,
            0.0,
            503.0,
            const.AC_MODE_HEAT,
            True,
            1.0,
            3.6,
            pir_state,
        )
        assert q_pir0 == pytest.approx(140.0)
        assert q_ac_def == pytest.approx(-125.0)

        for _ in range(3):
            q_pir_hold, _ = solar_weather.compute_internal_and_ac_loads_kt(
                None,
                0.0,
                0.0,
                0.0,
                const.AC_MODE_OFF,
                False,
                7.0,
                3.6,
                pir_state,
            )
            assert q_pir_hold == pytest.approx(140.0)
        q_pir_off, q_cool = solar_weather.compute_internal_and_ac_loads_kt(
            None,
            0.0,
            0.0,
            303.0,
            const.AC_MODE_COOL,
            False,
            26.0,
            3.6,
            pir_state,
        )
        assert q_pir_off == pytest.approx(0.0)
        assert q_cool == pytest.approx(-3.5 * 300.0)

        _, q_dry = solar_weather.compute_internal_and_ac_loads_kt(
            None,
            0.0,
            0.0,
            203.0,
            const.AC_MODE_DRY,
            False,
            24.0,
            3.6,
            pir_state,
        )
        assert q_dry == pytest.approx(-2.5 * 200.0)

    def test_surface_temperature_voltage_divider(self) -> None:
        t_surf_wk = zoh_kalman.estimate_surface_temp(
            20.0,
            24.0,
            priors.DAY1_PRIORS_WK.r_fh_k_kw,
            priors.AREA_WK_M2,
        )
        t_surf_kt = zoh_kalman.estimate_surface_temp(
            20.0,
            24.0,
            priors.DAY1_PRIORS_KT.r_fh_k_kw,
            priors.AREA_KT_M2,
        )
        assert 22.4 < t_surf_wk < 23.2
        assert t_surf_wk < t_surf_kt < 23.8


class TestChoiceC3Hydraulics:
    """Verifies Choice C3 mixed-zone split, Regimes 1/2/3, and defrost."""

    def test_mixed_cold_kt_and_warm_wk_eliminates_c1_flow_bias(self) -> None:
        """Cold `kt` (18°C) alongside warm `wk` (23.5°C) avoids -43% C1 bias."""
        hyd = hydraulics.HydraulicModel()
        rooms = {g: 19.0 for g in const.ALL_HYDRAULIC_GROUPS}
        res = hyd.step(
            dt_s=300.0,
            u_pump_sec=1.0,
            u_valves={"bg": 1.0, "kantoor": 1.0},
            threeway_room=True,
            wp_defrost=False,
            v_prim_l_min=14.0,
            p_prim_th_w=2400.0,
            f_comp_hz=20.0,
            t_buf_top_start_c=27.5,
            t_buf_top_end_c=27.5,
            t_buf_bot_start_c=24.0,
            t_buf_bot_end_c=24.0,
            t_floor_wk_c=23.5,
            t_floor_kt_c=18.0,
            room_temps_c=rooms,
        )
        assert res.f_flow_kt == pytest.approx(0.125)
        assert res.q_fh_kt_w == pytest.approx(65.0 * 9.5)
        assert res.q_fh_wk_w == pytest.approx(245.0 * 4.0)

    def test_regimes_1_2_3_telescoped_gamma_and_harvest_hold(self) -> None:
        hyd = hydraulics.HydraulicModel()
        t_sup_r2 = hyd.compute_effective_supply_temp(
            threeway_room=True,
            v_prim_l_min=6.0,
            v_sec_total_l_min=12.0,
            t_buf_top_c=30.0,
            t_buf_bot_raw_c=24.33,
        )
        expected_r2 = 0.5 * 30.0 + 0.5 * (
            24.33 + priors.BUF_BOT_SENSOR_OFFSET_C
        )
        assert t_sup_r2 == pytest.approx(expected_r2)

        t_sup_r3 = hyd.compute_effective_supply_temp(
            threeway_room=False,
            v_prim_l_min=15.0,
            v_sec_total_l_min=8.0,
            t_buf_top_c=29.2,
            t_buf_bot_raw_c=22.0,
        )
        assert t_sup_r3 == pytest.approx(29.2)

        rooms = {g: 20.0 for g in const.ALL_HYDRAULIC_GROUPS}
        expected_c2 = 245.0 * (28.0 - 22.0) + 98.0 * (28.0 - 20.0)
        p_true = 1.15 * expected_c2
        res_last = None
        for _ in range(6):
            res_last = hyd.step(
                dt_s=300.0,
                u_pump_sec=1.0,
                u_valves={"bg": 1.0},
                threeway_room=True,
                wp_defrost=False,
                v_prim_l_min=15.0,
                p_prim_th_w=p_true,
                f_comp_hz=21.0,
                t_buf_top_start_c=28.0,
                t_buf_top_end_c=28.0,
                t_buf_bot_start_c=25.0,
                t_buf_bot_end_c=25.0,
                t_floor_wk_c=22.0,
                t_floor_kt_c=19.0,
                room_temps_c=rooms,
            )
        assert res_last is not None
        assert res_last.gamma_cal == pytest.approx(1.15, rel=0.02)

        gamma_held = hyd.gamma_cal
        res_harvest = hyd.step(
            dt_s=300.0,
            u_pump_sec=1.0,
            u_valves={"bg": 1.0},
            threeway_room=False,
            wp_defrost=False,
            v_prim_l_min=0.0,
            p_prim_th_w=0.0,
            f_comp_hz=0.0,
            t_buf_top_start_c=27.8,
            t_buf_top_end_c=27.2,
            t_buf_bot_start_c=24.8,
            t_buf_bot_end_c=24.2,
            t_floor_wk_c=22.0,
            t_floor_kt_c=19.0,
            room_temps_c=rooms,
        )
        assert res_harvest.gamma_cal == pytest.approx(gamma_held)
        assert res_harvest.q_sec_calorimetric_w > 1200.0

    def test_defrost_extraction_1e_leak_bg_override_and_persistence(
        self,
    ) -> None:
        hyd = hydraulics.HydraulicModel()
        rooms = {g: 18.0 for g in const.ALL_HYDRAULIC_GROUPS}

        res_def = hyd.step(
            dt_s=300.0,
            u_pump_sec=1.0,
            u_valves={"bg": 1.0, "slaapkamer": 1.0, "kamer_1e_zuid": 1.0},
            threeway_room=True,
            wp_defrost=True,
            v_prim_l_min=18.0,
            p_prim_th_w=-4500.0,
            f_comp_hz=45.0,
            t_buf_top_start_c=27.0,
            t_buf_top_end_c=25.0,
            t_buf_bot_start_c=25.0,
            t_buf_bot_end_c=23.0,
            t_floor_wk_c=22.0,
            t_floor_kt_c=18.5,
            room_temps_c=rooms,
            t_bg_sup_c=26.0,
            t_bg_ret_c=23.1,
            bg_ret_offset_c=-0.1,
        )
        assert res_def.gamma_cal == pytest.approx(1.0)
        q_bg_expected = (
            priors.WATER_POWER_FACTOR_W_PER_LMIN_K * 7.0 * 3.0
        )
        assert res_def.q_fh_wk_w == pytest.approx(q_bg_expected * (5.0 / 7.0))
        assert res_def.q_fh_by_group_w["gang"] == pytest.approx(
            q_bg_expected * (2.0 / 7.0)
        )
        assert res_def.q_leak_1e_to_wk_w == pytest.approx(
            priors.BETA_DOWN_1E_LEAK
            * (
                res_def.q_fh_by_group_w["slaapkamer"]
                + res_def.q_fh_by_group_w["kamer_1e_zuid"]
            )
        )

        cap_w = hyd.estimate_total_sink_capacity_w(
            29.0, ["bg", "kt", "slaapkamer"], 22.0, 18.5
        )
        assert cap_w > 3200.0

        restored = hydraulics.HydraulicModel.from_dict(hyd.to_dict())
        assert restored.gamma_cal == pytest.approx(hyd.gamma_cal)
        assert restored.t_floor_proxy_c == pytest.approx(hyd.t_floor_proxy_c)

    def test_eps_ntu_flow_scaling_and_engine_update_flow_rates_sync(
        self,
    ) -> None:
        """Verifies eps-NTU flow scaling (never linear 2x) and engine sync."""
        k_old = priors.DAY1_K_HX_BY_GROUP_W_K["wk"]  # 245.0 @ 5 L/min
        k_doubled_flow = hydraulics.scale_k_hx_eps_ntu(k_old, 5.0, 10.0)
        # Doubling flow increases K_hx physically by ~10..35%, never 2.0x (490)
        assert k_old * 1.10 < k_doubled_flow < k_old * 1.35
        assert (
            k_doubled_flow
            < 0.75 * 10.0 * priors.WATER_POWER_FACTOR_W_PER_LMIN_K
        )
        # Round-trip reversibility
        assert hydraulics.scale_k_hx_eps_ntu(
            k_doubled_flow, 10.0, 5.0
        ) == pytest.approx(k_old, rel=1e-9)
        # No-op on identical or non-positive flow
        assert hydraulics.scale_k_hx_eps_ntu(k_old, 5.0, 5.0) == pytest.approx(
            k_old
        )
        assert hydraulics.scale_k_hx_eps_ntu(k_old, 0.0, 5.0) == pytest.approx(
            k_old
        )

        hyd_init = hydraulics.HydraulicModel(flow_rates_l_min={"wk": 10.0})
        assert hyd_init.k_hx_by_group_w_k["wk"] == pytest.approx(
            k_doubled_flow, rel=1e-9
        )

        # Pump-off fast-path decays unmodeled proxies identically to exact ZOH
        hyd_idle = hydraulics.HydraulicModel()
        rooms_init = {g: 18.0 for g in const.ALL_HYDRAULIC_GROUPS}
        res_seed = hyd_idle.step(
            dt_s=300.0,
            u_pump_sec=0.0,
            u_valves={},
            threeway_room=True,
            wp_defrost=False,
            v_prim_l_min=0.0,
            p_prim_th_w=0.0,
            f_comp_hz=0.0,
            t_buf_top_start_c=24.0,
            t_buf_top_end_c=25.0,
            t_buf_bot_start_c=22.0,
            t_buf_bot_end_c=23.0,
            t_floor_wk_c=21.0,
            t_floor_kt_c=19.0,
            room_temps_c=rooms_init,
        )
        assert res_seed.q_fh_wk_w == 0.0
        assert res_seed.t_sup_sec_c == pytest.approx(24.5)
        assert hyd_idle.t_floor_proxy_c["slaapkamer"] == pytest.approx(18.0)

        rooms_warm = {g: 22.0 for g in const.ALL_HYDRAULIC_GROUPS}
        hyd_idle.step(
            dt_s=300.0,
            u_pump_sec=0.0,
            u_valves={},
            threeway_room=True,
            wp_defrost=False,
            v_prim_l_min=0.0,
            p_prim_th_w=0.0,
            f_comp_hz=0.0,
            t_buf_top_start_c=24.0,
            t_buf_top_end_c=24.0,
            t_buf_bot_start_c=22.0,
            t_buf_bot_end_c=22.0,
            t_floor_wk_c=21.0,
            t_floor_kt_c=19.0,
            room_temps_c=rooms_warm,
        )
        assert 18.0 < hyd_idle.t_floor_proxy_c["slaapkamer"] < 22.0

        # DigitalTwinEngine.update_flow_rates syncs params, estimators & epoch
        eng = engine.DigitalTwinEngine()
        epoch_before = eng.param_epoch
        eng.update_flow_rates({"wk": 10.0, "kantoor": 2.0})
        assert eng.param_epoch == epoch_before + 1
        assert eng.hydraulics.k_hx_by_group_w_k["wk"] == pytest.approx(
            k_doubled_flow, rel=1e-9
        )
        assert eng.get_zone_snapshot("wk").params.k_hx_w_k == pytest.approx(
            k_doubled_flow, rel=1e-9
        )
        assert eng.estimator_wk.params.k_hx_w_k == pytest.approx(
            k_doubled_flow, rel=1e-9
        )
        expected_kt = hydraulics.scale_k_hx_eps_ntu(
            priors.DAY1_K_HX_BY_GROUP_W_K["kantoor"], 1.0, 2.0
        )
        assert eng.get_zone_snapshot("kt").params.k_hx_w_k == pytest.approx(
            expected_kt, rel=1e-9
        )
        assert eng.estimator_kt.params.k_hx_w_k == pytest.approx(
            expected_kt, rel=1e-9
        )

    def test_calibrate_buffer_sensor_offsets_0hz_sniffing_and_gates(
        self,
    ) -> None:
        """Recovers `-1.13` & `-1.63 °C` on 0 Hz sniffing and rejects noise."""
        n = 20
        flags = [const.FLAG_VALID_WK | const.FLAG_VALID_KT] * n
        f_comp_hz = [0.0] * n
        u_pump_sec = [0.0] * n
        v_prim = [15.0] * n
        t_wp_out = [27.25] * n
        t_wp_in = [27.25] * n
        t_top_s = [28.38] * n
        t_top_e = [28.38] * n
        t_bot_s = [28.88] * n
        t_bot_e = [28.88] * n

        # Inject non-qualifying regimes that must be excluded cleanly:
        # k=0: active compressor (resets 4-step equilibration until k=4)
        f_comp_hz[0] = 22.0
        t_top_s[0] = 35.0
        t_top_e[0] = 35.0
        # k=11: DHW tank flag (breaks equilibration for k=11..14)
        flags[11] |= const.FLAG_DHW_TANK
        t_wp_out[11] = 58.0
        t_wp_in[11] = 52.0
        t_top_s[11] = 39.0
        t_top_e[11] = 39.0
        # k=19: secondary floor pump active
        u_pump_sec[19] = 1.0
        t_bot_s[19] = 22.0
        t_bot_e[19] = 22.0

        cols: dict[str, Sequence[float | int]] = {
            "ts_utc": [1_763_000_000 + k * 300 for k in range(n)],
            "flags": flags,
            "f_comp_hz": f_comp_hz,
            "u_pump_sec": u_pump_sec,
            "v_prim_l_min": v_prim,
            "t_wp_out_c": t_wp_out,
            "t_wp_in_c": t_wp_in,
            "t_buf_top_start_c": t_top_s,
            "t_buf_top_end_c": t_top_e,
            "t_buf_bot_start_c": t_bot_s,
            "t_buf_bot_end_c": t_bot_e,
        }
        # Qualifying steps: k=4..10 (7 steps) and k=15..18 (4 steps) = 11 >= 6.
        fitted = hydraulics.calibrate_buffer_sensor_offsets(cols)
        assert fitted.buf_top_offset_c == pytest.approx(-1.13, abs=1e-6)
        assert fitted.buf_bot_offset_c == pytest.approx(-1.63, abs=1e-6)

        # Fallback to current_offsets when < 6 valid sniffing samples exist.
        custom_prev = contracts.BufferSensorOffsets(
            buf_top_offset_c=-0.25, buf_bot_offset_c=-0.55
        )
        short_cols = preprocessing.slice_columns(cols, 0, 9)  # k=4..8 -> 5 < 6
        assert (
            hydraulics.calibrate_buffer_sensor_offsets(
                short_cols, current_offsets=custom_prev
            )
            == custom_prev
        )

        # LTS mask & clamping to [-3.0, 3.0] deg C.
        lts_mask = [True] * 15 + [False] * 5
        assert (
            hydraulics.calibrate_buffer_sensor_offsets(
                cols, current_offsets=custom_prev, is_lts_row=lts_mask
            )
            == custom_prev
        )
        extreme_cols = dict(cols)
        extreme_cols["t_buf_top_start_c"] = [33.0] * n
        extreme_cols["t_buf_top_end_c"] = [33.0] * n
        extreme_cols["t_buf_bot_start_c"] = [22.0] * n
        extreme_cols["t_buf_bot_end_c"] = [22.0] * n
        clamped = hydraulics.calibrate_buffer_sensor_offsets(extreme_cols)
        assert clamped.buf_top_offset_c == pytest.approx(
            priors.BUF_SENSOR_OFFSET_MIN_C
        )
        assert clamped.buf_bot_offset_c == pytest.approx(
            priors.BUF_SENSOR_OFFSET_MAX_C
        )

    def test_hydraulic_model_applies_buffer_offsets_once(self) -> None:
        """Verifies single offset application in supply temp, step, & dict."""
        offsets = contracts.BufferSensorOffsets(
            buf_top_offset_c=-1.13, buf_bot_offset_c=-1.63
        )
        hyd = hydraulics.HydraulicModel(buffer_offsets=offsets)
        rooms = {g: 20.0 for g in const.ALL_HYDRAULIC_GROUPS}

        # Regime 2 bypass mixing (alpha_mix = 0.5):
        # raw top = 31.13 -> cal 30.00; raw bot = 25.63 -> cal 24.00 -> sup 27.0
        t_sup_mix = hyd.compute_effective_supply_temp(
            threeway_room=True,
            v_prim_l_min=6.0,
            v_sec_total_l_min=12.0,
            t_buf_top_c=31.13,
            t_buf_bot_raw_c=25.63,
        )
        assert t_sup_mix == pytest.approx(27.0, abs=1e-9)

        # step() passes raw top/bot into compute_effective_supply_temp ONCE.
        res_active = hyd.step(
            dt_s=300.0,
            u_pump_sec=1.0,
            u_valves={"wk": 1.0, "gang": 1.0},  # 5 + 2 = 7 L/min
            threeway_room=True,
            wp_defrost=False,
            v_prim_l_min=3.5,  # alpha_mix = 0.5
            p_prim_th_w=2000.0,
            f_comp_hz=20.0,
            t_buf_top_start_c=31.13,
            t_buf_top_end_c=31.13,
            t_buf_bot_start_c=25.63,
            t_buf_bot_end_c=25.63,
            t_floor_wk_c=22.0,
            t_floor_kt_c=19.0,
            room_temps_c=rooms,
        )
        assert res_active.t_sup_sec_c == pytest.approx(27.0, abs=1e-9)
        assert res_active.q_fh_wk_w == pytest.approx(
            245.0 * (27.0 - 22.0), rel=1e-9
        )

        # Pump-off fast path also applies buf_top_offset_c once.
        res_idle = hyd.step(
            dt_s=300.0,
            u_pump_sec=0.0,
            u_valves={},
            threeway_room=True,
            wp_defrost=False,
            v_prim_l_min=0.0,
            p_prim_th_w=0.0,
            f_comp_hz=0.0,
            t_buf_top_start_c=28.38,
            t_buf_top_end_c=28.38,
            t_buf_bot_start_c=28.88,
            t_buf_bot_end_c=28.88,
            t_floor_wk_c=22.0,
            t_floor_kt_c=19.0,
            room_temps_c=rooms,
        )
        assert res_idle.t_sup_sec_c == pytest.approx(27.25, abs=1e-9)

        # update_buffer_offsets clears _cal_window and round-trips via to_dict.
        new_offsets = contracts.BufferSensorOffsets(
            buf_top_offset_c=-0.50, buf_bot_offset_c=-0.90
        )
        hyd.update_buffer_offsets(new_offsets)
        assert len(hyd._cal_window) == 0
        restored = hydraulics.HydraulicModel.from_dict(hyd.to_dict())
        assert restored.buffer_offsets == new_offsets

    def test_preprocess_final_state_buffer_offsets_and_harvest_gamma_parity(
        self,
    ) -> None:
        """Verifies final hyd state, single offset, harvest gamma, & split."""
        offsets = contracts.BufferSensorOffsets(
            buf_top_offset_c=-1.00, buf_bot_offset_c=-1.50
        )
        # 6 steady CV steps (establishing gamma_cal ~ 1.18) + 6 harvest steps.
        n = 12
        base_ts = 1_763_000_000
        # Raw top = 29.0 -> calibrated 28.0; raw bot = 26.5 -> calibrated 25.0.
        flags = [const.FLAG_VALID_WK | const.FLAG_VALID_KT] * 6 + [
            const.FLAG_VALID_WK
            | const.FLAG_VALID_KT
            | const.FLAG_Z1_OFF_HARVEST
        ] * 6
        cols: dict[str, Sequence[float | int]] = {
            "ts_utc": [base_ts + k * 300 for k in range(n)],
            "flags": flags,
            "u_pump_sec": [1.0] * n,
            "u_eff_bg": [1.0] * n,
            "u_eff_slaapkamer": [1.0] * 6 + [0.0] * 6,
            "v_prim_l_min": [15.0] * 6 + [0.0] * 6,
            "f_comp_hz": [22.0] * 6 + [0.0] * 6,
            "p_prim_th_w": [4300.0] * 6 + [0.0] * 6,
            "t_buf_top_start_c": [29.0] * n,
            "t_buf_top_end_c": [29.0] * n,
            "t_buf_bot_start_c": [26.5] * n,
            "t_buf_bot_end_c": [26.5] * n,
            "t_air_wk_c": [20.0 + 0.03 * k for k in range(n)],
            "t_air_kantoor_c": [19.0] * n,
            "t_air_slaapkamer_c": [18.0] * n,
            "t_out_dak_c": [4.0] * n,
            "v_wind_ms": [2.5] * n,
            "p_pv_w": [0.0] * n,
        }

        hist = preprocessing.preprocess_history_columns(
            cols, buffer_offsets=offsets
        )
        assert hist.buf_top_offset_c == pytest.approx(-1.00)
        assert hist.buf_bot_offset_c == pytest.approx(-1.50)
        assert hist.t_buf_top_c[0] == pytest.approx(28.0)
        assert hist.t_buf_bot_cal_c[0] == pytest.approx(25.0)
        assert hist.t_sup_sec_c[0] == pytest.approx(28.0)
        assert hist.final_gamma_cal > 1.05
        assert hist.gamma_cal[5] == pytest.approx(hist.final_gamma_cal)
        assert hist.gamma_cal[11] == pytest.approx(hist.final_gamma_cal)
        assert hist.final_t_floor_proxy_c["slaapkamer"] > 18.0

        # recompute_history_for_khx must NOT double-apply buf_top_offset_c.
        hist_recomp = preprocessing.recompute_history_for_khx(
            hist,
            cols,
            k_hx_by_group={"wk": priors.DAY1_K_HX_BY_GROUP_W_K["wk"] + 15.0},
        )
        assert hist_recomp.t_buf_top_c[0] == pytest.approx(28.0)
        assert hist_recomp.t_sup_sec_c[0] == pytest.approx(28.0)

        # Harvest gamma_cal parity between step_q_fh_for_params & kalman_tape.
        params_wk = priors.DAY1_PRIORS_WK
        q_harv = preprocessing.step_q_fh_for_params(
            hist, "wk", 8, t_floor_prev_c=21.5, params=params_wk
        )
        assert q_harv == pytest.approx(
            hist.gamma_cal[8] * params_wk.k_hx_w_k * (28.0 - 21.5),
            rel=1e-12,
        )

        states_ref, q_fh_ref, _, _ = evaluation.reconstruct_kalman_states(
            hist, "wk", params_wk
        )
        tape_wk = kalman_tape.get_zone_tape(hist, "wk")
        states_tape = kalman_tape.reconstruct_states_on_tape(tape_wk, params_wk)
        for st_r, st_t in zip(states_ref, states_tape, strict=True):
            assert st_t[0] == pytest.approx(st_r.t_air_c, abs=1e-11)
            assert st_t[1] == pytest.approx(st_r.t_floor_c, abs=1e-11)
            assert st_t[2] == pytest.approx(st_r.d_est_w, abs=1e-9)

        ad_std, bd_std, _, _, _ = zoh_kalman.discretize_ab_cached(
            params_wk, 300.0, 0.0
        )
        errs_fast, q_fhs_fast = kalman_tape.rollout_open_loop_window_fast(
            tape_wk,
            k0=5,
            length_steps=6,
            state_init=(
                states_ref[5].t_air_c,
                states_ref[5].t_floor_c,
                0.0,
            ),
            params=params_wk,
            ad_std=ad_std,
            bd_std=bd_std,
            khx_ratio=1.0,
            khx_scale_act=1.0,
        )
        res_eval = kalman_tape.evaluate_windows_on_tape(
            tape_wk, params_wk, [(5, 6, [3, 6], 1.0)]
        )
        assert res_eval[0] == pytest.approx(errs_fast[2], abs=1e-11)
        assert res_eval[1] == pytest.approx(errs_fast[5], abs=1e-11)
        assert q_fhs_fast[0] == pytest.approx(
            hist.gamma_cal[6]
            * params_wk.k_hx_w_k
            * (28.0 - states_ref[5].t_floor_c),
            rel=1e-11,
        )
        assert q_fh_ref[8] > 0.0

        # F1 Case C: 80d LTS + 2d raw -> ~65% of raw in train_slice (never 0).
        n_lts = 80 * const.STEPS_PER_DAY
        n_raw = 2 * const.STEPS_PER_DAY
        cols_split: dict[str, Sequence[float | int]] = {
            "ts_utc": [base_ts + k * 300 for k in range(n_lts + n_raw)],
            "flags": [const.FLAG_LTS_BACKFILL] * n_lts + [0] * n_raw,
        }
        train_end = preprocessing.compute_train_holdout_split(cols_split)
        raw_in_train = train_end - n_lts
        raw_in_holdout = (n_lts + n_raw) - train_end
        assert raw_in_train == n_raw - round(0.35 * n_raw)
        assert raw_in_holdout >= preprocessing.HORIZON_6H_STEPS + 24
