"""Waterside hydraulic energy balance and epsilon-NTU floor heat distribution.

Owns the secondary manifold model (`HydraulicModel`), 200 L buffer return-water
bypass mixing, 30-minute telescoped calorimetric scaling (`gamma_cal`), 1-node
slab proxies for unmodeled hydraulic loops, downward 1st-floor ceiling heat
leakage into `wk`, and optional ground-floor clamp-on supply/return overrides.

Does not own room air/slab Kalman state estimation (`zoh_kalman.py`) or raw
telemetry ingestion (`preprocessing.py`).

Units: temperatures in °C, thermal powers in W, conductances in W/K, flow rates
in L/min, time in s.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Mapping
import dataclasses
import math
from typing import Any, TypeGuard

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import priors

GAMMA_CAL_MIN: float = 0.75
GAMMA_CAL_MAX: float = 1.25
TELESCOPE_STEPS: int = 6  # 6 * 5 min = 30 min telescoped energy window.

_LEAKING_1E_GROUPS: tuple[str, ...] = (
    "slaapkamer",
    "kamer_1e_zuid",
    "werkkamer_1e",
)

_ZERO_GROUP_POWERS: dict[str, float] = {
    g: 0.0 for g in const.ALL_HYDRAULIC_GROUPS
}


def _build_proxy_specs() -> tuple[
    tuple[str, float, float, float, float, str], ...
]:
    """Precomputes 300 s ZOH decay factors for the 8 unmodeled slab proxies."""
    specs: list[tuple[str, float, float, float, float, str]] = []
    for g, (
        c_floor_wh_k,
        r_fh_k_kw,
    ) in priors.UNMODELED_GROUP_THERMAL_PROPS.items():
        h_fh_w_k = 1000.0 / r_fh_k_kw
        tau_s = (c_floor_wh_k * 3600.0) / h_fh_w_k
        alpha_300 = math.exp(-300.0 / tau_s)
        upward_share = (
            1.0 - priors.BETA_DOWN_1E_LEAK
            if g in _LEAKING_1E_GROUPS
            else 1.0
        )
        specs.append(
            (
                g,
                tau_s,
                alpha_300,
                1.0 - alpha_300,
                upward_share / h_fh_w_k,
                f"t_air_{g}_c",
            )
        )
    return tuple(specs)


_PROXY_SPECS: tuple[tuple[str, float, float, float, float, str], ...] = (
    _build_proxy_specs()
)
_NUM_PROXY_GROUPS: int = len(_PROXY_SPECS)


def scale_k_hx_eps_ntu(
    k_hx_old: float,
    flow_old_l_min: float,
    flow_new_l_min: float,
) -> float:
    """Scales floor pipe conductance `K_hx` via the epsilon-NTU law.

    Args:
        k_hx_old: Existing effective pipe conductance [W/K].
        flow_old_l_min: Previous group rotameter flow rate [L/min].
        flow_new_l_min: New group rotameter flow rate [L/min].

    Returns:
        Physically scaled pipe conductance `K_hx` [W/K].
    """
    if (
        flow_old_l_min <= 0.0
        or flow_new_l_min <= 0.0
        or abs(flow_new_l_min - flow_old_l_min) < 1e-6
    ):
        return float(k_hx_old)
    c_old = flow_old_l_min * priors.WATER_POWER_FACTOR_W_PER_LMIN_K
    c_new = flow_new_l_min * priors.WATER_POWER_FACTOR_W_PER_LMIN_K
    eps_old = min(0.95, max(1e-4, k_hx_old / max(1e-3, c_old)))
    ua_pipe = -c_old * math.log(1.0 - eps_old)
    return c_new * (1.0 - math.exp(-ua_pipe / max(1e-3, c_new)))


def _is_valid_num(val: float | int | None) -> TypeGuard[float | int]:
    """Checks whether `val` is a finite `float` or `int`."""
    return val is not None and math.isfinite(val)


@dataclasses.dataclass(frozen=True, slots=True)
class HydraulicStepResult:
    """Outputs of a single waterside hydraulic energy-balance step.

    Attributes:
        q_fh_by_group_w: Thermal power delivered to each of the 10 groups [W].
        q_fh_wk_w: Thermal power delivered to the living-room loop (`wk`) [W].
        q_fh_kt_w: Thermal power delivered to the attic office (`kantoor`) [W].
        q_leak_1e_to_wk_w: Downward ceiling heat leak into `wk` from active
            1st-floor loops (`slaapkamer`, `kamer_1e_zuid`, `werkkamer_1e`) [W].
        t_sup_sec_c: Effective secondary manifold supply temperature [°C].
        q_sec_calorimetric_w: Single-step 200 L buffer calorimetric power [W].
        gamma_cal: Active 30-min telescoped calorimetric scale factor [-].
        f_flow_wk: Rotameter flow fraction of `wk` among open loops [-].
        f_flow_kt: Rotameter flow fraction of `kantoor` among open loops [-].
    """

    q_fh_by_group_w: dict[str, float]
    q_fh_wk_w: float
    q_fh_kt_w: float
    q_leak_1e_to_wk_w: float
    t_sup_sec_c: float
    q_sec_calorimetric_w: float
    gamma_cal: float
    f_flow_wk: float
    f_flow_kt: float


class HydraulicModel:
    """Uniform epsilon-NTU (`K_hx`) floor distribution with 30-min watcher.

    Attributes:
        flow_rates_l_min: Configured rotameter flow rates per group [L/min].
        k_hx_by_group_w_k: Effective pipe heat-transfer conductance [W/K] per
            hydraulic group.
        t_floor_proxy_c: Lightweight 1-node slab temperature estimates [°C] for
            the 8 unmodeled hydraulic groups.
        gamma_cal: Telescoped 30-minute calorimetric energy ratio clamped to
            `[0.75, 1.25]`.
    """

    @staticmethod
    def _lookup_group_val(
        mapping: Mapping[str, float] | None,
        group: str,
    ) -> float | None:
        """Resolves a group value from `mapping`, accepting `kt` for `kantoor`."""
        if mapping is None:
            return None
        if group in mapping and _is_valid_num(mapping[group]):
            return float(mapping[group])
        if (
            group == "kantoor"
            and "kt" in mapping
            and _is_valid_num(mapping["kt"])
        ):
            return float(mapping["kt"])
        return None

    def __init__(
        self,
        flow_rates_l_min: Mapping[str, float] | None = None,
        k_hx_by_group_w_k: Mapping[str, float] | None = None,
    ) -> None:
        """Initializes the hydraulic model from rotameter flows and `K_hx` priors.

        Args:
            flow_rates_l_min: Optional mapping of group rotameter flows [L/min].
            k_hx_by_group_w_k: Optional mapping of group `K_hx` values [W/K].
        """
        self.flow_rates_l_min: dict[str, float] = {
            g: max(
                0.05,
                v
                if (v := self._lookup_group_val(flow_rates_l_min, g))
                is not None
                else const.DEFAULT_FLOW_RATES_L_MIN[g],
            )
            for g in const.ALL_HYDRAULIC_GROUPS
        }
        if k_hx_by_group_w_k is not None:
            self.k_hx_by_group_w_k: dict[str, float] = {
                g: max(
                    1.0,
                    v
                    if (v := self._lookup_group_val(k_hx_by_group_w_k, g))
                    is not None
                    else scale_k_hx_eps_ntu(
                        priors.DAY1_K_HX_BY_GROUP_W_K[g],
                        const.DEFAULT_FLOW_RATES_L_MIN[g],
                        self.flow_rates_l_min[g],
                    ),
                )
                for g in const.ALL_HYDRAULIC_GROUPS
            }
        else:
            self.k_hx_by_group_w_k = {
                g: max(
                    1.0,
                    scale_k_hx_eps_ntu(
                        priors.DAY1_K_HX_BY_GROUP_W_K[g],
                        const.DEFAULT_FLOW_RATES_L_MIN[g],
                        self.flow_rates_l_min[g],
                    ),
                )
                for g in const.ALL_HYDRAULIC_GROUPS
            }

        self._proxy_specs = _PROXY_SPECS
        self.t_floor_proxy_c: dict[str, float] = {
            g: 19.0 for g in priors.UNMODELED_GROUP_THERMAL_PROPS
        }
        self._proxy_initialized: set[str] = set()
        self.gamma_cal: float = 1.0
        # Window entries: (e_prim_j, t_buf_mean_start_c, t_buf_mean_end_c, e_c2_j).
        self._cal_window: collections.deque[
            tuple[float, float, float, float]
        ] = collections.deque(maxlen=TELESCOPE_STEPS)

    def update_flow_rates(self, flow_rates_l_min: Mapping[str, float]) -> None:
        """Updates rotameter flow rates [L/min] and scales `k_hx` via eps-NTU.

        Args:
            flow_rates_l_min: Mapping from hydraulic group name to new [L/min].
        """
        for g in const.ALL_HYDRAULIC_GROUPS:
            val = self._lookup_group_val(flow_rates_l_min, g)
            if val is not None:
                old_flow = max(0.05, self.flow_rates_l_min[g])
                new_flow = max(0.05, val)
                self.k_hx_by_group_w_k[g] = max(
                    1.0,
                    scale_k_hx_eps_ntu(
                        self.k_hx_by_group_w_k[g], old_flow, new_flow
                    ),
                )
                self.flow_rates_l_min[g] = new_flow
        self._cal_window.clear()

    def update_k_hx(self, k_hx_by_group_w_k: Mapping[str, float]) -> None:
        """Applies calibrated `K_hx` [W/K] conductances per hydraulic group.

        Args:
            k_hx_by_group_w_k: Mapping from hydraulic group name to `K_hx` [W/K].
        """
        for g in const.ALL_HYDRAULIC_GROUPS:
            val = self._lookup_group_val(k_hx_by_group_w_k, g)
            if val is not None:
                self.k_hx_by_group_w_k[g] = max(1.0, val)
        self._cal_window.clear()

    def compute_effective_supply_temp(
        self,
        threeway_room: bool,
        v_prim_l_min: float,
        v_sec_total_l_min: float,
        t_buf_top_c: float,
        t_buf_bot_raw_c: float,
    ) -> float:
        """Computes secondary manifold supply temperature across hydraulic regimes.

        Applies `BUF_BOT_SENSOR_OFFSET_C` (`-0.33 °C`) to the raw bottom sensor.
        Accounts for bottom return-water bypass mixing when `threeway_room` is
        True and `v_sec_total_l_min > v_prim_l_min > 1.0`, and uses
        `t_buf_top_c` directly during passive buffer harvesting.

        Args:
            threeway_room: True when the primary 3-way valve is set to `Room`.
            v_prim_l_min: Primary heat-pump water flow rate [L/min].
            v_sec_total_l_min: Total active secondary manifold flow [L/min].
            t_buf_top_c: 200 L buffer tank top sensor temperature [°C].
            t_buf_bot_raw_c: Uncalibrated 200 L buffer bottom temperature [°C].

        Returns:
            Effective secondary supply water temperature [°C].
        """
        t_bot_cal_c = float(t_buf_bot_raw_c) + priors.BUF_BOT_SENSOR_OFFSET_C
        v_prim = (
            0.0
            if not _is_valid_num(v_prim_l_min)
            else max(0.0, float(v_prim_l_min))
        )
        v_sec = (
            0.0
            if not _is_valid_num(v_sec_total_l_min)
            else max(0.0, float(v_sec_total_l_min))
        )

        if threeway_room and v_prim > 1.0:
            alpha_mix = min(1.0, v_prim / max(v_sec, 0.1))
            return (
                alpha_mix * float(t_buf_top_c)
                + (1.0 - alpha_mix) * t_bot_cal_c
            )
        return float(t_buf_top_c)

    @staticmethod
    def _resolve_valve_openings(
        u_valves: Mapping[str, float],
    ) -> dict[str, float]:
        """Resolves effective valve openings `[0..1]` for all 10 groups."""
        bg_fallback = u_valves.get("bg", u_valves.get("u_eff_bg", 0.0))
        bg_val = (
            0.0
            if not _is_valid_num(bg_fallback)
            else max(0.0, min(1.0, float(bg_fallback)))
        )

        u_eff: dict[str, float] = {}
        for g in const.ALL_HYDRAULIC_GROUPS:
            raw = u_valves.get(g)
            if raw is None and g == "kantoor":
                raw = u_valves.get("kt")
            if raw is None:
                raw = u_valves.get(f"u_eff_{g}")
            if _is_valid_num(raw):
                u_eff[g] = max(0.0, min(1.0, float(raw)))
            elif g in ("wk", "gang"):
                u_eff[g] = bg_val
            else:
                u_eff[g] = 0.0
        return u_eff

    def step(
        self,
        dt_s: float,
        u_pump_sec: float,
        u_valves: Mapping[str, float],
        threeway_room: bool,
        wp_defrost: bool,
        v_prim_l_min: float,
        p_prim_th_w: float,
        f_comp_hz: float,
        t_buf_top_start_c: float,
        t_buf_top_end_c: float,
        t_buf_bot_start_c: float,
        t_buf_bot_end_c: float,
        t_floor_wk_c: float,
        t_floor_kt_c: float,
        room_temps_c: Mapping[str, float | None],
        t_bg_sup_c: float | None = None,
        t_bg_ret_c: float | None = None,
        bg_ret_offset_c: float = 0.0,
    ) -> HydraulicStepResult:
        """Advances the waterside energy balance by one ZOH step (`dt_s`).

        Args:
            dt_s: Step duration in seconds.
            u_pump_sec: Time-averaged secondary circulation pump duty `[0..1]`.
            u_valves: Effective valve openings `[0..1]` (`bg` maps to both
                `wk` and `gang`).
            threeway_room: True when the primary 3-way valve is on `Room`.
            wp_defrost: True when the primary heat pump is defrosting.
            v_prim_l_min: Time-averaged primary flow rate [L/min].
            p_prim_th_w: Sub-5m integrated primary thermal power [W].
            f_comp_hz: Time-averaged compressor frequency [Hz].
            t_buf_top_start_c: 200 L buffer top temp at step start [°C].
            t_buf_top_end_c: 200 L buffer top temp at step end [°C].
            t_buf_bot_start_c: Raw 200 L buffer bottom temp at step start [°C].
            t_buf_bot_end_c: Raw 200 L buffer bottom temp at step end [°C].
            t_floor_wk_c: Current estimated `wk` slab core temperature [°C].
            t_floor_kt_c: Current estimated `kt` slab core temperature [°C].
            room_temps_c: Measured air temperatures per room [°C].
            t_bg_sup_c: Optional clamp-on BG manifold supply sensor [°C].
            t_bg_ret_c: Optional clamp-on BG manifold return sensor [°C].
            bg_ret_offset_c: Isothermal calibration offset for `t_bg_ret_c` [K].

        Returns:
            A `HydraulicStepResult` with per-group powers, ceiling leak, and
            calorimetric diagnostics.
        """
        step_s = max(1.0, float(dt_s))
        u_pump = (
            0.0
            if not _is_valid_num(u_pump_sec)
            else max(0.0, min(1.0, float(u_pump_sec)))
        )

        # Fast path when secondary circulation pump is off (`u_pump == 0.0`).
        if u_pump == 0.0:
            if self._cal_window:
                self._cal_window.clear()
            proxies = self.t_floor_proxy_c
            inited = self._proxy_initialized
            need_seed = len(inited) < _NUM_PROXY_GROUPS
            is_300 = step_s == 300.0
            for g, tau_s, a300, oma300, _, alt_key in self._proxy_specs:
                r_air = room_temps_c.get(g)
                if r_air is None:
                    r_air = room_temps_c.get(alt_key)
                if r_air is not None and math.isfinite(r_air):
                    t_air_g = float(r_air)
                    if need_seed and g not in inited:
                        proxies[g] = t_air_g
                        inited.add(g)
                    elif is_300:
                        proxies[g] = a300 * proxies[g] + oma300 * t_air_g
                    else:
                        alpha = math.exp(-step_s / tau_s)
                        proxies[g] = (
                            alpha * proxies[g] + (1.0 - alpha) * t_air_g
                        )

            return HydraulicStepResult(
                q_fh_by_group_w=_ZERO_GROUP_POWERS.copy(),
                q_fh_wk_w=0.0,
                q_fh_kt_w=0.0,
                q_leak_1e_to_wk_w=0.0,
                t_sup_sec_c=0.5
                * (float(t_buf_top_start_c) + float(t_buf_top_end_c)),
                q_sec_calorimetric_w=0.0,
                gamma_cal=self.gamma_cal,
                f_flow_wk=0.0,
                f_flow_kt=0.0,
            )

        u_eff = self._resolve_valve_openings(u_valves)

        # Seed unmodeled slab proxies from first valid room air measurement.
        proxies = self.t_floor_proxy_c
        inited = self._proxy_initialized
        if len(inited) < _NUM_PROXY_GROUPS:
            for g, _, _, _, _, alt_key in self._proxy_specs:
                if g not in inited:
                    r_val = room_temps_c.get(g)
                    if r_val is None:
                        r_val = room_temps_c.get(alt_key)
                    if _is_valid_num(r_val):
                        proxies[g] = float(r_val)
                        inited.add(g)

        weighted_flows = {
            g: u_eff[g] * self.flow_rates_l_min[g]
            for g in const.ALL_HYDRAULIC_GROUPS
        }
        sum_weighted_flow = sum(weighted_flows.values())
        v_sec_total_l_min = u_pump * sum_weighted_flow

        if u_pump > 1e-3 and sum_weighted_flow > 1e-6:
            f_flow_wk = weighted_flows["wk"] / sum_weighted_flow
            f_flow_kt = weighted_flows["kantoor"] / sum_weighted_flow
        else:
            f_flow_wk = 0.0
            f_flow_kt = 0.0

        t_buf_top_mid_c = 0.5 * (
            float(t_buf_top_start_c) + float(t_buf_top_end_c)
        )
        t_buf_bot_raw_mid_c = 0.5 * (
            float(t_buf_bot_start_c) + float(t_buf_bot_end_c)
        )
        t_sup_sec_c = self.compute_effective_supply_temp(
            threeway_room=threeway_room,
            v_prim_l_min=v_prim_l_min,
            v_sec_total_l_min=v_sec_total_l_min,
            t_buf_top_c=t_buf_top_mid_c,
            t_buf_bot_raw_c=t_buf_bot_raw_mid_c,
        )

        # 200 L buffer tank calorimetric balance from window boundary values.
        bot_off = priors.BUF_BOT_SENSOR_OFFSET_C
        t_buf_mean_start = 0.5 * (
            float(t_buf_top_start_c) + float(t_buf_bot_start_c) + bot_off
        )
        t_buf_mean_end = 0.5 * (
            float(t_buf_top_end_c) + float(t_buf_bot_end_c) + bot_off
        )
        d_e_buf_w = (
            priors.C_BUF_J_K * (t_buf_mean_end - t_buf_mean_start) / step_s
        )
        p_prim_clean = (
            0.0 if not _is_valid_num(p_prim_th_w) else float(p_prim_th_w)
        )
        p_prim_into_buf_w = p_prim_clean if threeway_room else 0.0
        q_sec_calorimetric_w = (
            (p_prim_into_buf_w - d_e_buf_w) if u_pump > 0.01 else 0.0
        )

        # Unscaled epsilon-NTU group heat transfers.
        q_c2_by_group: dict[str, float] = {}
        for g in const.ALL_HYDRAULIC_GROUPS:
            if g == "wk":
                t_fl = float(t_floor_wk_c)
            elif g == "kantoor":
                t_fl = float(t_floor_kt_c)
            else:
                t_fl = self.t_floor_proxy_c[g]
            q_c2_by_group[g] = (
                u_pump
                * u_eff[g]
                * self.k_hx_by_group_w_k[g]
                * (t_sup_sec_c - t_fl)
            )

        sum_q_c2_w = sum(q_c2_by_group.values())
        f_hz = 0.0 if not _is_valid_num(f_comp_hz) else float(f_comp_hz)

        # Telescoped 30-min (6-step) calorimetric watcher during steady CV runs.
        steady_cv_step = (
            threeway_room
            and not wp_defrost
            and f_hz >= 15.0
            and u_pump >= 0.90
            and sum_q_c2_w >= 300.0
        )
        if steady_cv_step:
            self._cal_window.append(
                (
                    p_prim_into_buf_w * step_s,
                    t_buf_mean_start,
                    t_buf_mean_end,
                    sum_q_c2_w * step_s,
                )
            )
            if len(self._cal_window) == TELESCOPE_STEPS:
                e_prim_j = sum(item[0] for item in self._cal_window)
                d_e_buf_telescoped_j = priors.C_BUF_J_K * (
                    self._cal_window[-1][2] - self._cal_window[0][1]
                )
                e_sec_cal_j = e_prim_j - d_e_buf_telescoped_j
                e_c2_j = sum(item[3] for item in self._cal_window)
                if e_c2_j > 1.0e5:
                    raw_gamma = e_sec_cal_j / e_c2_j
                    self.gamma_cal = max(
                        GAMMA_CAL_MIN, min(GAMMA_CAL_MAX, raw_gamma)
                    )
        else:
            # Hold gamma_cal during buffer harvesting, DHW runs, or defrosts.
            self._cal_window.clear()

        q_fh_by_group_w: dict[str, float] = {
            g: self.gamma_cal * q_c2_by_group[g]
            for g in const.ALL_HYDRAULIC_GROUPS
        }

        # Optional direct BG manifold clamp-on 1-Wire override (wk + gang).
        if (
            u_pump > 0.01
            and _is_valid_num(t_bg_sup_c)
            and _is_valid_num(t_bg_ret_c)
        ):
            ret_off = (
                0.0
                if not _is_valid_num(bg_ret_offset_c)
                else float(bg_ret_offset_c)
            )
            t_bg_ret_cal = float(t_bg_ret_c) + ret_off
            v_wk = u_pump * u_eff["wk"] * self.flow_rates_l_min["wk"]
            v_gang = u_pump * u_eff["gang"] * self.flow_rates_l_min["gang"]
            v_bg = v_wk + v_gang
            if v_bg > 1e-6:
                q_bg_total_w = (
                    priors.WATER_POWER_FACTOR_W_PER_LMIN_K
                    * v_bg
                    * (float(t_bg_sup_c) - t_bg_ret_cal)
                )
                q_fh_by_group_w["wk"] = q_bg_total_w * (v_wk / v_bg)
                q_fh_by_group_w["gang"] = q_bg_total_w * (v_gang / v_bg)
            else:
                q_fh_by_group_w["wk"] = 0.0
                q_fh_by_group_w["gang"] = 0.0

        # Downward thermal leak from active 1st-floor loops above living room.
        q_leak_1e_to_wk_w = priors.BETA_DOWN_1E_LEAK * (
            q_fh_by_group_w["slaapkamer"]
            + q_fh_by_group_w["kamer_1e_zuid"]
            + q_fh_by_group_w["werkkamer_1e"]
        )

        # Advance 1-node slab proxies for the 8 unmodeled groups via exact ZOH.
        is_300 = step_s == 300.0
        for g, tau_s, a300, oma300, share_over_h, alt_key in self._proxy_specs:
            r_air = room_temps_c.get(g)
            if r_air is None:
                r_air = room_temps_c.get(alt_key)
            t_cur = proxies[g]
            t_air_g = float(r_air) if _is_valid_num(r_air) else t_cur
            t_eq_g = t_air_g + share_over_h * q_fh_by_group_w[g]
            if is_300:
                proxies[g] = a300 * t_cur + oma300 * t_eq_g
            else:
                alpha = math.exp(-step_s / tau_s)
                proxies[g] = alpha * t_cur + (1.0 - alpha) * t_eq_g

        return HydraulicStepResult(
            q_fh_by_group_w=q_fh_by_group_w,
            q_fh_wk_w=q_fh_by_group_w["wk"],
            q_fh_kt_w=q_fh_by_group_w["kantoor"],
            q_leak_1e_to_wk_w=q_leak_1e_to_wk_w,
            t_sup_sec_c=t_sup_sec_c,
            q_sec_calorimetric_w=q_sec_calorimetric_w,
            gamma_cal=self.gamma_cal,
            f_flow_wk=f_flow_wk,
            f_flow_kt=f_flow_kt,
        )

    def estimate_total_sink_capacity_w(
        self,
        t_supply_sec_c: float,
        open_zones: Iterable[str] | Mapping[str, float],
        t_floor_wk_c: float,
        t_floor_kt_c: float,
    ) -> float:
        """Estimates waterside sink absorption [W] for anti-short-cycling checks.

        Args:
            t_supply_sec_c: Candidate secondary supply water temperature [°C].
            open_zones: Iterable of open zone/group identifiers or mapping of
                `{zone_or_group: u_eff}` (`bg` expands to `wk` and `gang`;
                `kt` maps to `kantoor`).
            t_floor_wk_c: Current `wk` slab core temperature [°C].
            t_floor_kt_c: Current `kt` slab core temperature [°C].

        Returns:
            Total instantaneous heat absorption capacity [W] across open loops.
        """
        weights: dict[str, float] = {}
        if isinstance(open_zones, Mapping):
            for z, raw_u in open_zones.items():
                if not _is_valid_num(raw_u):
                    continue
                u_val = max(0.0, min(1.0, float(raw_u)))
                if u_val <= 0.0:
                    continue
                if z == "bg":
                    weights["wk"] = max(weights.get("wk", 0.0), u_val)
                    weights["gang"] = max(weights.get("gang", 0.0), u_val)
                elif z == "kt":
                    weights["kantoor"] = max(weights.get("kantoor", 0.0), u_val)
                elif z in const.ALL_HYDRAULIC_GROUPS:
                    weights[z] = max(weights.get(z, 0.0), u_val)
        else:
            for z in open_zones:
                if z == "bg":
                    weights["wk"] = 1.0
                    weights["gang"] = 1.0
                elif z == "kt":
                    weights["kantoor"] = 1.0
                elif z in const.ALL_HYDRAULIC_GROUPS:
                    weights[z] = 1.0

        total_w = 0.0
        t_sup = float(t_supply_sec_c)
        for g, u_val in weights.items():
            if g == "wk":
                t_fl = float(t_floor_wk_c)
            elif g == "kantoor":
                t_fl = float(t_floor_kt_c)
            else:
                t_fl = self.t_floor_proxy_c[g]
            total_w += (
                u_val
                * self.gamma_cal
                * self.k_hx_by_group_w_k[g]
                * max(0.0, t_sup - t_fl)
            )
        return total_w

    def to_dict(self) -> dict[str, Any]:
        """Serializes hydraulic model state to a JSON-compatible dictionary.

        Returns:
            Dictionary suitable for Home Assistant `Store` persistence.
        """
        return {
            "flow_rates_l_min": dict(self.flow_rates_l_min),
            "k_hx_by_group_w_k": dict(self.k_hx_by_group_w_k),
            "t_floor_proxy_c": dict(self.t_floor_proxy_c),
            "gamma_cal": float(self.gamma_cal),
            "proxy_initialized": sorted(self._proxy_initialized),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        flow_rates_l_min: Mapping[str, float] | None = None,
        k_hx_by_group_w_k: Mapping[str, float] | None = None,
    ) -> HydraulicModel:
        """Restores a `HydraulicModel` from a persisted dictionary.

        Args:
            data: Persisted dictionary produced by `to_dict()`.
            flow_rates_l_min: Optional override for rotameter flows [L/min].
            k_hx_by_group_w_k: Optional override for `K_hx` conductances [W/K].

        Returns:
            Restored `HydraulicModel` instance.
        """
        saved_flows = data.get("flow_rates_l_min")
        saved_khx = data.get("k_hx_by_group_w_k")
        eff_flows = (
            flow_rates_l_min if flow_rates_l_min is not None else saved_flows
        )
        eff_khx = (
            k_hx_by_group_w_k if k_hx_by_group_w_k is not None else saved_khx
        )

        model = cls(flow_rates_l_min=eff_flows, k_hx_by_group_w_k=eff_khx)

        saved_proxies = data.get("t_floor_proxy_c")
        if isinstance(saved_proxies, Mapping):
            for g in priors.UNMODELED_GROUP_THERMAL_PROPS:
                if g in saved_proxies and _is_valid_num(saved_proxies[g]):
                    model.t_floor_proxy_c[g] = float(saved_proxies[g])
                    model._proxy_initialized.add(g)

        init_list = data.get("proxy_initialized")
        if isinstance(init_list, list):
            for g in init_list:
                if g in priors.UNMODELED_GROUP_THERMAL_PROPS:
                    model._proxy_initialized.add(str(g))

        gamma = data.get("gamma_cal")
        if _is_valid_num(gamma):
            model.gamma_cal = max(
                GAMMA_CAL_MIN, min(GAMMA_CAL_MAX, float(gamma))
            )

        return model
