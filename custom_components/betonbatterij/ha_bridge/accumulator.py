"""Event-driven sub-5-minute time-weighted accumulator (Component 5 §2).

Core classes (`StepSignalAccumulator`, `WaxValveTracker`,
`HydraulicPowerAccumulator`, `ClockJumpGuard`, `WindowAccumulatorCore`) have
zero top-level `homeassistant.*` imports so `pytest` runs 100% offline.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..const import (
    AC_MODE_COOL,
    AC_MODE_DRY,
    AC_MODE_HEAT,
    AC_MODE_OFF,
    AC_MODE_OTHER,
    DT_STEP_S,
    FLAG_AANWEZIG,
    FLAG_AC_COOL_KT,
    FLAG_DEFROST_AC,
    FLAG_DEFROST_WP,
    FLAG_DHW_TANK,
    FLAG_FLOOR_COOL,
    FLAG_INTERNAL_HEATER,
    FLAG_PUMP_SEC_ON,
    FLAG_PV_FALLBACK,
    FLAG_SUMMER,
    FLAG_VALID_KT,
    FLAG_VALID_WK,
    FLAG_Z1_OFF_HARVEST,
)
from ..house_config import (
    HOUSE_ENTITIES,
    ROOM_SENSORS,
    WATER_POWER_FACTOR_W_PER_LMIN_K,
    WAX_VALVE_CLOSE_TIME_S,
    WAX_VALVE_OPEN_TIME_S,
    HouseEntityConfig,
    RoomSensorConfig,
)

if TYPE_CHECKING:
    from homeassistant.core import Event, HomeAssistant, State

_INVALID_HA_STATES: frozenset[str] = frozenset(
    {"unavailable", "unknown", "none", ""}
)

ZHA_STALE_THRESHOLD_S: float = 7200.0
MIN_PLAUSIBLE_UTC_EPOCH: int = 1_704_067_200


def _parse_float(raw: Any) -> float | None:
    """Parse a finite float from a state string or numeric attribute value."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.lower() in _INVALID_HA_STATES:
            return None
        try:
            val = float(stripped)
        except ValueError:
            return None
        return val if math.isfinite(val) else None
    return None


def convert_power_to_w(
    value: float, unit: str | None, default_unit: str = "W"
) -> float:
    """Convert an electrical power reading into Watts [W].

    Args:
        value: Numeric power reading.
        unit: Reported `unit_of_measurement` attribute, if present.
        default_unit: Fallback unit (`"W"` or `"kW"`) when `unit` is `None`.

    Returns:
        Converted electrical power in Watts [W].
    """
    effective_unit = (unit or default_unit).strip().lower()
    if effective_unit == "kw":
        return value * 1000.0
    if effective_unit == "mw":
        return value * 1_000_000.0
    if effective_unit == "w":
        return value
    return value * 1000.0 if default_unit.lower() == "kw" else value


def convert_wind_speed_to_ms(
    value: float, unit: str | None, default_unit: str = "km/h"
) -> float:
    """Convert wind speed from Home Assistant weather units into [m/s].

    Args:
        value: Raw numeric wind speed.
        unit: Reported `wind_speed_unit` (`"km/h"`, `"mph"`, `"kn"`, `"m/s"`).
        default_unit: Fallback unit when `unit` is omitted (metric HA uses km/h).

    Returns:
        Wind speed in meters per second [m/s].
    """
    effective_unit = (unit or default_unit).strip().lower()
    if effective_unit in ("km/h", "kmh", "kph"):
        return value / 3.6
    if effective_unit in ("mph", "mi/h"):
        return value * 0.44704
    if effective_unit in ("kn", "kt", "knots"):
        return value * 0.514444
    if effective_unit == "ft/s":
        return value * 0.3048
    if effective_unit == "m/s":
        return value
    return value / 3.6 if default_unit.lower() == "km/h" else value


def encode_ac_mode(state_str: str | None) -> float:
    """Map a Home Assistant climate state string to RingBuffer `ac_mode_code`."""
    if state_str is None:
        return AC_MODE_OFF
    norm = state_str.strip().lower()
    if norm in ("off", "unavailable", "unknown", "none", ""):
        return AC_MODE_OFF
    if norm == "heat":
        return AC_MODE_HEAT
    if norm == "cool":
        return AC_MODE_COOL
    if norm == "dry":
        return AC_MODE_DRY
    return AC_MODE_OTHER


class StepSignalAccumulator:
    """Piecewise-constant ZOH time integrator clocked on `time.monotonic()`.

    Attributes:
        value: Latest valid numeric signal value.
        last_t: Monotonic timestamp of the most recent update.
        window_start_t: Monotonic start timestamp of the active window.
        integral: Accumulated `value * dt` integral over valid intervals.
        valid_dt: Total valid duration accumulated inside the active window [s].
        is_valid: Whether the signal currently holds a valid finite reading.
    """

    __slots__ = (
        "integral",
        "is_valid",
        "last_t",
        "valid_dt",
        "value",
        "window_start_t",
    )

    def __init__(
        self,
        initial_value: float = 0.0,
        t_mono: float = 0.0,
        is_valid: bool = False,
    ) -> None:
        """Initialize the ZOH step accumulator."""
        self.value: float = (
            float(initial_value) if math.isfinite(initial_value) else 0.0
        )
        self.last_t: float = float(t_mono)
        self.window_start_t: float = float(t_mono)
        self.integral: float = 0.0
        self.valid_dt: float = 0.0
        self.is_valid: bool = bool(is_valid and math.isfinite(initial_value))

    def update(
        self,
        new_value: float | None,
        t_mono: float,
        is_valid: bool = True,
    ) -> None:
        """Advance integral to `t_mono` under previous value, then update state."""
        dt = max(0.0, t_mono - self.last_t)
        if self.is_valid and dt > 0.0:
            self.integral += self.value * dt
            self.valid_dt += dt
        self.last_t = t_mono

        if is_valid and new_value is not None and math.isfinite(new_value):
            self.value = float(new_value)
            self.is_valid = True
        else:
            self.is_valid = False

    def close_window(
        self,
        t_mono: float,
        window_dt: float | None = None,
    ) -> tuple[float | None, float]:
        """Close the current window and reset integral accumulators.

        Args:
            t_mono: Monotonic timestamp marking window end.
            window_dt: Optional nominal window duration [s] for coverage ratio.

        Returns:
            A tuple `(time_weighted_mean, coverage)` where `coverage` lies in
            `[0.0, 1.0]` and `time_weighted_mean` is `None` if never valid.
        """
        dt = max(0.0, t_mono - self.last_t)
        if self.is_valid and dt > 0.0:
            self.integral += self.value * dt
            self.valid_dt += dt

        elapsed = (
            float(window_dt)
            if window_dt is not None and window_dt > 0.0
            else max(0.0, t_mono - self.window_start_t)
        )
        if self.valid_dt > 0.0:
            mean_val: float | None = self.integral / self.valid_dt
            coverage = (
                min(1.0, max(0.0, self.valid_dt / elapsed))
                if elapsed > 0.0
                else 1.0
            )
        elif self.is_valid:
            mean_val = self.value
            coverage = 1.0 if elapsed <= 0.0 else 0.0
        else:
            mean_val = None
            coverage = 0.0

        self.last_t = t_mono
        self.window_start_t = t_mono
        self.integral = 0.0
        self.valid_dt = 0.0
        return mean_val, coverage


class WaxValveTracker:
    """Analytical trapezoid integrator for thermal wax actuators (`228 s` stroke).

    Handles saturation at `0.0` and `1.0` as well as exact mid-stroke direction
    reversals when a valve relay switches before finishing its previous ramp.
    """

    __slots__ = (
        "close_time_s",
        "has_state",
        "integral",
        "last_t",
        "open_time_s",
        "position",
        "target_on",
        "window_start_t",
    )

    def __init__(
        self,
        initial_on: bool = False,
        t_mono: float = 0.0,
        open_time_s: float = WAX_VALVE_OPEN_TIME_S,
        close_time_s: float = WAX_VALVE_CLOSE_TIME_S,
        initial_position: float | None = None,
        has_state: bool = False,
    ) -> None:
        """Initialize the wax-actuator stroke tracker."""
        self.open_time_s: float = max(1e-6, float(open_time_s))
        self.close_time_s: float = max(1e-6, float(close_time_s))
        self.target_on: bool = bool(initial_on)
        if initial_position is not None:
            self.position: float = min(1.0, max(0.0, float(initial_position)))
        else:
            self.position = 1.0 if self.target_on else 0.0
        self.last_t: float = float(t_mono)
        self.window_start_t: float = float(t_mono)
        self.integral: float = 0.0
        self.has_state: bool = bool(has_state)

    def _advance_to(self, t_mono: float) -> None:
        dt = max(0.0, t_mono - self.last_t)
        if dt <= 0.0:
            self.last_t = max(self.last_t, t_mono)
            return

        u0 = self.position
        if self.target_on:
            t_rem = (1.0 - u0) * self.open_time_s
            if dt <= t_rem:
                u1 = min(1.0, u0 + dt / self.open_time_s)
                area = 0.5 * (u0 + u1) * dt
            else:
                u1 = 1.0
                area = 0.5 * (u0 + 1.0) * t_rem + (dt - t_rem)
        else:
            t_rem = u0 * self.close_time_s
            if dt <= t_rem:
                u1 = max(0.0, u0 - dt / self.close_time_s)
                area = 0.5 * (u0 + u1) * dt
            else:
                u1 = 0.0
                area = 0.5 * u0 * t_rem

        self.integral += area
        self.position = u1
        self.last_t = t_mono

    def set_switch(
        self,
        is_on: bool | None,
        t_mono: float,
        *,
        snap: bool = False,
    ) -> None:
        """Advance stroke to `t_mono` and apply a new binary relay state."""
        self._advance_to(t_mono)
        if is_on is None:
            self.has_state = False
            return
        self.has_state = True
        self.target_on = bool(is_on)
        if snap:
            self.position = 1.0 if self.target_on else 0.0

    def close_window(
        self,
        t_mono: float,
        window_dt: float | None = None,
    ) -> float:
        """Close `[window_start_t, t_mono]` and return mean opening `u_eff`."""
        self._advance_to(t_mono)
        denom = (
            float(window_dt)
            if window_dt is not None and window_dt > 0.0
            else (t_mono - self.window_start_t)
        )
        u_eff = (self.integral / denom) if denom > 0.0 else self.position
        u_eff = min(1.0, max(0.0, u_eff))

        self.integral = 0.0
        self.window_start_t = t_mono
        self.last_t = t_mono
        return u_eff


@dataclass(frozen=True, slots=True)
class HydraulicWindowSummary:
    """Closed-window summary returned by `HydraulicPowerAccumulator`."""

    p_prim_th_w: float
    p_prim_raw_w: float
    coverage: float
    dhw_duty: float
    t_buf_top_start_c: float
    t_buf_top_end_c: float
    t_buf_bot_start_c: float
    t_buf_bot_end_c: float


class HydraulicPowerAccumulator:
    """Coupled sub-5m primary thermal power and buffer boundary tracker.

    Integrates instantaneous `69.73 * V_prim(t) * (T_out(t) - T_in(t))` so
    correlated flow/temperature transients do not alias, while capturing exact
    start (`t_0`) and end (`t_1`) temperatures on the 200 L buffer tank.
    """

    __slots__ = (
        "dhw_dt",
        "integral_raw_w_s",
        "integral_room_w_s",
        "is_room_mode",
        "last_t",
        "power_factor",
        "t_buf_bot_end_c",
        "t_buf_bot_start_c",
        "t_buf_top_end_c",
        "t_buf_top_start_c",
        "t_in_c",
        "t_in_valid",
        "t_out_c",
        "t_out_valid",
        "total_dt",
        "v_prim_l_min",
        "v_valid",
        "valid_dt",
        "window_start_t",
    )

    def __init__(
        self,
        t_mono: float = 0.0,
        power_factor: float = WATER_POWER_FACTOR_W_PER_LMIN_K,
    ) -> None:
        """Initialize the coupled hydraulic power accumulator."""
        self.power_factor: float = float(power_factor)
        self.v_prim_l_min: float = 0.0
        self.v_valid: bool = False
        self.t_out_c: float = 0.0
        self.t_out_valid: bool = False
        self.t_in_c: float = 0.0
        self.t_in_valid: bool = False
        self.is_room_mode: bool = True

        self.last_t: float = float(t_mono)
        self.window_start_t: float = float(t_mono)
        self.integral_room_w_s: float = 0.0
        self.integral_raw_w_s: float = 0.0
        self.valid_dt: float = 0.0
        self.dhw_dt: float = 0.0
        self.total_dt: float = 0.0

        self.t_buf_top_start_c: float | None = None
        self.t_buf_top_end_c: float | None = None
        self.t_buf_bot_start_c: float | None = None
        self.t_buf_bot_end_c: float | None = None

    def _advance_to(self, t_mono: float) -> None:
        dt = max(0.0, t_mono - self.last_t)
        if dt <= 0.0:
            self.last_t = max(self.last_t, t_mono)
            return

        self.total_dt += dt
        if not self.is_room_mode:
            self.dhw_dt += dt

        if self.v_valid and self.t_out_valid and self.t_in_valid:
            inst_power_w = (
                self.power_factor
                * self.v_prim_l_min
                * (self.t_out_c - self.t_in_c)
            )
            self.integral_raw_w_s += inst_power_w * dt
            if self.is_room_mode:
                self.integral_room_w_s += inst_power_w * dt
            self.valid_dt += dt

        self.last_t = t_mono

    def update_flow(
        self, v_prim_l_min: float | None, t_mono: float, is_valid: bool = True
    ) -> None:
        """Update primary heat pump water flow [L/min] at `t_mono`."""
        self._advance_to(t_mono)
        if is_valid and v_prim_l_min is not None and math.isfinite(v_prim_l_min):
            self.v_prim_l_min = max(0.0, float(v_prim_l_min))
            self.v_valid = True
        else:
            self.v_valid = False

    def update_outlet_temp(
        self, t_out_c: float | None, t_mono: float, is_valid: bool = True
    ) -> None:
        """Update primary heat pump outlet temperature [°C] at `t_mono`."""
        self._advance_to(t_mono)
        if is_valid and t_out_c is not None and math.isfinite(t_out_c):
            self.t_out_c = float(t_out_c)
            self.t_out_valid = True
        else:
            self.t_out_valid = False

    def update_inlet_temp(
        self, t_in_c: float | None, t_mono: float, is_valid: bool = True
    ) -> None:
        """Update primary heat pump inlet temperature [°C] at `t_mono`."""
        self._advance_to(t_mono)
        if is_valid and t_in_c is not None and math.isfinite(t_in_c):
            self.t_in_c = float(t_in_c)
            self.t_in_valid = True
        else:
            self.t_in_valid = False

    def update_threeway_valve(
        self, valve_state: str | None, t_mono: float
    ) -> None:
        """Update 3-way valve routing (`"Room"` vs `"Tank"`) at `t_mono`."""
        self._advance_to(t_mono)
        if valve_state is None:
            return
        norm = valve_state.strip().lower()
        if norm in _INVALID_HA_STATES:
            return
        if "tank" in norm or "dhw" in norm:
            self.is_room_mode = False
        elif "room" in norm or "heat" in norm or norm == "0":
            self.is_room_mode = True

    def update_buffer_top(
        self, temp_c: float | None, is_valid: bool = True
    ) -> None:
        """Record 200 L buffer top sensor reading (capturing `t_0` and `t_1`)."""
        if is_valid and temp_c is not None and math.isfinite(temp_c):
            val = float(temp_c)
            if self.t_buf_top_start_c is None:
                self.t_buf_top_start_c = val
            self.t_buf_top_end_c = val

    def update_buffer_bot(
        self, temp_c: float | None, is_valid: bool = True
    ) -> None:
        """Record raw uncalibrated 200 L buffer bottom reading (`t_0`, `t_1`)."""
        if is_valid and temp_c is not None and math.isfinite(temp_c):
            val = float(temp_c)
            if self.t_buf_bot_start_c is None:
                self.t_buf_bot_start_c = val
            self.t_buf_bot_end_c = val

    def close_window(
        self,
        t_mono: float,
        window_dt: float | None = None,
    ) -> HydraulicWindowSummary:
        """Close window at `t_mono` and roll boundary buffer temps `t_1 -> t_0`."""
        self._advance_to(t_mono)
        elapsed = (
            float(window_dt)
            if window_dt is not None and window_dt > 0.0
            else max(0.0, t_mono - self.window_start_t)
        )
        if self.valid_dt > 0.0:
            p_room = self.integral_room_w_s / self.valid_dt
            p_raw = self.integral_raw_w_s / self.valid_dt
            coverage = (
                min(1.0, max(0.0, self.valid_dt / elapsed))
                if elapsed > 0.0
                else 1.0
            )
        elif self.v_valid and self.t_out_valid and self.t_in_valid:
            inst = (
                self.power_factor
                * self.v_prim_l_min
                * (self.t_out_c - self.t_in_c)
            )
            p_raw = inst
            p_room = inst if self.is_room_mode else 0.0
            coverage = 1.0 if elapsed <= 0.0 else 0.0
        else:
            p_room = 0.0
            p_raw = 0.0
            coverage = 0.0

        dhw_duty = (
            min(1.0, max(0.0, self.dhw_dt / self.total_dt))
            if self.total_dt > 0.0
            else (0.0 if self.is_room_mode else 1.0)
        )

        top_start = (
            self.t_buf_top_start_c
            if self.t_buf_top_start_c is not None
            else float("nan")
        )
        top_end = (
            self.t_buf_top_end_c
            if self.t_buf_top_end_c is not None
            else top_start
        )
        bot_start = (
            self.t_buf_bot_start_c
            if self.t_buf_bot_start_c is not None
            else float("nan")
        )
        bot_end = (
            self.t_buf_bot_end_c
            if self.t_buf_bot_end_c is not None
            else bot_start
        )

        # Carry t_1 boundary readings forward as t_0 of the next window.
        if math.isfinite(top_end):
            self.t_buf_top_start_c = top_end
            self.t_buf_top_end_c = top_end
        if math.isfinite(bot_end):
            self.t_buf_bot_start_c = bot_end
            self.t_buf_bot_end_c = bot_end

        self.integral_room_w_s = 0.0
        self.integral_raw_w_s = 0.0
        self.valid_dt = 0.0
        self.dhw_dt = 0.0
        self.total_dt = 0.0
        self.window_start_t = t_mono
        self.last_t = t_mono

        return HydraulicWindowSummary(
            p_prim_th_w=p_room,
            p_prim_raw_w=p_raw,
            coverage=coverage,
            dhw_duty=dhw_duty,
            t_buf_top_start_c=top_start,
            t_buf_top_end_c=top_end,
            t_buf_bot_start_c=bot_start,
            t_buf_bot_end_c=bot_end,
        )


class ClockJumpGuard:
    """Guard against pre-NTP epochs and NTP wall-clock jumps on RTC-less RPi 4."""

    __slots__ = (
        "last_accepted_utc",
        "last_mono",
        "min_valid_utc",
        "ntp_synced",
        "rejected_ticks",
    )

    def __init__(
        self,
        saved_wall_utc: int = 0,
        t_mono_start: float | None = None,
        min_epoch_utc: int = MIN_PLAUSIBLE_UTC_EPOCH,
    ) -> None:
        """Initialize the wall-clock jump guard."""
        self.last_accepted_utc: int = max(0, int(saved_wall_utc))
        self.min_valid_utc: int = max(int(min_epoch_utc), self.last_accepted_utc)
        self.last_mono: float | None = (
            float(t_mono_start) if t_mono_start is not None else None
        )
        self.ntp_synced: bool = False
        self.rejected_ticks: int = 0

    def reset_monotonic(self, t_mono: float) -> None:
        """Anchor the monotonic clock reference."""
        self.last_mono = float(t_mono)

    def record_bridged_utc(self, bridged_utc: int, t_mono: float) -> None:
        """Advance accepted UTC floor after restart-gap bridging or backfill."""
        if bridged_utc > self.last_accepted_utc and bridged_utc >= self.min_valid_utc:
            self.last_accepted_utc = int(bridged_utc)
            self.min_valid_utc = max(self.min_valid_utc, self.last_accepted_utc)
            self.ntp_synced = True
        self.last_mono = float(t_mono)

    def check_and_advance(
        self, wall_utc: float | int, t_mono: float
    ) -> tuple[bool, int, float]:
        """Validate `wall_utc` and compute elapsed step duration from `t_mono`.

        Args:
            wall_utc: Candidate UTC epoch timestamp at tick boundary.
            t_mono: Current `time.monotonic()` reading.

        Returns:
            A tuple `(accepted, ts_utc, dt_mono_s)`.
        """
        if self.last_mono is None:
            dt_mono = float(DT_STEP_S)
        else:
            dt_mono = max(0.0, float(t_mono) - self.last_mono)
        self.last_mono = float(t_mono)

        ts_int = int(round(float(wall_utc)))
        if ts_int < self.min_valid_utc or ts_int <= self.last_accepted_utc:
            self.rejected_ticks += 1
            return False, ts_int, dt_mono

        self.ntp_synced = True
        self.last_accepted_utc = ts_int
        return True, ts_int, dt_mono


@dataclass(slots=True)
class _RoomTracker:
    """Internal per-room ZHA temperature and staleness tracker."""

    temp_c: float | None = None
    temp_valid: bool = False
    last_temp_changed_utc: float | None = None
    last_hum_changed_utc: float | None = None
    last_heard_utc: float | None = None
    temp_changed_in_window: bool = False
    hum_changed_in_window: bool = False


@dataclass(frozen=True, slots=True)
class WindowCloseResult:
    """Closed 5-minute raw row and ZHA change/staleness flags for the engine."""

    raw_row: dict[str, float | int]
    temp_changed_wk: bool
    hum_changed_wk: bool
    stale_wk: bool
    temp_changed_kt: bool
    hum_changed_kt: bool
    stale_kt: bool
    dt_s: float


class WindowAccumulatorCore:
    """Pure-Python sub-5m event accumulator for all BetonBatterij input signals."""

    def __init__(
        self,
        t_mono: float = 0.0,
        entities: HouseEntityConfig = HOUSE_ENTITIES,
        room_sensors: Mapping[str, RoomSensorConfig] = ROOM_SENSORS,
    ) -> None:
        """Initialize signal accumulators, wax trackers, and dispatch maps."""
        self.entities = entities
        self.room_sensors = dict(room_sensors)
        self.window_start_t: float = float(t_mono)

        self._valves: dict[str, WaxValveTracker] = {
            "bg": WaxValveTracker(t_mono=t_mono),
            "slaapkamer_main": WaxValveTracker(t_mono=t_mono),
            "slaapkamer_voor": WaxValveTracker(t_mono=t_mono),
            "slaapkamer_achter": WaxValveTracker(t_mono=t_mono),
            "badkamer": WaxValveTracker(t_mono=t_mono),
            "werkkamer_1e": WaxValveTracker(t_mono=t_mono),
            "kamer_1e_midden": WaxValveTracker(t_mono=t_mono),
            "kamer_1e_zuid": WaxValveTracker(t_mono=t_mono),
            "kantoor": WaxValveTracker(t_mono=t_mono),
            "badkamer_boven": WaxValveTracker(t_mono=t_mono),
            "logeerkamer": WaxValveTracker(t_mono=t_mono),
        }

        self.hydraulics = HydraulicPowerAccumulator(t_mono=t_mono)

        self._signals: dict[str, StepSignalAccumulator] = {
            name: StepSignalAccumulator(t_mono=t_mono)
            for name in (
                "u_pump_sec",
                "presence",
                "defrost_wp",
                "internal_heater",
                "defrost_ac",
                "v_prim_l_min",
                "t_wp_out_c",
                "t_wp_in_c",
                "f_comp_hz",
                "p_wp_raw_el_w",
                "t_dhw_top_c",
                "t_dhw_bot_c",
                "p1_consumed_w",
                "p1_produced_w",
                "t_out_dak_c",
                "t_out_weather_c",
                "t_out_hm_c",
                "rh_weather_pct",
                "rh_dak_pct",
                "v_wind_ms",
                "p_pv_w",
                "p_pv_forecast_w",
                "epex_eur_kwh",
                "screen_small_pos",
                "screen_large_pos",
                "p_el_ac_kt_w",
                "p_desk_kt_w",
                "pir_occ_kt",
                "p_growlights_kt_w",
                "t_bg_sup_c",
                "t_bg_ret_c",
                "t_floor_surf_meas_wk_c",
                "ac_mode_heat",
                "ac_mode_cool",
                "ac_mode_dry",
                "ac_mode_other",
            )
        }

        self._rooms: dict[str, _RoomTracker] = {
            room: _RoomTracker() for room in self.room_sensors
        }

        self._wp_z1_state: str = "heat"
        self._wp_z1_cool_state: str = "off"
        self._wp_op_mode_state: str = "Heat+DHW"
        self._ac_mode_latest: float = AC_MODE_OFF

        self._valve_by_entity: dict[str, str] = {
            entities.valve_bg: "bg",
            entities.valve_slaapkamer: "slaapkamer_main",
            entities.valve_slaapkamer_voor: "slaapkamer_voor",
            entities.valve_slaapkamer_achter: "slaapkamer_achter",
            entities.valve_badkamer: "badkamer",
            entities.valve_werkkamer_1e: "werkkamer_1e",
            entities.valve_kamer_1e_midden: "kamer_1e_midden",
            entities.valve_kamer_1e_zuid: "kamer_1e_zuid",
            entities.valve_kantoor: "kantoor",
            entities.valve_badkamer_boven: "badkamer_boven",
            entities.valve_logeerkamer: "logeerkamer",
        }
        self._temp_room_by_entity: dict[str, str] = {}
        self._preswap_room_by_entity: dict[str, tuple[str, int]] = {}
        self._hum_room_by_entity: dict[str, str] = {}
        for room_key, cfg in self.room_sensors.items():
            self._temp_room_by_entity[cfg.temp_entity_id] = room_key
            if cfg.humidity_entity_id:
                self._hum_room_by_entity[cfg.humidity_entity_id] = room_key
            if cfg.pre_swap_temp_entity_id and cfg.swap_timestamp_utc is not None:
                self._preswap_room_by_entity[cfg.pre_swap_temp_entity_id] = (
                    room_key,
                    cfg.swap_timestamp_utc,
                )

    def watched_entity_ids(self) -> list[str]:
        """Return deduplicated list of all Home Assistant entity IDs tracked."""
        e = self.entities
        ids: list[str] = [
            e.presence_boolean,
            e.wp_z1_climate,
            e.wp_z1_cooling_climate,
            e.wp_operating_mode,
            e.wp_threeway_valve,
            e.wp_defrosting,
            e.wp_internal_heater,
            e.wp_pump_flow,
            e.wp_outlet_temp,
            e.wp_inlet_temp,
            e.wp_compressor_freq,
            e.wp_power_consumption,
            e.wp_outside_temp,
            e.buf_top_temp,
            e.buf_bot_temp,
            e.dhw_top_temp,
            e.dhw_bot_temp,
            e.pump_secondary_switch,
            e.valve_bg,
            e.valve_slaapkamer,
            e.valve_slaapkamer_voor,
            e.valve_slaapkamer_achter,
            e.valve_badkamer,
            e.valve_werkkamer_1e,
            e.valve_kamer_1e_midden,
            e.valve_kamer_1e_zuid,
            e.valve_kantoor,
            e.valve_badkamer_boven,
            e.valve_logeerkamer,
            e.outdoor_temp_dak,
            e.outdoor_humidity_dak,
            e.weather_entity,
            e.pv_power_w,
            e.pv_forecast_now_w,
            e.epex_price_eur_kwh,
            e.screen_small_cover,
            e.screen_large_cover,
            e.p1_power_consumed_kw,
            e.p1_power_produced_kw,
            e.ac_kantoor_climate,
            e.ac_kantoor_power_w,
            e.ac_kantoor_defrosting,
            e.ac_kantoor_occupancy,
            e.growlights_kantoor_power_w,
        ]
        for opt_id in (
            e.desk_kantoor_power_w,
            e.bg_manifold_supply_temp,
            e.bg_manifold_return_temp,
            e.floor_surface_sensor_wk,
        ):
            if opt_id:
                ids.append(opt_id)
        for cfg in self.room_sensors.values():
            ids.append(cfg.temp_entity_id)
            if cfg.humidity_entity_id:
                ids.append(cfg.humidity_entity_id)
            if cfg.pre_swap_temp_entity_id:
                ids.append(cfg.pre_swap_temp_entity_id)

        seen: set[str] = set()
        unique: list[str] = []
        for eid in ids:
            if eid and eid not in seen:
                seen.add(eid)
                unique.append(eid)
        return unique

    def ingest_state(
        self,
        entity_id: str,
        state_str: str | None,
        attributes: Mapping[str, Any] | None,
        t_mono: float,
        last_changed_utc: float | None = None,
        last_reported_utc: float | None = None,
        *,
        is_initial_seed: bool = False,
        event_wall_utc: float | None = None,
    ) -> None:
        """Process a single entity state change at monotonic timestamp `t_mono`."""
        attrs = attributes or {}
        e = self.entities
        norm_state = (state_str or "").strip().lower()
        is_avail = norm_state not in _INVALID_HA_STATES

        if entity_id in self._temp_room_by_entity:
            room_key = self._temp_room_by_entity[entity_id]
            cfg = self.room_sensors[room_key]
            if (
                cfg.swap_timestamp_utc is not None
                and event_wall_utc is not None
                and event_wall_utc < cfg.swap_timestamp_utc
            ):
                return
            self._update_room_temp(
                room_key,
                state_str,
                last_changed_utc,
                last_reported_utc,
                is_initial_seed=is_initial_seed,
            )
            return

        if entity_id in self._preswap_room_by_entity:
            room_key, swap_ts = self._preswap_room_by_entity[entity_id]
            if event_wall_utc is not None and event_wall_utc < swap_ts:
                self._update_room_temp(
                    room_key,
                    state_str,
                    last_changed_utc,
                    last_reported_utc,
                    is_initial_seed=is_initial_seed,
                )
            return

        if entity_id in self._hum_room_by_entity:
            room_key = self._hum_room_by_entity[entity_id]
            self._update_room_humidity(
                room_key,
                state_str,
                last_changed_utc,
                last_reported_utc,
                is_initial_seed=is_initial_seed,
            )
            return

        if entity_id in self._valve_by_entity:
            v_key = self._valve_by_entity[entity_id]
            if not is_avail:
                self._valves[v_key].set_switch(None, t_mono)
            else:
                self._valves[v_key].set_switch(
                    norm_state == "on", t_mono, snap=is_initial_seed
                )
            return

        if entity_id == e.pump_secondary_switch:
            self._signals["u_pump_sec"].update(
                1.0 if norm_state == "on" else 0.0, t_mono, is_valid=is_avail
            )
            return
        if entity_id == e.presence_boolean:
            self._signals["presence"].update(
                1.0 if norm_state == "on" else 0.0, t_mono, is_valid=is_avail
            )
            return
        if entity_id == e.wp_defrosting:
            self._signals["defrost_wp"].update(
                1.0 if norm_state == "on" else 0.0, t_mono, is_valid=is_avail
            )
            return
        if entity_id == e.wp_internal_heater:
            self._signals["internal_heater"].update(
                1.0 if norm_state == "on" else 0.0, t_mono, is_valid=is_avail
            )
            return
        if entity_id == e.ac_kantoor_defrosting:
            self._signals["defrost_ac"].update(
                1.0 if norm_state == "on" else 0.0, t_mono, is_valid=is_avail
            )
            return
        if entity_id == e.ac_kantoor_occupancy:
            self._signals["pir_occ_kt"].update(
                1.0 if norm_state == "on" else 0.0, t_mono, is_valid=is_avail
            )
            return

        if entity_id == e.wp_pump_flow:
            val = _parse_float(state_str)
            self._signals["v_prim_l_min"].update(val, t_mono, is_valid=val is not None)
            self.hydraulics.update_flow(val, t_mono, is_valid=val is not None)
            return
        if entity_id == e.wp_outlet_temp:
            val = _parse_float(state_str)
            self._signals["t_wp_out_c"].update(val, t_mono, is_valid=val is not None)
            self.hydraulics.update_outlet_temp(val, t_mono, is_valid=val is not None)
            return
        if entity_id == e.wp_inlet_temp:
            val = _parse_float(state_str)
            self._signals["t_wp_in_c"].update(val, t_mono, is_valid=val is not None)
            self.hydraulics.update_inlet_temp(val, t_mono, is_valid=val is not None)
            return
        if entity_id == e.wp_threeway_valve:
            self.hydraulics.update_threeway_valve(state_str, t_mono)
            return
        if entity_id == e.wp_compressor_freq:
            val = _parse_float(state_str)
            self._signals["f_comp_hz"].update(val, t_mono, is_valid=val is not None)
            return
        if entity_id == e.wp_power_consumption:
            val = _parse_float(state_str)
            unit = attrs.get("unit_of_measurement")
            val_w = convert_power_to_w(val, unit, "W") if val is not None else None
            self._signals["p_wp_raw_el_w"].update(
                val_w, t_mono, is_valid=val_w is not None
            )
            return

        if entity_id == e.buf_top_temp:
            val = _parse_float(state_str)
            self.hydraulics.update_buffer_top(val, is_valid=val is not None)
            return
        if entity_id == e.buf_bot_temp:
            val = _parse_float(state_str)
            self.hydraulics.update_buffer_bot(val, is_valid=val is not None)
            return
        if entity_id == e.dhw_top_temp:
            val = _parse_float(state_str)
            self._signals["t_dhw_top_c"].update(val, t_mono, is_valid=val is not None)
            return
        if entity_id == e.dhw_bot_temp:
            val = _parse_float(state_str)
            self._signals["t_dhw_bot_c"].update(val, t_mono, is_valid=val is not None)
            return

        if entity_id == e.p1_power_consumed_kw:
            val = _parse_float(state_str)
            unit = attrs.get("unit_of_measurement")
            val_w = convert_power_to_w(val, unit, "kW") if val is not None else None
            self._signals["p1_consumed_w"].update(
                val_w, t_mono, is_valid=val_w is not None
            )
            return
        if entity_id == e.p1_power_produced_kw:
            val = _parse_float(state_str)
            unit = attrs.get("unit_of_measurement")
            val_w = convert_power_to_w(val, unit, "kW") if val is not None else None
            self._signals["p1_produced_w"].update(
                val_w, t_mono, is_valid=val_w is not None
            )
            return

        if entity_id == e.outdoor_temp_dak:
            val = _parse_float(state_str)
            self._signals["t_out_dak_c"].update(val, t_mono, is_valid=val is not None)
            return
        if entity_id == e.outdoor_humidity_dak:
            val = _parse_float(state_str)
            self._signals["rh_dak_pct"].update(val, t_mono, is_valid=val is not None)
            return
        if entity_id == e.wp_outside_temp:
            val = _parse_float(state_str)
            self._signals["t_out_hm_c"].update(val, t_mono, is_valid=val is not None)
            return
        if entity_id == e.weather_entity:
            w_temp = _parse_float(attrs.get("temperature"))
            w_hum = _parse_float(attrs.get("humidity"))
            w_wind_raw = _parse_float(attrs.get("wind_speed"))
            w_wind_unit = attrs.get("wind_speed_unit")
            w_wind_ms = (
                convert_wind_speed_to_ms(w_wind_raw, w_wind_unit, "km/h")
                if w_wind_raw is not None
                else None
            )
            self._signals["t_out_weather_c"].update(
                w_temp, t_mono, is_valid=is_avail and w_temp is not None
            )
            self._signals["rh_weather_pct"].update(
                w_hum, t_mono, is_valid=is_avail and w_hum is not None
            )
            self._signals["v_wind_ms"].update(
                w_wind_ms, t_mono, is_valid=is_avail and w_wind_ms is not None
            )
            return
        if entity_id == e.pv_power_w:
            val = _parse_float(state_str)
            unit = attrs.get("unit_of_measurement")
            val_w = convert_power_to_w(val, unit, "W") if val is not None else None
            self._signals["p_pv_w"].update(val_w, t_mono, is_valid=val_w is not None)
            return
        if entity_id == e.pv_forecast_now_w:
            val = _parse_float(state_str)
            unit = attrs.get("unit_of_measurement")
            val_w = convert_power_to_w(val, unit, "W") if val is not None else None
            self._signals["p_pv_forecast_w"].update(
                val_w, t_mono, is_valid=val_w is not None
            )
            return
        if entity_id == e.epex_price_eur_kwh:
            val = _parse_float(state_str)
            self._signals["epex_eur_kwh"].update(val, t_mono, is_valid=val is not None)
            return
        if entity_id in (e.screen_small_cover, e.screen_large_cover):
            sig_key = (
                "screen_small_pos"
                if entity_id == e.screen_small_cover
                else "screen_large_pos"
            )
            pos = _parse_float(attrs.get("current_position"))
            if pos is None and is_avail:
                if norm_state == "open":
                    pos = 100.0
                elif norm_state == "closed":
                    pos = 0.0
            if pos is not None:
                pos = min(100.0, max(0.0, pos))
            self._signals[sig_key].update(pos, t_mono, is_valid=pos is not None)
            return

        if entity_id == e.wp_z1_climate and is_avail:
            self._wp_z1_state = norm_state
            return
        if entity_id == e.wp_z1_cooling_climate and is_avail:
            self._wp_z1_cool_state = norm_state
            return
        if entity_id == e.wp_operating_mode and is_avail:
            self._wp_op_mode_state = (state_str or "").strip()
            return
        if entity_id == e.ac_kantoor_climate:
            code = encode_ac_mode(state_str)
            self._ac_mode_latest = code
            self._signals["ac_mode_heat"].update(
                1.0 if code == AC_MODE_HEAT else 0.0, t_mono, is_valid=True
            )
            self._signals["ac_mode_cool"].update(
                1.0 if code == AC_MODE_COOL else 0.0, t_mono, is_valid=True
            )
            self._signals["ac_mode_dry"].update(
                1.0 if code == AC_MODE_DRY else 0.0, t_mono, is_valid=True
            )
            self._signals["ac_mode_other"].update(
                1.0 if code == AC_MODE_OTHER else 0.0, t_mono, is_valid=True
            )
            return
        if entity_id == e.ac_kantoor_power_w:
            val = _parse_float(state_str)
            unit = attrs.get("unit_of_measurement")
            val_w = convert_power_to_w(val, unit, "W") if val is not None else None
            self._signals["p_el_ac_kt_w"].update(
                val_w, t_mono, is_valid=val_w is not None
            )
            return
        if entity_id == e.growlights_kantoor_power_w:
            val = _parse_float(state_str)
            unit = attrs.get("unit_of_measurement")
            val_w = convert_power_to_w(val, unit, "W") if val is not None else None
            self._signals["p_growlights_kt_w"].update(
                val_w, t_mono, is_valid=val_w is not None
            )
            return

        if e.desk_kantoor_power_w and entity_id == e.desk_kantoor_power_w:
            val = _parse_float(state_str)
            unit = attrs.get("unit_of_measurement")
            val_w = convert_power_to_w(val, unit, "W") if val is not None else None
            self._signals["p_desk_kt_w"].update(
                val_w, t_mono, is_valid=val_w is not None
            )
            return
        if e.bg_manifold_supply_temp and entity_id == e.bg_manifold_supply_temp:
            val = _parse_float(state_str)
            self._signals["t_bg_sup_c"].update(val, t_mono, is_valid=val is not None)
            return
        if e.bg_manifold_return_temp and entity_id == e.bg_manifold_return_temp:
            val = _parse_float(state_str)
            adj = (
                val + e.bg_manifold_return_offset_c if val is not None else None
            )
            self._signals["t_bg_ret_c"].update(adj, t_mono, is_valid=adj is not None)
            return
        if e.floor_surface_sensor_wk and entity_id == e.floor_surface_sensor_wk:
            val = _parse_float(state_str)
            self._signals["t_floor_surf_meas_wk_c"].update(
                val, t_mono, is_valid=val is not None
            )

    def _update_room_temp(
        self,
        room_key: str,
        state_str: str | None,
        last_changed_utc: float | None,
        last_reported_utc: float | None,
        *,
        is_initial_seed: bool,
    ) -> None:
        rt = self._rooms[room_key]
        val = _parse_float(state_str)
        if val is None:
            rt.temp_valid = False
            return

        prev_val = rt.temp_c
        prev_changed_ts = rt.last_temp_changed_utc
        rt.temp_c = val
        rt.temp_valid = True

        if last_changed_utc is not None:
            if (
                not is_initial_seed
                and prev_changed_ts is not None
                and last_changed_utc > prev_changed_ts + 1e-6
            ):
                rt.temp_changed_in_window = True
            elif not is_initial_seed and prev_changed_ts is None:
                rt.temp_changed_in_window = True
            rt.last_temp_changed_utc = max(
                prev_changed_ts or 0.0, float(last_changed_utc)
            )
        elif not is_initial_seed and (
            prev_val is None or abs(val - prev_val) > 1e-6
        ):
            rt.temp_changed_in_window = True

        heard_candidates = [
            ts
            for ts in (rt.last_heard_utc, last_changed_utc, last_reported_utc)
            if ts is not None
        ]
        if heard_candidates:
            rt.last_heard_utc = max(heard_candidates)

    def _update_room_humidity(
        self,
        room_key: str,
        state_str: str | None,
        last_changed_utc: float | None,
        last_reported_utc: float | None,
        *,
        is_initial_seed: bool,
    ) -> None:
        rt = self._rooms[room_key]
        val = _parse_float(state_str)
        if val is None:
            return

        prev_hum_ts = rt.last_hum_changed_utc
        if last_changed_utc is not None:
            if (
                not is_initial_seed
                and prev_hum_ts is not None
                and last_changed_utc > prev_hum_ts + 1e-6
            ):
                rt.hum_changed_in_window = True
            elif not is_initial_seed and prev_hum_ts is None:
                rt.hum_changed_in_window = True
            rt.last_hum_changed_utc = max(
                prev_hum_ts or 0.0, float(last_changed_utc)
            )
        elif not is_initial_seed:
            rt.hum_changed_in_window = True

        heard_candidates = [
            ts
            for ts in (rt.last_heard_utc, last_changed_utc, last_reported_utc)
            if ts is not None
        ]
        if heard_candidates:
            rt.last_heard_utc = max(heard_candidates)

    def _is_room_stale(self, room_key: str, ts_utc: int) -> bool:
        rt = self._rooms[room_key]
        if not rt.temp_valid or rt.temp_c is None:
            return True
        if rt.last_heard_utc is not None:
            age_s = float(ts_utc) - rt.last_heard_utc
            if age_s > ZHA_STALE_THRESHOLD_S:
                return True
        return False

    def close_window(
        self,
        ts_utc: int,
        t_mono: float,
        window_dt: float | None = None,
    ) -> WindowCloseResult:
        """Close the sub-5m window at `(ts_utc, t_mono)` and assemble `raw_row`."""
        dt_s = (
            float(window_dt)
            if window_dt is not None and window_dt > 0.0
            else max(1e-3, t_mono - self.window_start_t)
        )
        self.window_start_t = t_mono

        u_bg = self._valves["bg"].close_window(t_mono, dt_s)
        u_slk_main = self._valves["slaapkamer_main"].close_window(t_mono, dt_s)
        u_slk_voor = self._valves["slaapkamer_voor"].close_window(t_mono, dt_s)
        u_slk_achter = self._valves["slaapkamer_achter"].close_window(t_mono, dt_s)
        if self._valves["slaapkamer_main"].has_state:
            u_slaapkamer = u_slk_main
        elif (
            self._valves["slaapkamer_voor"].has_state
            and self._valves["slaapkamer_achter"].has_state
        ):
            u_slaapkamer = 0.5 * (u_slk_voor + u_slk_achter)
        elif self._valves["slaapkamer_voor"].has_state:
            u_slaapkamer = u_slk_voor
        elif self._valves["slaapkamer_achter"].has_state:
            u_slaapkamer = u_slk_achter
        else:
            u_slaapkamer = 0.0

        u_badkamer = self._valves["badkamer"].close_window(t_mono, dt_s)
        u_werkkamer_1e = self._valves["werkkamer_1e"].close_window(
            t_mono, dt_s
        )
        u_kamer_1e_midden = self._valves["kamer_1e_midden"].close_window(
            t_mono, dt_s
        )
        u_kamer_1e_zuid = self._valves["kamer_1e_zuid"].close_window(
            t_mono, dt_s
        )
        u_kantoor = self._valves["kantoor"].close_window(t_mono, dt_s)
        u_badkamer_boven = self._valves["badkamer_boven"].close_window(t_mono, dt_s)
        u_logeerkamer = self._valves["logeerkamer"].close_window(t_mono, dt_s)

        hyd = self.hydraulics.close_window(t_mono, dt_s)

        closed: dict[str, tuple[float | None, float]] = {
            key: acc.close_window(t_mono, dt_s)
            for key, acc in self._signals.items()
        }

        def _val_or(key: str, default: float, min_cov: float = 0.0) -> float:
            mean_v, cov = closed[key]
            if mean_v is not None and cov >= min_cov:
                return mean_v
            if mean_v is not None and min_cov == 0.0:
                return mean_v
            return default

        u_pump_sec = min(1.0, max(0.0, _val_or("u_pump_sec", 0.0)))
        presence_duty = _val_or("presence", 1.0)
        defrost_wp_duty = _val_or("defrost_wp", 0.0)
        internal_heater_duty = _val_or("internal_heater", 0.0)
        defrost_ac_duty = _val_or("defrost_ac", 0.0)

        t_dak_mean, t_dak_cov = closed["t_out_dak_c"]
        t_wea_mean, t_wea_cov = closed["t_out_weather_c"]
        t_hm_mean, t_hm_cov = closed["t_out_hm_c"]

        has_fallback_out = (t_wea_mean is not None and t_wea_cov >= 0.50) or (
            t_hm_mean is not None and t_hm_cov >= 0.50
        )
        if t_dak_mean is not None and (t_dak_cov >= 0.50 or not has_fallback_out):
            t_out_dak_c = t_dak_mean
        else:
            t_out_dak_c = float("nan")

        t_out_weather_c = (
            t_wea_mean
            if t_wea_mean is not None
            and (t_wea_cov >= 0.50 or t_hm_mean is None)
            else float("nan")
        )
        t_out_hm_c = t_hm_mean if t_hm_mean is not None else float("nan")
        any_outdoor_valid = any(
            math.isfinite(x) for x in (t_out_dak_c, t_out_weather_c, t_out_hm_c)
        )

        rh_wea, rh_wea_cov = closed["rh_weather_pct"]
        rh_dak, rh_dak_cov = closed["rh_dak_pct"]
        if rh_wea is not None and rh_wea_cov >= 0.50:
            rh_out_pct = rh_wea
        elif rh_dak is not None and (rh_dak_cov >= 0.50 or rh_wea is None):
            rh_out_pct = rh_dak
        elif rh_wea is not None:
            rh_out_pct = rh_wea
        else:
            rh_out_pct = 80.0

        v_wind_ms = max(0.0, _val_or("v_wind_ms", 3.0, min_cov=0.50))

        p_pv_forecast_w = max(0.0, _val_or("p_pv_forecast_w", 0.0))
        p_pv_mean, p_pv_cov = closed["p_pv_w"]
        epex_eur_kwh = _val_or("epex_eur_kwh", float("nan"))
        pv_fallback_used = False
        if p_pv_mean is not None and p_pv_cov >= 0.50:
            p_pv_w = max(0.0, p_pv_mean)
            if (
                math.isfinite(epex_eur_kwh)
                and epex_eur_kwh < 0.0
                and p_pv_forecast_w > 50.0
                and p_pv_w < 0.30 * p_pv_forecast_w
            ):
                pv_fallback_used = True
        elif p_pv_forecast_w > 0.0 and p_pv_mean is None:
            p_pv_w = p_pv_forecast_w
            pv_fallback_used = True
        else:
            p_pv_w = max(0.0, p_pv_mean or 0.0)

        p1_cons_mean, _ = closed["p1_consumed_w"]
        p1_prod_mean, _ = closed["p1_produced_w"]
        if p1_cons_mean is not None or p1_prod_mean is not None:
            p_p1_net_w = (p1_cons_mean or 0.0) - (p1_prod_mean or 0.0)
        else:
            p_p1_net_w = float("nan")

        dhw_top_acc = self._signals["t_dhw_top_c"]
        t_dhw_top_c = (
            dhw_top_acc.value
            if dhw_top_acc.is_valid
            else _val_or("t_dhw_top_c", float("nan"))
        )
        dhw_bot_acc = self._signals["t_dhw_bot_c"]
        t_dhw_bot_c = (
            dhw_bot_acc.value
            if dhw_bot_acc.is_valid
            else _val_or("t_dhw_bot_c", float("nan"))
        )

        p_el_ac_kt_w = max(0.0, _val_or("p_el_ac_kt_w", 0.0))
        ac_heat_duty = _val_or("ac_mode_heat", 0.0)
        ac_cool_duty = _val_or("ac_mode_cool", 0.0)
        ac_dry_duty = _val_or("ac_mode_dry", 0.0)
        ac_other_duty = _val_or("ac_mode_other", 0.0)
        if ac_heat_duty > 0.0 and ac_heat_duty >= max(ac_cool_duty, ac_dry_duty):
            ac_mode_code = AC_MODE_HEAT
        elif ac_cool_duty > 0.0 and ac_cool_duty >= ac_dry_duty:
            ac_mode_code = AC_MODE_COOL
        elif ac_dry_duty > 0.0:
            ac_mode_code = AC_MODE_DRY
        elif ac_other_duty >= 0.5:
            ac_mode_code = AC_MODE_OTHER
        else:
            ac_mode_code = self._ac_mode_latest

        stale_wk = self._is_room_stale("wk", ts_utc)
        stale_kt = self._is_room_stale("kantoor", ts_utc)

        rt_wk = self._rooms["wk"]
        rt_kt = self._rooms["kantoor"]
        temp_changed_wk = rt_wk.temp_changed_in_window
        hum_changed_wk = rt_wk.hum_changed_in_window
        temp_changed_kt = rt_kt.temp_changed_in_window
        hum_changed_kt = rt_kt.hum_changed_in_window

        for rt in self._rooms.values():
            rt.temp_changed_in_window = False
            rt.hum_changed_in_window = False

        def _room_temp(key: str) -> float:
            tracker = self._rooms.get(key)
            if tracker is None or not tracker.temp_valid or tracker.temp_c is None:
                return float("nan")
            return tracker.temp_c

        t_air_wk_c = _room_temp("wk")
        t_air_kt_c = _room_temp("kantoor")
        f_comp_hz = max(0.0, _val_or("f_comp_hz", 0.0))

        flags: int = 0
        if math.isfinite(t_air_wk_c) and not stale_wk and any_outdoor_valid:
            flags |= FLAG_VALID_WK
        if math.isfinite(t_air_kt_c) and not stale_kt and any_outdoor_valid:
            flags |= FLAG_VALID_KT
        if presence_duty >= 0.5:
            flags |= FLAG_AANWEZIG

        op_lower = self._wp_op_mode_state.lower()
        if op_lower == "dhw" or (
            "cool" in op_lower and "heat" not in op_lower
        ):
            flags |= FLAG_SUMMER
        if (
            ("cool" in op_lower and "auto" not in op_lower)
            or self._wp_z1_state == "cool"
            or self._wp_z1_cool_state in ("cool", "heat_cool", "auto")
        ):
            flags |= FLAG_FLOOR_COOL
        if ac_mode_code in (AC_MODE_COOL, AC_MODE_DRY):
            flags |= FLAG_AC_COOL_KT
        if hyd.dhw_duty >= 0.5:
            flags |= FLAG_DHW_TANK
        if u_pump_sec > 0.01:
            flags |= FLAG_PUMP_SEC_ON
        if (
            u_pump_sec > 0.01
            and not (flags & FLAG_DHW_TANK)
            and (self._wp_z1_state == "off" or f_comp_hz < 0.5)
        ):
            flags |= FLAG_Z1_OFF_HARVEST
        if pv_fallback_used:
            flags |= FLAG_PV_FALLBACK
        if defrost_wp_duty > 0.0:
            flags |= FLAG_DEFROST_WP
        if defrost_ac_duty > 0.0:
            flags |= FLAG_DEFROST_AC
        if internal_heater_duty > 0.0:
            flags |= FLAG_INTERNAL_HEATER

        # Preserve total primary DHW thermal power whenever FLAG_DHW_TANK is set
        # (hydraulics.py zeroes buffer injection when FLAG_DHW_TANK is set).
        p_prim_col_w = (
            hyd.p_prim_raw_w if (flags & FLAG_DHW_TANK) else hyd.p_prim_th_w
        )
        p_desk_mean, _ = closed["p_desk_kt_w"]
        p_desk_kt_w = (
            max(0.0, p_desk_mean) if p_desk_mean is not None else float("nan")
        )

        raw_row: dict[str, float | int] = {
            "ts_utc": int(ts_utc),
            "flags": int(flags),
            "u_eff_bg": u_bg,
            "u_eff_slaapkamer": u_slaapkamer,
            "u_eff_badkamer": u_badkamer,
            "u_eff_werkkamer_1e": u_werkkamer_1e,
            "u_eff_kamer_1e_midden": u_kamer_1e_midden,
            "u_eff_kamer_1e_zuid": u_kamer_1e_zuid,
            "u_eff_kantoor": u_kantoor,
            "u_eff_badkamer_boven": u_badkamer_boven,
            "u_eff_logeerkamer": u_logeerkamer,
            "u_pump_sec": u_pump_sec,
            "t_air_wk_c": t_air_wk_c,
            "t_air_slaapkamer_c": _room_temp("slaapkamer"),
            "t_air_badkamer_c": _room_temp("badkamer"),
            "t_air_werkkamer_1e_c": _room_temp("werkkamer_1e"),
            "t_air_kamer_1e_midden_c": _room_temp("kamer_1e_midden"),
            "t_air_kamer_1e_zuid_c": _room_temp("kamer_1e_zuid"),
            "t_air_kantoor_c": t_air_kt_c,
            "t_air_badkamer_boven_c": _room_temp("badkamer_boven"),
            "t_air_logeerkamer_c": _room_temp("logeerkamer"),
            "v_prim_l_min": max(0.0, _val_or("v_prim_l_min", 0.0)),
            "t_wp_out_c": _val_or("t_wp_out_c", float("nan")),
            "t_wp_in_c": _val_or("t_wp_in_c", float("nan")),
            "p_prim_th_w": p_prim_col_w,
            "f_comp_hz": f_comp_hz,
            "p_wp_raw_el_w": max(0.0, _val_or("p_wp_raw_el_w", 0.0)),
            "t_buf_top_start_c": hyd.t_buf_top_start_c,
            "t_buf_top_end_c": hyd.t_buf_top_end_c,
            "t_buf_bot_start_c": hyd.t_buf_bot_start_c,
            "t_buf_bot_end_c": hyd.t_buf_bot_end_c,
            "t_dhw_top_c": t_dhw_top_c,
            "t_dhw_bot_c": t_dhw_bot_c,
            "p_p1_net_w": p_p1_net_w,
            "t_out_dak_c": t_out_dak_c,
            "t_out_weather_c": t_out_weather_c,
            "t_out_hm_c": t_out_hm_c,
            "rh_out_pct": rh_out_pct,
            "v_wind_ms": v_wind_ms,
            "p_pv_w": p_pv_w,
            "p_pv_forecast_w": p_pv_forecast_w,
            "epex_eur_kwh": epex_eur_kwh,
            "screen_small_pos": _val_or("screen_small_pos", 100.0),
            "screen_large_pos": _val_or("screen_large_pos", 100.0),
            "p_el_ac_kt_w": p_el_ac_kt_w,
            "ac_mode_code": ac_mode_code,
            "p_desk_kt_w": p_desk_kt_w,
            "pir_occ_kt": min(1.0, max(0.0, _val_or("pir_occ_kt", 0.0))),
            "p_growlights_kt_w": max(0.0, _val_or("p_growlights_kt_w", 0.0)),
            "t_bg_sup_c": _val_or("t_bg_sup_c", float("nan")),
            "t_bg_ret_c": _val_or("t_bg_ret_c", float("nan")),
            "t_floor_surf_meas_wk_c": _val_or(
                "t_floor_surf_meas_wk_c", float("nan")
            ),
            "t_floor_wk_c": float("nan"),
            "d_est_wk_w": float("nan"),
            "q_fh_wk_w": float("nan"),
            "t_hind6h_wk_c": float("nan"),
            "t_floor_kt_c": float("nan"),
            "d_est_kt_w": float("nan"),
            "q_fh_kt_w": float("nan"),
            "t_hind6h_kt_c": float("nan"),
        }

        return WindowCloseResult(
            raw_row=raw_row,
            temp_changed_wk=temp_changed_wk,
            hum_changed_wk=hum_changed_wk,
            stale_wk=stale_wk,
            temp_changed_kt=temp_changed_kt,
            hum_changed_kt=hum_changed_kt,
            stale_kt=stale_kt,
            dt_s=dt_s,
        )


def _dt_to_utc_ts(dt_val: datetime | None) -> float | None:
    if dt_val is None:
        return None
    if dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=timezone.utc).timestamp()
    return dt_val.timestamp()


class HAWindowAccumulator:
    """Home Assistant event-bus adapter around `WindowAccumulatorCore`."""

    def __init__(
        self,
        hass: HomeAssistant,
        entities: HouseEntityConfig = HOUSE_ENTITIES,
        room_sensors: Mapping[str, RoomSensorConfig] = ROOM_SENSORS,
    ) -> None:
        """Initialize the Home Assistant state-change adapter."""
        self._hass = hass
        self._entities = entities
        self._room_sensors = dict(room_sensors)
        self.core = WindowAccumulatorCore(
            t_mono=time.monotonic(),
            entities=self._entities,
            room_sensors=self._room_sensors,
        )
        self._unsub: Callable[[], None] | None = None

    def configure(
        self,
        entities: HouseEntityConfig,
        room_sensors: Mapping[str, RoomSensorConfig],
    ) -> None:
        """Apply loaded entity and room-sensor mappings prior to `start()`."""
        self._entities = entities
        self._room_sensors = dict(room_sensors)
        self.core = WindowAccumulatorCore(
            t_mono=time.monotonic(),
            entities=self._entities,
            room_sensors=self._room_sensors,
        )

    def start(self) -> None:
        """Seed current Home Assistant states and subscribe to state events."""
        from homeassistant.core import callback
        from homeassistant.helpers.event import async_track_state_change_event

        t_now = time.monotonic()
        self.core = WindowAccumulatorCore(
            t_mono=t_now,
            entities=self._entities,
            room_sensors=self._room_sensors,
        )
        watched = self.core.watched_entity_ids()
        for entity_id in watched:
            state = self._hass.states.get(entity_id)
            if state is not None:
                self._ingest_ha_state(state, t_now, is_initial_seed=True)

        @callback
        def _on_state_event(event: Event) -> None:
            new_state: State | None = event.data.get("new_state")
            if new_state is None:
                return
            self._ingest_ha_state(
                new_state, time.monotonic(), is_initial_seed=False
            )

        self._unsub = async_track_state_change_event(
            self._hass, watched, _on_state_event
        )

    def stop(self) -> None:
        """Unsubscribe from Home Assistant state change events."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    def _ingest_ha_state(
        self, state: State, t_mono: float, *, is_initial_seed: bool
    ) -> None:
        last_changed = _dt_to_utc_ts(getattr(state, "last_changed", None))
        last_reported = _dt_to_utc_ts(
            getattr(state, "last_reported", None)
        ) or last_changed
        self.core.ingest_state(
            entity_id=state.entity_id,
            state_str=state.state,
            attributes=state.attributes,
            t_mono=t_mono,
            last_changed_utc=last_changed,
            last_reported_utc=last_reported,
            is_initial_seed=is_initial_seed,
        )

    def close_window(
        self, ts_utc: int, t_mono: float | None = None
    ) -> WindowCloseResult:
        """Close the active 5-minute window and return `WindowCloseResult`."""
        mono = time.monotonic() if t_mono is None else float(t_mono)
        return self.core.close_window(ts_utc=ts_utc, t_mono=mono)
