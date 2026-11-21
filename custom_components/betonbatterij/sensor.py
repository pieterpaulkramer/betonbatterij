"""Layer 1 SmartThrottledSensor entities and diagnostic fit sensors.

Home Assistant imports include lightweight offline fallbacks so
`SmartThrottledSensor` deadband and heartbeat logic can be unit-tested offline
without Home Assistant installed.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from . import BetonBatterijConfigEntry, build_device_info
from .const import DOMAIN
from .domain.contracts import ZoneId, ZoneSnapshot
from .house_config import DAY1_COP_PARAMS, DAY1_PRIORS_KT, DAY1_PRIORS_WK

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .ha_bridge.coordinator import BetonBatterijCoordinator

try:
    from homeassistant.components.sensor import (
        SensorDeviceClass,
        SensorEntity,
        SensorStateClass,
    )
    from homeassistant.const import (
        EntityCategory,
        UnitOfPower,
        UnitOfTemperature,
    )
    from homeassistant.core import callback
    from homeassistant.helpers.entity_platform import (
        AddConfigEntryEntitiesCallback,
    )
    from homeassistant.helpers.update_coordinator import CoordinatorEntity
except ImportError:
    try:
        from homeassistant.components.sensor import (
            SensorDeviceClass,
            SensorEntity,
            SensorStateClass,
        )
        from homeassistant.const import (
            EntityCategory,
            UnitOfPower,
            UnitOfTemperature,
        )
        from homeassistant.core import callback
        from homeassistant.helpers.entity_platform import (
            AddEntitiesCallback as AddConfigEntryEntitiesCallback,
        )
        from homeassistant.helpers.update_coordinator import CoordinatorEntity
    except ModuleNotFoundError:  # Offline pytest fallback stubs

        def callback(
            func: Callable[..., Any],
        ) -> Callable[..., Any]:  # type: ignore[misc]
            return func

        AddConfigEntryEntitiesCallback = Any  # type: ignore[assignment,misc]

        class _EnumStub:
            TEMPERATURE = "temperature"
            POWER = "power"
            MEASUREMENT = "measurement"
            DIAGNOSTIC = "diagnostic"
            CONFIG = "config"
            CELSIUS = "°C"
            WATT = "W"

        SensorDeviceClass = _EnumStub  # type: ignore[assignment,misc]
        SensorStateClass = _EnumStub  # type: ignore[assignment,misc]
        EntityCategory = _EnumStub  # type: ignore[assignment,misc]
        UnitOfTemperature = _EnumStub  # type: ignore[assignment,misc]
        UnitOfPower = _EnumStub  # type: ignore[assignment,misc]

        class CoordinatorEntity:  # type: ignore[no-redef]
            def __class_getitem__(cls, item: Any) -> type:
                del item  # Unused.
                return cls

            def __init__(self, coordinator: Any) -> None:
                self.coordinator = coordinator

            @property
            def available(self) -> bool:
                return True

            async def async_added_to_hass(self) -> None:
                return None

        class SensorEntity:  # type: ignore[no-redef]
            entity_id: str | None = None
            _attr_native_value: Any = None
            _attr_extra_state_attributes: dict[str, Any] | None = None

            @property
            def native_value(self) -> Any:
                return self._attr_native_value

            def async_write_ha_state(self) -> None:
                return None


PARALLEL_UPDATES = 0


def _sanitize_attr_value(obj: Any) -> Any:
    """Recursively replace non-finite floats (`NaN`/`Inf`) with `None`."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize_attr_value(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _sanitize_attr_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_attr_value(item) for item in obj]
    return obj


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping safely into an `orjson`-safe dict."""
    if obj is None:
        return {}
    if is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize_attr_value(asdict(obj))  # type: ignore[no-any-return]
    if isinstance(obj, Mapping):
        return _sanitize_attr_value(dict(obj))  # type: ignore[no-any-return]
    return {}


class SmartThrottledSensor(
    CoordinatorEntity["BetonBatterijCoordinator"], SensorEntity
):
    """Base sensor with deadband and heartbeat Recorder throttling.

    Publishes to the Home Assistant state machine only when:
    1. Publishing for the very first time,
    2. Transitioning to or from `unavailable` (`None` / `NaN`),
    3. Crossing to or from `0.0` (when `zero_transition_immediate=True`),
    4. `|val - last_published| >= deadband`, or
    5. Elapsed monotonic time reaches `max_interval_s` (or
       `idle_zero_max_interval_s` while remaining continuously at `0.0`).
    """

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: BetonBatterijCoordinator,
        entry_id: str,
        object_id: str,
        translation_key: str,
        *,
        deadband: float,
        max_interval_s: float,
        precision: int = 1,
        idle_zero_max_interval_s: float | None = None,
        zero_clamp_threshold: float = 0.0,
        zero_transition_immediate: bool = False,
    ) -> None:
        """Initialize the smart-throttled sensor."""
        super().__init__(coordinator)
        self.entity_id = f"sensor.{object_id}"
        self._attr_unique_id = f"{DOMAIN}_{object_id}"
        self._attr_translation_key = translation_key
        self._attr_device_info = build_device_info(entry_id)
        self._attr_suggested_display_precision = precision

        self.deadband: float = float(deadband)
        self.max_interval_s: float = float(max_interval_s)
        self.idle_zero_max_interval_s: float | None = (
            float(idle_zero_max_interval_s)
            if idle_zero_max_interval_s is not None
            else None
        )
        self.zero_clamp_threshold: float = float(zero_clamp_threshold)
        self.zero_transition_immediate: bool = bool(zero_transition_immediate)
        self.precision: int = int(precision)

        self._has_published: bool = False
        self._is_value_available: bool = True
        self._last_published_raw: float | None = None
        self._last_publish_mono: float = 0.0
        self.write_count: int = 0

    @property
    def available(self) -> bool:
        """Whether the coordinator and underlying signal are available."""
        return super().available and self._is_value_available

    def preprocess_value(self, raw_value: float | None) -> float | None:
        """Normalize non-finite values to `None` and apply zero-clamping."""
        if raw_value is None or not math.isfinite(raw_value):
            return None
        val = float(raw_value)
        if (
            self.zero_clamp_threshold > 0.0
            and abs(val) < self.zero_clamp_threshold
        ):
            return 0.0
        return val

    def evaluate_update(
        self, raw_value: float | None, t_mono: float
    ) -> bool:
        """Apply deadband and heartbeat gate; update state if triggered.

        Args:
            raw_value: Latest physical value (`None` or `NaN` if unavailable).
            t_mono: Current `time.monotonic()` timestamp [s].

        Returns:
            `True` if a state write should be emitted, `False` if suppressed.
        """
        candidate = self.preprocess_value(raw_value)

        should_write = False
        if not self._has_published:
            should_write = True
        elif (candidate is None) != (self._last_published_raw is None):
            should_write = True
        elif candidate is None or self._last_published_raw is None:
            should_write = False
        else:
            prev_raw = self._last_published_raw
            if self.zero_transition_immediate and (
                (candidate == 0.0) != (prev_raw == 0.0)
            ):
                should_write = True
            elif abs(candidate - prev_raw) >= (self.deadband - 1e-9):
                should_write = True
            else:
                effective_interval = (
                    self.idle_zero_max_interval_s
                    if (
                        self.idle_zero_max_interval_s is not None
                        and candidate == 0.0
                        and prev_raw == 0.0
                    )
                    else self.max_interval_s
                )
                if (t_mono - self._last_publish_mono) >= (
                    effective_interval - 1e-6
                ):
                    should_write = True

        if not should_write:
            return False

        self._has_published = True
        self._last_published_raw = candidate
        self._last_publish_mono = float(t_mono)
        self.write_count += 1

        if candidate is None:
            self._is_value_available = False
            self._attr_native_value = None
        else:
            self._is_value_available = True
            if self.precision <= 0:
                self._attr_native_value = int(round(candidate))
            else:
                self._attr_native_value = round(candidate, self.precision)
        return True

    def _read_raw_value(self) -> float | None:
        return None

    def _build_attributes(self) -> dict[str, Any] | None:
        return None

    async def async_added_to_hass(self) -> None:
        """Seed initial state immediately upon entity registration."""
        await super().async_added_to_hass()
        self._handle_coordinator_update()

    @callback
    def _handle_coordinator_update(self) -> None:
        raw_val = self._read_raw_value()
        attrs = self._build_attributes()
        if attrs is not None:
            self._attr_extra_state_attributes = attrs
        if self.evaluate_update(raw_val, time.monotonic()):
            self.async_write_ha_state()


class ZoneMetricThrottledSensor(SmartThrottledSensor):
    """Zone-bound telemetry sensor reading from `ZoneSnapshot` (`wk` or `kt`)."""

    _unrecorded_attributes = frozenset(
        {"t_floor_std_c", "stored_slab_energy_wh", "sensor_stale"}
    )

    def __init__(
        self,
        coordinator: BetonBatterijCoordinator,
        entry_id: str,
        zone: ZoneId,
        object_id: str,
        *,
        unit: str,
        device_class: Any,
        deadband: float,
        max_interval_s: float,
        precision: int,
        extractor: Callable[[ZoneSnapshot], float | None],
        idle_zero_max_interval_s: float | None = None,
        zero_clamp_threshold: float = 0.0,
        zero_transition_immediate: bool = False,
        attr_builder: Callable[
            [ZoneSnapshot, BetonBatterijCoordinator], dict[str, Any]
        ]
        | None = None,
    ) -> None:
        """Initialize the zone telemetry sensor."""
        super().__init__(
            coordinator,
            entry_id,
            object_id,
            object_id,
            deadband=deadband,
            max_interval_s=max_interval_s,
            precision=precision,
            idle_zero_max_interval_s=idle_zero_max_interval_s,
            zero_clamp_threshold=zero_clamp_threshold,
            zero_transition_immediate=zero_transition_immediate,
        )
        self._zone: ZoneId = zone
        self._extractor = extractor
        self._attr_builder = attr_builder
        self._attr_native_unit_of_measurement = unit
        self._attr_device_class = device_class
        self._attr_state_class = SensorStateClass.MEASUREMENT

    def _get_snapshot(self) -> ZoneSnapshot | None:
        if self.coordinator.data and self._zone in self.coordinator.data:
            return self.coordinator.data[self._zone]  # type: ignore[no-any-return]
        return self.coordinator.engine.get_zone_snapshot(self._zone)

    def _read_raw_value(self) -> float | None:
        snap = self._get_snapshot()
        return None if snap is None else self._extractor(snap)

    def _build_attributes(self) -> dict[str, Any] | None:
        if self._attr_builder is None:
            return None
        snap = self._get_snapshot()
        return None if snap is None else self._attr_builder(snap, self.coordinator)


class BetonBatterijZoneFitSensor(
    CoordinatorEntity["BetonBatterijCoordinator"], SensorEntity
):
    """Diagnostic 14-day 6h-hindcast RMSE sensor (`4x/day` + on button)."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 2
    _unrecorded_attributes = frozenset(
        {
            "parameters_active",
            "parameters_day1_prior",
            "fisher_1sigma_std",
            "correlation_matrix",
            "collinearity_warnings",
            "regime_rmses",
            "overall_6h",
            "last_calibration",
            "ringbuffer_rows",
        }
    )

    def __init__(
        self,
        coordinator: BetonBatterijCoordinator,
        entry_id: str,
        zone: ZoneId,
        object_id: str,
        translation_key: str,
    ) -> None:
        """Initialize the diagnostic zone hindcast-fit sensor."""
        super().__init__(coordinator)
        self._zone: ZoneId = zone
        self.entity_id = f"sensor.{object_id}"
        self._attr_unique_id = f"{DOMAIN}_{object_id}"
        self._attr_translation_key = translation_key
        self._attr_device_info = build_device_info(entry_id)
        self._last_fit_gen: int | None = None

    async def async_added_to_hass(self) -> None:
        """Populate initial diagnostic readout on startup."""
        await super().async_added_to_hass()
        self._refresh_from_engine(force=True)

    @callback
    def _handle_coordinator_update(self) -> None:
        self._refresh_from_engine(force=False)

    def _refresh_from_engine(self, *, force: bool) -> None:
        gen = self.coordinator.fit_sensor_generation
        if (
            not force
            and self._last_fit_gen is not None
            and gen == self._last_fit_gen
        ):
            return
        self._last_fit_gen = gen

        engine = self.coordinator.engine
        snap = engine.get_zone_snapshot(self._zone)
        report = (
            engine.eval_report_wk
            if self._zone == "wk"
            else engine.eval_report_kt
        )
        priors = DAY1_PRIORS_WK if self._zone == "wk" else DAY1_PRIORS_KT

        rmse_val: float | None = None
        overall_dict: dict[str, Any] = {}
        regime_dict: dict[str, Any] = {}
        if report is not None:
            rep_d = _to_plain_dict(report)
            overall_dict = _to_plain_dict(rep_d.get("overall_6h"))
            rmse_raw = overall_dict.get("rmse_end_c")
            if isinstance(rmse_raw, (int, float)) and math.isfinite(rmse_raw):
                rmse_val = float(rmse_raw)
            by_reg = rep_d.get("by_regime") or {}
            if isinstance(by_reg, Mapping):
                regime_dict = {
                    k: _to_plain_dict(v) for k, v in by_reg.items()
                }

        if (
            rmse_val is None
            and snap.rmse_6h_c is not None
            and math.isfinite(snap.rmse_6h_c)
        ):
            rmse_val = float(snap.rmse_6h_c)

        fisher_std: dict[str, Any] = {}
        corr_matrix: dict[str, Any] = {}
        collinearity_warnings: list[str] = []
        if hasattr(engine, "get_zone_fit_diagnostics"):
            fit_diag = _to_plain_dict(
                engine.get_zone_fit_diagnostics(self._zone)
            )
            fisher_std = dict(fit_diag.get("posterior_std") or {})
            corr_matrix = dict(fit_diag.get("correlation_matrix") or {})
            collinearity_warnings = list(
                fit_diag.get("collinearity_warnings") or []
            )

        last_cal_summary: dict[str, Any] = {}
        acc_key = f"accepted_{self._zone}"
        reason_key = f"reason_{self._zone}"
        jb_key = f"holdout_j_{self._zone}_before"
        ja_key = f"holdout_j_{self._zone}_after"
        cal_hist = getattr(engine, "calibration_history", None) or []
        for item in reversed(list(cal_hist)):
            d_item = _to_plain_dict(item)
            if acc_key in d_item:
                ev_str = str(d_item.get("event") or "")
                mode_val = d_item.get("mode") or (
                    ev_str.removeprefix("calibration_")
                    if ev_str.startswith("calibration_")
                    else ev_str
                )
                last_cal_summary = {
                    "accepted": d_item.get(acc_key),
                    "reason": d_item.get(reason_key),
                    "mode": mode_val,
                    "ts_utc": d_item.get("ts_utc"),
                    "holdout_j_before": d_item.get(jb_key),
                    "holdout_j_after": d_item.get(ja_key),
                }
                break
            if d_item.get("zone") == self._zone:
                if not fisher_std:
                    fisher_std = dict(d_item.get("posterior_std") or {})
                if not corr_matrix:
                    corr_matrix = dict(d_item.get("correlation_matrix") or {})
                if not collinearity_warnings:
                    collinearity_warnings = list(
                        d_item.get("collinearity_warnings") or []
                    )
                last_cal_summary = {
                    "accepted": d_item.get("accepted"),
                    "reason": d_item.get("reason"),
                    "mode": d_item.get("mode"),
                    "ts_utc": d_item.get("ts_utc"),
                }
                break

        self._attr_extra_state_attributes = {
            "parameters_active": _to_plain_dict(snap.params),
            "parameters_day1_prior": _to_plain_dict(priors),
            "fisher_1sigma_std": fisher_std,
            "correlation_matrix": corr_matrix,
            "collinearity_warnings": collinearity_warnings,
            "regime_rmses": regime_dict,
            "overall_6h": overall_dict,
            "last_calibration": last_cal_summary,
            "ringbuffer_rows": self.coordinator.ringbuffer.row_count,
        }
        self._attr_native_value = (
            None if rmse_val is None else round(rmse_val, 2)
        )
        self.async_write_ha_state()


class BetonBatterijCopFitSensor(
    CoordinatorEntity["BetonBatterijCoordinator"], SensorEntity
):
    """Diagnostic Panasonic 9 kW T-CAP COP curve sensor (`4x/day` + on button)."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = "COP"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 2
    _unrecorded_attributes = frozenset(
        {
            "eta_carnot_cv",
            "beta_defrost",
            "eta_carnot_dhw",
            "delta_p_wp_bias_w",
            "live_expected_cop",
            "cop_diagnostics",
        }
    )

    def __init__(
        self,
        coordinator: BetonBatterijCoordinator,
        entry_id: str,
    ) -> None:
        """Initialize the diagnostic heat-pump COP fit sensor."""
        super().__init__(coordinator)
        object_id = "betonbatterij_fit_warmtepomp_cop"
        self.entity_id = f"sensor.{object_id}"
        self._attr_unique_id = f"{DOMAIN}_{object_id}"
        self._attr_translation_key = "fit_warmtepomp_cop"
        self._attr_device_info = build_device_info(entry_id)
        self._last_fit_gen: int | None = None

    async def async_added_to_hass(self) -> None:
        """Populate initial COP readout on startup."""
        await super().async_added_to_hass()
        self._refresh_from_engine(force=True)

    @callback
    def _handle_coordinator_update(self) -> None:
        self._refresh_from_engine(force=False)

    def _compute_live_expected_cop(
        self, cop_dict: Mapping[str, Any], diag_dict: Mapping[str, Any]
    ) -> float:
        row = self.coordinator.last_raw_row
        t_out = 7.0
        for col in ("t_out_dak_c", "t_out_weather_c", "t_out_hm_c"):
            cand = row.get(col)
            if isinstance(cand, (int, float)) and math.isfinite(cand):
                t_out = float(cand)
                break

        t_sup = 30.0
        for col in ("t_wp_out_c", "t_buf_top_end_c"):
            cand = row.get(col)
            if (
                isinstance(cand, (int, float))
                and math.isfinite(cand)
                and cand > 15.0
            ):
                t_sup = float(cand)
                break

        rh_out = 80.0
        cand_rh = row.get("rh_out_pct")
        if isinstance(cand_rh, (int, float)) and math.isfinite(cand_rh):
            rh_out = float(cand_rh)

        engine = self.coordinator.engine
        if hasattr(engine, "predict_cop_cv"):
            try:
                pred = float(engine.predict_cop_cv(t_out, t_sup, rh_out))
            except (TypeError, ValueError, ZeroDivisionError):
                pred = float("nan")
            if math.isfinite(pred) and pred > 0.0:
                return pred

        for key in ("live_expected_cop", "expected_cop_cv"):
            val = diag_dict.get(key)
            if isinstance(val, (int, float)) and math.isfinite(val) and val > 0.0:
                return float(val)

        eta_cv = float(
            cop_dict.get("eta_carnot_cv", DAY1_COP_PARAMS.eta_carnot_cv)
        )
        lift_k = max(5.0, (t_sup + 2.0) - (t_out - 3.0))
        return max(1.0, min(8.0, eta_cv * ((t_sup + 275.15) / lift_k)))

    def _refresh_from_engine(self, *, force: bool) -> None:
        gen = self.coordinator.fit_sensor_generation
        if (
            not force
            and self._last_fit_gen is not None
            and gen == self._last_fit_gen
        ):
            return
        self._last_fit_gen = gen

        engine = self.coordinator.engine
        cop_params = _to_plain_dict(
            getattr(engine, "cop_params", DAY1_COP_PARAMS)
        )
        cop_diag = _to_plain_dict(getattr(engine, "cop_diagnostics", None))
        live_cop = self._compute_live_expected_cop(cop_params, cop_diag)

        eta_cv = round(
            float(cop_params.get("eta_carnot_cv", DAY1_COP_PARAMS.eta_carnot_cv)),
            4,
        )
        beta_def = round(
            float(cop_params.get("beta_defrost", DAY1_COP_PARAMS.beta_defrost)),
            4,
        )
        eta_dhw = round(
            float(
                cop_params.get("eta_carnot_dhw", DAY1_COP_PARAMS.eta_carnot_dhw)
            ),
            4,
        )
        p_bias = round(
            float(
                cop_params.get(
                    "delta_p_wp_bias_w", DAY1_COP_PARAMS.delta_p_wp_bias_w
                )
            ),
            1,
        )

        self._attr_native_value = round(live_cop, 2)
        self._attr_extra_state_attributes = {
            "eta_carnot_cv": eta_cv,
            "beta_defrost": beta_def,
            "eta_carnot_dhw": eta_dhw,
            "delta_p_wp_bias_w": p_bias,
            "live_expected_cop": round(live_cop, 2),
            "cop_diagnostics": cop_diag,
        }
        self.async_write_ha_state()


def _floor_temp_attrs(
    snap: ZoneSnapshot, coord: BetonBatterijCoordinator
) -> dict[str, Any]:
    del coord  # Unused.
    return {
        "t_surface_c": (
            round(snap.t_surface_est_c, 2)
            if math.isfinite(snap.t_surface_est_c)
            else None
        ),
        "t_floor_std_c": (
            round(snap.t_floor_std_c, 3)
            if math.isfinite(snap.t_floor_std_c)
            else None
        ),
        "stored_slab_energy_wh": (
            round(snap.stored_slab_energy_wh, 0)
            if math.isfinite(snap.stored_slab_energy_wh)
            else None
        ),
        "sensor_stale": snap.sensor_stale,
    }


def _disturbance_attrs(
    snap: ZoneSnapshot, coord: BetonBatterijCoordinator
) -> dict[str, Any]:
    del coord  # Unused.
    return {"open_window_active": snap.open_window_active}


def _hindcast_attrs(
    snap: ZoneSnapshot, coord: BetonBatterijCoordinator
) -> dict[str, Any]:
    if snap.t_hindcast_6h_c is not None:
        return {"warmup_remaining_min": 0}
    tracker = (
        getattr(coord.engine, "hindcast_wk", None)
        if snap.zone == "wk"
        else getattr(coord.engine, "hindcast_kt", None)
    )
    if tracker is not None and hasattr(tracker, "warmup_remaining_min"):
        return {"warmup_remaining_min": int(tracker.warmup_remaining_min)}
    rem_steps = max(0, 72 - coord.ringbuffer.row_count)
    return {"warmup_remaining_min": rem_steps * 5}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BetonBatterijConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Register all 13 Layer 1 BetonBatterij sensor entities."""
    del hass  # Unused.
    coordinator = entry.runtime_data
    eid = entry.entry_id

    entities: list[SensorEntity] = [
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "wk",
            "woonkamer_geschatte_vloertemperatuur",
            unit=UnitOfTemperature.CELSIUS,
            device_class=SensorDeviceClass.TEMPERATURE,
            deadband=0.05,
            max_interval_s=3600.0,
            precision=1,
            extractor=lambda s: s.state.t_floor_c,
            attr_builder=_floor_temp_attrs,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "wk",
            "woonkamer_geschatte_pvc_oppervlaktetemperatuur",
            unit=UnitOfTemperature.CELSIUS,
            device_class=SensorDeviceClass.TEMPERATURE,
            deadband=0.05,
            max_interval_s=3600.0,
            precision=1,
            extractor=lambda s: s.t_surface_est_c,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "wk",
            "woonkamer_ongeplande_warmtelast",
            unit=UnitOfPower.WATT,
            device_class=SensorDeviceClass.POWER,
            deadband=25.0,
            max_interval_s=3600.0,
            precision=0,
            extractor=lambda s: s.state.d_est_w,
            zero_clamp_threshold=15.0,
            zero_transition_immediate=True,
            attr_builder=_disturbance_attrs,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "wk",
            "woonkamer_vloer_afgiftevermogen",
            unit=UnitOfPower.WATT,
            device_class=SensorDeviceClass.POWER,
            deadband=50.0,
            max_interval_s=900.0,
            idle_zero_max_interval_s=21600.0,
            precision=0,
            extractor=lambda s: s.q_fh_now_w,
            zero_transition_immediate=True,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "wk",
            "woonkamer_hindcast_6u",
            unit=UnitOfTemperature.CELSIUS,
            device_class=SensorDeviceClass.TEMPERATURE,
            deadband=0.05,
            max_interval_s=1800.0,
            precision=1,
            extractor=lambda s: s.t_hindcast_6h_c,
            attr_builder=_hindcast_attrs,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "kt",
            "kantoor_geschatte_vloertemperatuur",
            unit=UnitOfTemperature.CELSIUS,
            device_class=SensorDeviceClass.TEMPERATURE,
            deadband=0.05,
            max_interval_s=3600.0,
            precision=1,
            extractor=lambda s: s.state.t_floor_c,
            attr_builder=_floor_temp_attrs,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "kt",
            "kantoor_geschatte_pvc_oppervlaktetemperatuur",
            unit=UnitOfTemperature.CELSIUS,
            device_class=SensorDeviceClass.TEMPERATURE,
            deadband=0.05,
            max_interval_s=3600.0,
            precision=1,
            extractor=lambda s: s.t_surface_est_c,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "kt",
            "kantoor_ongeplande_warmtelast",
            unit=UnitOfPower.WATT,
            device_class=SensorDeviceClass.POWER,
            deadband=25.0,
            max_interval_s=3600.0,
            precision=0,
            extractor=lambda s: s.state.d_est_w,
            zero_clamp_threshold=15.0,
            zero_transition_immediate=True,
            attr_builder=_disturbance_attrs,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "kt",
            "kantoor_vloer_afgiftevermogen",
            unit=UnitOfPower.WATT,
            device_class=SensorDeviceClass.POWER,
            deadband=35.0,
            max_interval_s=900.0,
            idle_zero_max_interval_s=21600.0,
            precision=0,
            extractor=lambda s: s.q_fh_now_w,
            zero_transition_immediate=True,
        ),
        ZoneMetricThrottledSensor(
            coordinator,
            eid,
            "kt",
            "kantoor_hindcast_6u",
            unit=UnitOfTemperature.CELSIUS,
            device_class=SensorDeviceClass.TEMPERATURE,
            deadband=0.05,
            max_interval_s=1800.0,
            precision=1,
            extractor=lambda s: s.t_hindcast_6h_c,
            attr_builder=_hindcast_attrs,
        ),
        BetonBatterijZoneFitSensor(
            coordinator,
            eid,
            "wk",
            "betonbatterij_fit_woonkamer",
            "fit_woonkamer",
        ),
        BetonBatterijZoneFitSensor(
            coordinator,
            eid,
            "kt",
            "betonbatterij_fit_kantoor",
            "fit_kantoor",
        ),
        BetonBatterijCopFitSensor(
            coordinator,
            eid,
        ),
    ]

    async_add_entities(entities)
