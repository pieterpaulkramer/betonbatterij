"""Layer 1 SmartThrottledSensor entities and diagnostic fit sensors.

Offline Home Assistant fallback stubs are centralized in
`ha_bridge._ha_compat`, and all thermodynamic COP / zone-fit computations
delegate to `domain/` rather than duplicating physics formulas in the entity
layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Any

from . import BetonBatterijConfigEntry, build_device_info
from .const import DOMAIN
from .domain import cop_model
from .domain.contracts import CopParameters, ZoneId, ZoneSnapshot
from .domain.priors import DAY1_COP_PARAMS
from .ha_bridge._ha_compat import (
    AddConfigEntryEntitiesCallback,
    CoordinatorEntity,
    EntityCategory,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
    UnitOfPower,
    UnitOfTemperature,
    callback,
)
from .ha_bridge.row_utils import to_plain_dict

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .ha_bridge.coordinator import BetonBatterijCoordinator

PARALLEL_UPDATES = 0


def _is_finite_number(val: Any) -> bool:
    """Return True if `val` is a finite non-bool numeric scalar."""
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(val)
    )


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


@dataclass(frozen=True, kw_only=True)
class ZoneMetricSpec:
    """Declarative specification for one throttled zone telemetry sensor."""

    zone: ZoneId
    object_id: str
    unit: str
    device_class: Any
    deadband: float
    max_interval_s: float
    precision: int
    extractor: Callable[[ZoneSnapshot], float | None]
    idle_zero_max_interval_s: float | None = None
    zero_clamp_threshold: float = 0.0
    zero_transition_immediate: bool = False
    attr_builder: (
        Callable[[ZoneSnapshot, BetonBatterijCoordinator], dict[str, Any]]
        | None
    ) = None


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
        attr_builder: (
            Callable[[ZoneSnapshot, BetonBatterijCoordinator], dict[str, Any]]
            | None
        ) = None,
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

    @classmethod
    def from_spec(
        cls,
        coordinator: BetonBatterijCoordinator,
        entry_id: str,
        spec: ZoneMetricSpec,
    ) -> ZoneMetricThrottledSensor:
        """Construct a `ZoneMetricThrottledSensor` from a `ZoneMetricSpec`."""
        return cls(
            coordinator,
            entry_id,
            spec.zone,
            spec.object_id,
            unit=spec.unit,
            device_class=spec.device_class,
            deadband=spec.deadband,
            max_interval_s=spec.max_interval_s,
            precision=spec.precision,
            extractor=spec.extractor,
            idle_zero_max_interval_s=spec.idle_zero_max_interval_s,
            zero_clamp_threshold=spec.zero_clamp_threshold,
            zero_transition_immediate=spec.zero_transition_immediate,
            attr_builder=spec.attr_builder,
        )

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
        return (
            None if snap is None else self._attr_builder(snap, self.coordinator)
        )


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
            "bound_warnings",
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
        summary = to_plain_dict(engine.get_zone_fit_summary(self._zone))
        rmse_raw = summary.get("rmse_6h_c")
        if rmse_raw is None:
            rmse_raw = summary.get("rmse_end_c")
        if rmse_raw is None:
            overall = summary.get("overall_6h")
            if isinstance(overall, Mapping):
                rmse_raw = overall.get("rmse_end_c")
        rmse_val: float | None = (
            float(rmse_raw) if _is_finite_number(rmse_raw) else None
        )
        attrs: dict[str, Any] = {
            "parameters_active": dict(summary.get("parameters_active") or {}),
            "parameters_day1_prior": dict(
                summary.get("parameters_day1_prior") or {}
            ),
            "fisher_1sigma_std": dict(
                summary.get("fisher_1sigma_std")
                or summary.get("posterior_std")
                or {}
            ),
            "correlation_matrix": dict(summary.get("correlation_matrix") or {}),
            "collinearity_warnings": list(
                summary.get("collinearity_warnings") or []
            ),
            "bound_warnings": list(summary.get("bound_warnings") or []),
            "regime_rmses": dict(summary.get("regime_rmses") or {}),
            "overall_6h": dict(summary.get("overall_6h") or {}),
            "last_calibration": dict(summary.get("last_calibration") or {}),
            "ringbuffer_rows": self.coordinator.ringbuffer.row_count,
        }

        self._attr_extra_state_attributes = attrs
        self._attr_native_value = (
            None if rmse_val is None else round(rmse_val, 2)
        )
        self.async_write_ha_state()


class BetonBatterijCopFitSensor(
    CoordinatorEntity["BetonBatterijCoordinator"], SensorEntity
):
    """Diagnostic Panasonic 9 kW T-CAP COP curve sensor.

    Recomputes the live expected space-heating COP on every 5-minute tick with
    a `0.05` deadband and 60-minute heartbeat to limit Recorder writes, while
    refreshing fitted COP curve parameters and diagnostics `4x/day` or after
    calibration.
    """

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
        self._cached_cop_params: dict[str, Any] = {}
        self._cached_cop_diag: dict[str, Any] = {}
        self._last_published_cop: float | None = None
        self._last_write_mono: float | None = None

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
        """Delegate live space-heating COP prediction strictly to `domain/`."""
        row = self.coordinator.last_raw_row
        engine = self.coordinator.engine

        if hasattr(engine, "predict_live_expected_cop"):
            try:
                pred = float(engine.predict_live_expected_cop(row))
            except (TypeError, ValueError, ZeroDivisionError):
                pred = float("nan")
            if math.isfinite(pred) and pred > 0.0:
                return pred

        if cop_dict:
            cop_params = CopParameters.from_dict(cop_dict, DAY1_COP_PARAMS)
        elif isinstance(getattr(engine, "cop_params", None), CopParameters):
            cop_params = engine.cop_params
        else:
            cop_params = DAY1_COP_PARAMS

        return float(
            cop_model.predict_live_expected_cop_from_row(
                cop_params, row, cop_diagnostics=diag_dict
            )
        )

    def _refresh_from_engine(self, *, force: bool) -> None:
        gen = self.coordinator.fit_sensor_generation
        gen_changed = self._last_fit_gen is None or gen != self._last_fit_gen
        if force or gen_changed:
            engine = self.coordinator.engine
            self._cached_cop_params = to_plain_dict(
                getattr(engine, "cop_params", DAY1_COP_PARAMS)
            )
            self._cached_cop_diag = to_plain_dict(
                getattr(engine, "cop_diagnostics", None)
            )
            self._last_fit_gen = gen

        live_cop = round(
            self._compute_live_expected_cop(
                self._cached_cop_params, self._cached_cop_diag
            ),
            2,
        )
        now_mono = time.monotonic()
        should_write = (
            force
            or gen_changed
            or self._last_published_cop is None
            or abs(live_cop - self._last_published_cop) >= (0.05 - 1e-9)
            or self._last_write_mono is None
            or (now_mono - self._last_write_mono) >= 3600.0
        )
        if not should_write:
            return

        cop_params = self._cached_cop_params
        eta_cv = round(
            float(
                cop_params.get("eta_carnot_cv", DAY1_COP_PARAMS.eta_carnot_cv)
            ),
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

        self._last_published_cop = live_cop
        self._last_write_mono = now_mono
        self._attr_native_value = live_cop
        self._attr_extra_state_attributes = {
            "eta_carnot_cv": eta_cv,
            "beta_defrost": beta_def,
            "eta_carnot_dhw": eta_dhw,
            "delta_p_wp_bias_w": p_bias,
            "live_expected_cop": live_cop,
            "cop_diagnostics": self._cached_cop_diag,
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
    del coord  # Unused.
    if snap.t_hindcast_6h_c is not None:
        return {"warmup_remaining_min": 0}
    return {
        "warmup_remaining_min": int(getattr(snap, "warmup_remaining_min", 0))
    }


ZONE_METRIC_SPECS: tuple[ZoneMetricSpec, ...] = (
    ZoneMetricSpec(
        zone="wk",
        object_id="woonkamer_geschatte_vloertemperatuur",
        unit=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        deadband=0.05,
        max_interval_s=3600.0,
        precision=1,
        extractor=lambda s: s.state.t_floor_c,
        attr_builder=_floor_temp_attrs,
    ),
    ZoneMetricSpec(
        zone="wk",
        object_id="woonkamer_geschatte_pvc_oppervlaktetemperatuur",
        unit=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        deadband=0.05,
        max_interval_s=3600.0,
        precision=1,
        extractor=lambda s: s.t_surface_est_c,
    ),
    ZoneMetricSpec(
        zone="wk",
        object_id="woonkamer_ongeplande_warmtelast",
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
    ZoneMetricSpec(
        zone="wk",
        object_id="woonkamer_vloer_afgiftevermogen",
        unit=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        deadband=50.0,
        max_interval_s=900.0,
        idle_zero_max_interval_s=21600.0,
        precision=0,
        extractor=lambda s: s.q_fh_now_w,
        zero_transition_immediate=True,
    ),
    ZoneMetricSpec(
        zone="wk",
        object_id="woonkamer_hindcast_6u",
        unit=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        deadband=0.05,
        max_interval_s=1800.0,
        precision=1,
        extractor=lambda s: s.t_hindcast_6h_c,
        attr_builder=_hindcast_attrs,
    ),
    ZoneMetricSpec(
        zone="kt",
        object_id="kantoor_geschatte_vloertemperatuur",
        unit=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        deadband=0.05,
        max_interval_s=3600.0,
        precision=1,
        extractor=lambda s: s.state.t_floor_c,
        attr_builder=_floor_temp_attrs,
    ),
    ZoneMetricSpec(
        zone="kt",
        object_id="kantoor_geschatte_pvc_oppervlaktetemperatuur",
        unit=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        deadband=0.05,
        max_interval_s=3600.0,
        precision=1,
        extractor=lambda s: s.t_surface_est_c,
    ),
    ZoneMetricSpec(
        zone="kt",
        object_id="kantoor_ongeplande_warmtelast",
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
    ZoneMetricSpec(
        zone="kt",
        object_id="kantoor_vloer_afgiftevermogen",
        unit=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        deadband=35.0,
        max_interval_s=900.0,
        idle_zero_max_interval_s=21600.0,
        precision=0,
        extractor=lambda s: s.q_fh_now_w,
        zero_transition_immediate=True,
    ),
    ZoneMetricSpec(
        zone="kt",
        object_id="kantoor_hindcast_6u",
        unit=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        deadband=0.05,
        max_interval_s=1800.0,
        precision=1,
        extractor=lambda s: s.t_hindcast_6h_c,
        attr_builder=_hindcast_attrs,
    ),
)


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
        ZoneMetricThrottledSensor.from_spec(coordinator, eid, spec)
        for spec in ZONE_METRIC_SPECS
    ]
    entities.extend(
        [
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
            BetonBatterijCopFitSensor(coordinator, eid),
        ]
    )

    async_add_entities(entities)
