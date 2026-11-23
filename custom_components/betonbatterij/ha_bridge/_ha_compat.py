"""Internal Home Assistant compatibility shims for offline unit testing.

Centralizes guarded `homeassistant.*` imports and lightweight offline fallback
stubs used by `coordinator.py`, `persistence.py`, `sensor.py`, and `__init__.py`
so pure-Python `pytest` suites run without Home Assistant Core installed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import logging
from typing import Any

try:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.const import (
        EVENT_HOMEASSISTANT_STOP,
        EntityCategory,
        UnitOfPower,
        UnitOfTemperature,
    )
    from homeassistant.core import Event, HomeAssistant, State, callback
    from homeassistant.helpers.event import (
        async_track_state_change_event,
        async_track_time_change,
        async_track_utc_time_change,
    )
    from homeassistant.helpers.storage import Store
    from homeassistant.helpers.update_coordinator import (
        CoordinatorEntity,
        DataUpdateCoordinator,
    )
    from homeassistant.util import dt as dt_util

    HA_AVAILABLE: bool = True
except ModuleNotFoundError:  # Offline pytest fallback stubs.
    HA_AVAILABLE = False
    ConfigEntry = Any  # type: ignore[assignment,misc]
    Event = Any  # type: ignore[assignment,misc]
    HomeAssistant = Any  # type: ignore[assignment,misc]
    State = Any  # type: ignore[assignment,misc]
    EVENT_HOMEASSISTANT_STOP = "homeassistant_stop"

    class _CoreEnumStub:
        """Offline enum fallback matching Home Assistant Core string values."""

        DIAGNOSTIC = "diagnostic"
        CONFIG = "config"
        CELSIUS = "°C"
        WATT = "W"

    EntityCategory = _CoreEnumStub  # type: ignore[assignment,misc]
    UnitOfTemperature = _CoreEnumStub  # type: ignore[assignment,misc]
    UnitOfPower = _CoreEnumStub  # type: ignore[assignment,misc]

    def callback(
        func: Callable[..., Any],
    ) -> Callable[..., Any]:  # type: ignore[misc]
        """No-op decorator stub for Home Assistant `@callback`."""
        return func

    def async_track_state_change_event(
        *args: Any, **kwargs: Any
    ) -> Callable[[], None]:
        """No-op state-change listener stub."""
        del args, kwargs  # Unused.
        return lambda: None

    def async_track_time_change(
        *args: Any, **kwargs: Any
    ) -> Callable[[], None]:
        """No-op local time-change listener stub."""
        del args, kwargs  # Unused.
        return lambda: None

    def async_track_utc_time_change(
        *args: Any, **kwargs: Any
    ) -> Callable[[], None]:
        """No-op UTC time-change listener stub."""
        del args, kwargs  # Unused.
        return lambda: None

    class Store:  # type: ignore[no-redef]
        """In-memory stub mirroring `homeassistant.helpers.storage.Store`."""

        def __class_getitem__(cls, item: Any) -> type:
            del item  # Unused.
            return cls

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs  # Unused.
            self.saved_data: Any = None

        async def async_load(self) -> Any:
            return self.saved_data

        async def async_save(self, data: Any) -> None:
            self.saved_data = data

    class CoordinatorEntity:  # type: ignore[no-redef]
        """Minimal offline stub for `CoordinatorEntity`."""

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

    class DataUpdateCoordinator:  # type: ignore[no-redef]
        """Minimal offline stub for `DataUpdateCoordinator`."""

        def __class_getitem__(cls, item: Any) -> type:
            del item  # Unused.
            return cls

        def __init__(
            self,
            hass: Any,
            logger: logging.Logger,
            *,
            name: str,
            config_entry: Any = None,
            update_interval: Any = None,
            always_update: bool = True,
        ) -> None:
            del update_interval, always_update  # Unused.
            self.hass = hass
            self.logger = logger
            self.name = name
            self.config_entry = config_entry
            self.data: Any = None

        def async_set_updated_data(self, data: Any) -> None:
            self.data = data

    class _DtUtilStub:
        """Minimal offline stub for `homeassistant.util.dt`."""

        @staticmethod
        def utcnow() -> datetime:
            return datetime.now(timezone.utc)

        @staticmethod
        def as_local(dt_val: datetime) -> datetime:
            return dt_val

    dt_util = _DtUtilStub()  # type: ignore[assignment]


try:
    from homeassistant.components.sensor import (
        SensorDeviceClass,
        SensorEntity,
        SensorStateClass,
    )
    try:
        from homeassistant.helpers.entity_platform import (
            AddConfigEntryEntitiesCallback,
        )
    except ImportError:
        from homeassistant.helpers.entity_platform import (
            AddEntitiesCallback as AddConfigEntryEntitiesCallback,
        )
except ModuleNotFoundError:  # Offline pytest fallback stubs.
    AddConfigEntryEntitiesCallback = Any  # type: ignore[assignment,misc]

    class _SensorEnumStub:
        """Offline enum fallback matching Home Assistant sensor constants."""

        TEMPERATURE = "temperature"
        POWER = "power"
        MEASUREMENT = "measurement"

    SensorDeviceClass = _SensorEnumStub  # type: ignore[assignment,misc]
    SensorStateClass = _SensorEnumStub  # type: ignore[assignment,misc]

    class SensorEntity:  # type: ignore[no-redef]
        """Minimal offline stub for `SensorEntity`."""

        entity_id: str | None = None
        _attr_native_value: Any = None
        _attr_extra_state_attributes: dict[str, Any] | None = None

        @property
        def native_value(self) -> Any:
            return self._attr_native_value

        def async_write_ha_state(self) -> None:
            return None


__all__ = [
    "AddConfigEntryEntitiesCallback",
    "ConfigEntry",
    "CoordinatorEntity",
    "DataUpdateCoordinator",
    "EVENT_HOMEASSISTANT_STOP",
    "EntityCategory",
    "Event",
    "HA_AVAILABLE",
    "HomeAssistant",
    "SensorDeviceClass",
    "SensorEntity",
    "SensorStateClass",
    "State",
    "Store",
    "UnitOfPower",
    "UnitOfTemperature",
    "async_track_state_change_event",
    "async_track_time_change",
    "async_track_utc_time_change",
    "callback",
    "dt_util",
]
