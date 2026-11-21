"""BetonBatterij Home Assistant Custom Integration (Layer 1 Digital Twin).

Runtime `homeassistant.*` imports are guarded so offline `pytest` suites can
import `custom_components.betonbatterij.domain.*` and `ha_bridge.*` directly on
machines without Home Assistant installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from .const import DOMAIN, VERSION

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.const import Platform
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.device_registry import DeviceInfo

    from .domain.engine import DigitalTwinEngine
    from .ha_bridge.coordinator import BetonBatterijCoordinator

    BetonBatterijConfigEntry: TypeAlias = ConfigEntry[BetonBatterijCoordinator]
else:
    BetonBatterijConfigEntry: TypeAlias = Any

try:
    from homeassistant.const import Platform as _HAPlatform

    PLATFORMS: list[Platform] = [
        _HAPlatform.SENSOR,
        _HAPlatform.SWITCH,
        _HAPlatform.BUTTON,
    ]
except ModuleNotFoundError:
    PLATFORMS = ["sensor", "switch", "button"]  # type: ignore[assignment]


def build_device_info(entry_id: str) -> DeviceInfo:
    """Build the shared Home Assistant `DeviceInfo` for BetonBatterij.

    Args:
        entry_id: Active `ConfigEntry.entry_id`.

    Returns:
        `DeviceInfo` dictionary grouping all Layer 1 entities under one device.
    """
    try:
        from homeassistant.helpers.device_registry import (
            DeviceInfo as HADeviceInfo,
        )

        return HADeviceInfo(
            identifiers={(DOMAIN, entry_id)},
            name="BetonBatterij",
            manufacturer="BetonBatterij",
            model="Laag 1 Digital Twin (2R2C + ZOH Kalman)",
            sw_version=VERSION,
        )
    except ModuleNotFoundError:
        return {  # type: ignore[return-value]
            "identifiers": {(DOMAIN, entry_id)},
            "name": "BetonBatterij",
            "manufacturer": "BetonBatterij",
            "model": "Laag 1 Digital Twin (2R2C + ZOH Kalman)",
            "sw_version": VERSION,
        }


def async_get_engine(hass: HomeAssistant) -> DigitalTwinEngine | None:
    """Return the active `DigitalTwinEngine` instance for Layer 2 consumers."""
    loaded = hass.config_entries.async_loaded_entries(DOMAIN)
    if not loaded:
        return None
    coordinator: BetonBatterijCoordinator = loaded[0].runtime_data
    return coordinator.engine


async def async_setup_entry(
    hass: HomeAssistant, entry: BetonBatterijConfigEntry
) -> bool:
    """Set up BetonBatterij Layer 1 Digital Twin from a config entry."""
    from .ha_bridge.coordinator import (
        BetonBatterijCoordinator,
        extract_flow_rates_from_entry,
    )

    coordinator = BetonBatterijCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    async def _async_update_options(
        hass_instance: HomeAssistant, updated_entry: BetonBatterijConfigEntry
    ) -> None:
        del hass_instance  # Unused.
        await coordinator.async_update_flow_rates(
            extract_flow_rates_from_entry(updated_entry)
        )

    entry.async_on_unload(entry.add_update_listener(_async_update_options))
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: BetonBatterijConfigEntry
) -> bool:
    """Unload a BetonBatterij config entry and flush state to disk."""
    unload_ok = await hass.config_entries.async_unload_platforms(
        entry, PLATFORMS
    )
    if unload_ok:
        coordinator: BetonBatterijCoordinator | None = getattr(
            entry, "runtime_data", None
        )
        if coordinator is not None:
            await coordinator.async_shutdown()
    return unload_ok
