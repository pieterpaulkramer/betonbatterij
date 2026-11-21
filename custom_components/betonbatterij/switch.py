"""Switch platform for BetonBatterij (`switch.betonbatterij_auto_kalibratie`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity

try:
    from homeassistant.helpers.entity_platform import (
        AddConfigEntryEntitiesCallback,
    )
except ImportError:
    from homeassistant.helpers.entity_platform import (
        AddEntitiesCallback as AddConfigEntryEntitiesCallback,
    )

from . import BetonBatterijConfigEntry, build_device_info
from .const import DOMAIN

if TYPE_CHECKING:
    from .ha_bridge.coordinator import BetonBatterijCoordinator

PARALLEL_UPDATES = 0


class BetonBatterijAutoKalibratieSwitch(
    CoordinatorEntity["BetonBatterijCoordinator"], RestoreEntity, SwitchEntity
):
    """Toggle weekly Sunday `03:15` automatic MAP parameter calibration."""

    _attr_has_entity_name = True
    _attr_translation_key = "auto_kalibratie"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_icon = "mdi:tune-variant"

    def __init__(
        self, coordinator: BetonBatterijCoordinator, entry_id: str
    ) -> None:
        """Initialize the auto-calibration toggle switch."""
        super().__init__(coordinator)
        object_id = "betonbatterij_auto_kalibratie"
        self.entity_id = f"switch.{object_id}"
        self._attr_unique_id = f"{DOMAIN}_{object_id}"
        self._attr_device_info = build_device_info(entry_id)

    @property
    def is_on(self) -> bool:
        """Whether automatic weekly Sunday `03:15` calibration is enabled."""
        return bool(self.coordinator.auto_calibrate_enabled)

    async def async_added_to_hass(self) -> None:
        """Restore prior switch state (defaulting to `on` on first install)."""
        await super().async_added_to_hass()
        if not self.coordinator.has_persisted_auto_cal_pref:
            last_state = await self.async_get_last_state()
            if last_state is not None and last_state.state in ("on", "off"):
                restored = last_state.state == "on"
                await self.coordinator.async_set_auto_calibrate(
                    restored, persist=True
                )
            else:
                await self.coordinator.async_set_auto_calibrate(
                    True, persist=False
                )
        self.async_write_ha_state()

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Enable automatic weekly calibration."""
        del kwargs  # Unused.
        await self.coordinator.async_set_auto_calibrate(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Disable automatic weekly calibration."""
        del kwargs  # Unused.
        await self.coordinator.async_set_auto_calibrate(False)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BetonBatterijConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Register the BetonBatterij auto-calibration switch."""
    del hass  # Unused.
    coordinator = entry.runtime_data
    async_add_entities(
        [BetonBatterijAutoKalibratieSwitch(coordinator, entry.entry_id)]
    )
