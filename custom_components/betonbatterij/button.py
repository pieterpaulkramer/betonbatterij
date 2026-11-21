"""Button platform for BetonBatterij calibration, rollback, reset, and export."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from homeassistant.components.button import (
    ButtonEntity,
    ButtonEntityDescription,
)
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
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


@dataclass(frozen=True, kw_only=True)
class BetonBatterijButtonEntityDescription(ButtonEntityDescription):
    """Describe a BetonBatterij action button entity."""

    object_id: str
    press_fn: Callable[[BetonBatterijCoordinator], Awaitable[Any]]


BUTTON_DESCRIPTIONS: tuple[BetonBatterijButtonEntityDescription, ...] = (
    BetonBatterijButtonEntityDescription(
        key="herbereken_parameters",
        translation_key="herbereken_parameters",
        object_id="betonbatterij_herbereken_parameters",
        entity_category=EntityCategory.CONFIG,
        icon="mdi:calculator-variant",
        press_fn=lambda coord: coord.async_trigger_manual_calibration(),
    ),
    BetonBatterijButtonEntityDescription(
        key="rollback_parameters",
        translation_key="rollback_parameters",
        object_id="betonbatterij_rollback_parameters",
        entity_category=EntityCategory.CONFIG,
        icon="mdi:undo-variant",
        press_fn=lambda coord: coord.async_trigger_rollback(),
    ),
    BetonBatterijButtonEntityDescription(
        key="reset_parameters",
        translation_key="reset_parameters",
        object_id="betonbatterij_reset_parameters",
        entity_category=EntityCategory.CONFIG,
        icon="mdi:restore",
        press_fn=lambda coord: coord.async_trigger_reset(),
    ),
    BetonBatterijButtonEntityDescription(
        key="exporteer_csv",
        translation_key="exporteer_csv",
        object_id="betonbatterij_exporteer_csv",
        entity_category=EntityCategory.DIAGNOSTIC,
        icon="mdi:file-delimited-outline",
        press_fn=lambda coord: coord.async_trigger_export_csv(),
    ),
)


class BetonBatterijActionButton(
    CoordinatorEntity["BetonBatterijCoordinator"], ButtonEntity
):
    """Button entity invoking an executor-backed DigitalTwinEngine action."""

    _attr_has_entity_name = True
    entity_description: BetonBatterijButtonEntityDescription

    def __init__(
        self,
        coordinator: BetonBatterijCoordinator,
        entry_id: str,
        description: BetonBatterijButtonEntityDescription,
    ) -> None:
        """Initialize the action button."""
        super().__init__(coordinator)
        self.entity_description = description
        self.entity_id = f"button.{description.object_id}"
        self._attr_unique_id = f"{DOMAIN}_{description.object_id}"
        self._attr_device_info = build_device_info(entry_id)

    async def async_press(self) -> None:
        """Execute the button's coordinator action."""
        await self.entity_description.press_fn(self.coordinator)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BetonBatterijConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Register all 4 Layer 1 BetonBatterij action buttons."""
    del hass  # Unused.
    coordinator = entry.runtime_data
    async_add_entities(
        BetonBatterijActionButton(coordinator, entry.entry_id, description)
        for description in BUTTON_DESCRIPTIONS
    )
