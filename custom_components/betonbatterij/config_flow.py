"""ConfigFlow and OptionsFlow for BetonBatterij (Layer 1 Shadow Mode).

Handles single-instance 1-click onboarding (`BetonBatterijConfigFlow`) and
runtime tuning of the 10 manifold rotameter flow rates (`BetonBatterijOptionsFlow`)
validated against `[FLOW_RATE_MIN_L_MIN, FLOW_RATE_MAX_L_MIN]`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback

from .const import (
    ALL_HYDRAULIC_GROUPS,
    DEFAULT_FLOW_RATES_L_MIN,
    DOMAIN,
    FLOW_OPTION_KEYS,
    FLOW_RATE_MAX_L_MIN,
    FLOW_RATE_MIN_L_MIN,
)

ENTRY_TITLE: str = "BetonBatterij — Laag 1 Shadow Mode"


def build_rotameter_options_schema(
    current_options: Mapping[str, Any] | None = None,
) -> vol.Schema:
    """Build Voluptuous schema for the 10 manifold rotameter flow rates [L/min].

    Args:
        current_options: Existing `ConfigEntry.options` mapping for defaults.

    Returns:
        Validated Voluptuous schema enforcing
        `[FLOW_RATE_MIN_L_MIN, FLOW_RATE_MAX_L_MIN]` L/min per hydraulic group.
    """
    opts = current_options or {}
    schema_dict: dict[Any, Any] = {}
    for group in ALL_HYDRAULIC_GROUPS:
        opt_key = FLOW_OPTION_KEYS[group]
        default_val = float(
            opts.get(opt_key, DEFAULT_FLOW_RATES_L_MIN[group])
        )
        schema_dict[vol.Required(opt_key, default=default_val)] = vol.All(
            vol.Coerce(float),
            vol.Range(min=FLOW_RATE_MIN_L_MIN, max=FLOW_RATE_MAX_L_MIN),
        )
    return vol.Schema(schema_dict)


class BetonBatterijConfigFlow(ConfigFlow, domain=DOMAIN):
    """Single-instance 1-click ConfigFlow for BetonBatterij."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle initial 1-click user setup."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        if user_input is not None:
            seeded_options = {
                FLOW_OPTION_KEYS[group]: float(DEFAULT_FLOW_RATES_L_MIN[group])
                for group in ALL_HYDRAULIC_GROUPS
            }
            return self.async_create_entry(
                title=ENTRY_TITLE,
                data={},
                options=seeded_options,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({}),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> BetonBatterijOptionsFlow:
        """Return the OptionsFlow handler for manifold rotameter flow rates."""
        del config_entry  # Unused; Home Assistant injects `self.config_entry`.
        return BetonBatterijOptionsFlow()


class BetonBatterijOptionsFlow(OptionsFlow):
    """OptionsFlow exposing all 10 manifold rotameter flow rates [L/min]."""

    def __init__(self, config_entry: ConfigEntry | None = None) -> None:
        """Initialize the options flow with optional test fallback entry."""
        self._fallback_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage manifold rotameter flow rate options."""
        if user_input is not None:
            cleaned = {
                FLOW_OPTION_KEYS[group]: round(
                    float(user_input[FLOW_OPTION_KEYS[group]]),
                    2,
                )
                for group in ALL_HYDRAULIC_GROUPS
            }
            return self.async_create_entry(title="", data=cleaned)

        entry = getattr(self, "config_entry", None) or self._fallback_entry
        current_opts = dict(entry.options) if entry is not None else {}
        base_schema = build_rotameter_options_schema(current_opts)
        if hasattr(self, "add_suggested_values_to_schema"):
            base_schema = self.add_suggested_values_to_schema(
                base_schema, current_opts
            )
        return self.async_show_form(
            step_id="init",
            data_schema=base_schema,
        )
