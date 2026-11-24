"""Diagnostics support for BetonBatterij (Integration Quality Scale Gold).

Exports a JSON-sanitized, location-redacted snapshot of coordinator configuration,
clock-guard health, columnar ringbuffer status, active thermal zone states, COP
curve parameters, and 14-day hindcast evaluation metrics.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from . import BetonBatterijConfigEntry
from .ha_bridge.row_utils import sanitize_json_value

TO_REDACT: frozenset[str] = frozenset(
    {"latitude", "longitude", "unique_id"}
)


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: BetonBatterijConfigEntry
) -> dict[str, Any]:
    """Return redacted diagnostics for a BetonBatterij config entry."""
    del hass  # Unused.
    coordinator = entry.runtime_data
    engine = coordinator.engine
    rb = coordinator.ringbuffer

    payload: dict[str, Any] = {
        "local_config_loaded": bool(coordinator.local_config_loaded),
        "entry_options": dict(entry.options),
        "flow_rates_l_min": dict(coordinator.flow_rates_l_min),
        "auto_calibrate_enabled": coordinator.auto_calibrate_enabled,
        "clock_guard": {
            "last_accepted_utc": coordinator.clock_guard.last_accepted_utc,
            "ntp_synced": coordinator.clock_guard.ntp_synced,
            "rejected_ticks": coordinator.clock_guard.rejected_ticks,
        },
        "ringbuffer": {
            "row_count": rb.row_count,
            "capacity": rb.capacity,
            "oldest_ts_utc": rb.oldest_ts_utc,
            "latest_ts_utc": rb.latest_ts_utc,
            "dirty_slots": sorted(rb.dirty_slots),
            "corrupt_chunks_recovered": rb.corrupt_chunks_recovered,
        },
        "zones": {
            "wk": sanitize_json_value(engine.get_zone_snapshot("wk")),
            "kt": sanitize_json_value(engine.get_zone_snapshot("kt")),
        },
        "zone_fit_diagnostics": {
            "wk": sanitize_json_value(engine.get_zone_fit_diagnostics("wk")),
            "kt": sanitize_json_value(engine.get_zone_fit_diagnostics("kt")),
        },
        "buffer_offsets": sanitize_json_value(
            getattr(engine, "buffer_offsets", None)
        ),
        "cop_params": sanitize_json_value(engine.cop_params),
        "cop_diagnostics": sanitize_json_value(engine.cop_diagnostics),
        "evaluation_14d": {
            "wk": sanitize_json_value(engine.eval_report_wk),
            "kt": sanitize_json_value(engine.eval_report_kt),
        },
        "calibration_history_count": len(engine.calibration_history),
    }
    return async_redact_data(payload, TO_REDACT)  # type: ignore[no-any-return]
