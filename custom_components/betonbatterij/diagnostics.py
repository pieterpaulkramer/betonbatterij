"""Diagnostics support for BetonBatterij (Integration Quality Scale Gold)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from . import BetonBatterijConfigEntry

TO_REDACT: frozenset[str] = frozenset(
    {"latitude", "longitude", "unique_id"}
)


def _serialize_diagnostic_obj(obj: Any) -> Any:
    """Convert dataclasses and mappings into `orjson`-safe JSON structures."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if is_dataclass(obj) and not isinstance(obj, type):
        return _serialize_diagnostic_obj(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _serialize_diagnostic_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_diagnostic_obj(item) for item in obj]
    return obj


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
            "wk": _serialize_diagnostic_obj(engine.get_zone_snapshot("wk")),
            "kt": _serialize_diagnostic_obj(engine.get_zone_snapshot("kt")),
        },
        "cop_params": _serialize_diagnostic_obj(engine.cop_params),
        "cop_diagnostics": _serialize_diagnostic_obj(engine.cop_diagnostics),
        "evaluation_14d": {
            "wk": _serialize_diagnostic_obj(engine.eval_report_wk),
            "kt": _serialize_diagnostic_obj(engine.eval_report_kt),
        },
        "calibration_history_count": len(engine.calibration_history),
    }
    return async_redact_data(payload, TO_REDACT)  # type: ignore[no-any-return]
