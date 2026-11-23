"""Home Assistant bridge layer for BetonBatterij (Layer 1 Digital Twin).

Exports only pure-Python accumulator, clock-jump guard, row helpers, and
columnar ring-buffer classes with zero `homeassistant.*` import attempts at
package import time. Callers needing `BetonBatterijPersistence` import it
directly from `ha_bridge.persistence`.
"""

from __future__ import annotations

from .accumulator import (
    HydraulicPowerAccumulator,
    HydraulicWindowSummary,
    StepSignalAccumulator,
    WaxValveTracker,
    WindowAccumulatorCore,
    WindowCloseResult,
)
from .clock import ClockJumpGuard
from .row_utils import (
    apply_engine_outputs_to_row,
    extract_flow_rates_from_entry,
    make_empty_raw_row,
    sanitize_json_value,
)
from .storage_ring import ColumnarRingBuffer

__all__ = [
    "ClockJumpGuard",
    "ColumnarRingBuffer",
    "HydraulicPowerAccumulator",
    "HydraulicWindowSummary",
    "StepSignalAccumulator",
    "WaxValveTracker",
    "WindowAccumulatorCore",
    "WindowCloseResult",
    "apply_engine_outputs_to_row",
    "extract_flow_rates_from_entry",
    "make_empty_raw_row",
    "sanitize_json_value",
]
