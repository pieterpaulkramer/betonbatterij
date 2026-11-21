"""Home Assistant bridge layer for BetonBatterij (Layer 1 Digital Twin).

Core accumulator and columnar ring-buffer classes avoid top-level
`homeassistant.*` imports so offline unit tests run without Home Assistant.
"""

from __future__ import annotations

from .accumulator import (
    ClockJumpGuard,
    HydraulicPowerAccumulator,
    HydraulicWindowSummary,
    StepSignalAccumulator,
    WaxValveTracker,
    WindowAccumulatorCore,
    WindowCloseResult,
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
]
