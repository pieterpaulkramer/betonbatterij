"""Shared RingBuffer row construction, output merging, and JSON helpers.

Pure Python module with zero `homeassistant.*` imports so both offline tests
and runtime Home Assistant platforms share identical row schemas, JSON
sanitization, and rotameter flow-rate clamping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math
from typing import Any

from ..const import (
    ALL_HYDRAULIC_GROUPS,
    DEFAULT_FLOW_RATES_L_MIN,
    FLAG_OPEN_WIN_KT,
    FLAG_OPEN_WIN_WK,
    FLOW_OPTION_KEYS,
    FLOW_RATE_MAX_L_MIN,
    FLOW_RATE_MIN_L_MIN,
    RINGBUFFER_FLOAT_COLS,
    RINGBUFFER_INT_COLS,
)

ENGINE_OUTPUT_KEYS: tuple[str, ...] = (
    "t_floor_wk_c",
    "d_est_wk_w",
    "q_fh_wk_w",
    "t_hind6h_wk_c",
    "t_floor_kt_c",
    "d_est_kt_w",
    "q_fh_kt_w",
    "t_hind6h_kt_c",
)


def sanitize_json_value(value: Any) -> Any:
    """Recursively convert dataclasses and non-finite floats into JSON types.

    Replaces `NaN` and `±Inf` floats with `None` so Home Assistant's `orjson`
    state/diagnostics serializers never fail.

    Args:
        value: Arbitrary scalar, dataclass, mapping, list, or tuple.

    Returns:
        An `orjson`-serializable copy of `value`.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    return value


def to_plain_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping into an `orjson`-safe plain `dict`.

    Args:
        obj: Dataclass instance, mapping, or `None`.

    Returns:
        Sanitized dictionary, or `{}` if `obj` is `None` or not dict-like.
    """
    if obj is None:
        return {}
    if is_dataclass(obj) and not isinstance(obj, type):
        return sanitize_json_value(asdict(obj))  # type: ignore[no-any-return]
    if isinstance(obj, Mapping):
        return sanitize_json_value(dict(obj))  # type: ignore[no-any-return]
    return {}


def make_empty_raw_row(ts_utc: int, flags: int = 0) -> dict[str, float | int]:
    """Build a 60-column RingBuffer row dictionary initialized from `const.py`.

    Int columns default to `0` (with `ts_utc` and `flags` populated) and float
    columns default to `math.nan` so callers never drift from the canonical
    `RINGBUFFER_INT_COLS` / `RINGBUFFER_FLOAT_COLS` schema.

    Args:
        ts_utc: Aligned UTC epoch timestamp [s].
        flags: Initial status bitmask flags.

    Returns:
        Dictionary containing every key in `const.RINGBUFFER_ALL_COLS`.
    """
    row: dict[str, float | int] = {col: 0 for col in RINGBUFFER_INT_COLS}
    row["ts_utc"] = int(ts_utc)
    row["flags"] = int(flags)
    for col in RINGBUFFER_FLOAT_COLS:
        row[col] = math.nan
    return row


def apply_engine_outputs_to_row(
    raw_row: dict[str, float | int],
    outputs: Mapping[str, Any],
    *,
    engine: Any = None,
    open_window_wk: bool = False,
    open_window_kt: bool = False,
) -> None:
    """Merge engine estimator outputs and open-window flags into `raw_row`.

    Args:
        raw_row: Mutable 60-column RingBuffer row dictionary to update in place.
        outputs: Output mapping returned by `DigitalTwinEngine.process_5m_window`.
        engine: Optional engine instance queried for live `open_window_active`
            snapshot flags when `get_zone_snapshot` is available.
        open_window_wk: Explicit living-room open-window flag override.
        open_window_kt: Explicit attic-office open-window flag override.
    """
    for key, val in outputs.items():
        if key in ("open_window_wk", "open_window_kt"):
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            raw_row[key] = val

    win_wk = bool(open_window_wk or outputs.get("open_window_wk", False))
    win_kt = bool(open_window_kt or outputs.get("open_window_kt", False))
    if engine is not None and hasattr(engine, "get_zone_snapshot"):
        win_wk = win_wk or bool(
            engine.get_zone_snapshot("wk").open_window_active
        )
        win_kt = win_kt or bool(
            engine.get_zone_snapshot("kt").open_window_active
        )

    flags = int(raw_row.get("flags", 0))
    if win_wk:
        flags |= FLAG_OPEN_WIN_WK
    if win_kt:
        flags |= FLAG_OPEN_WIN_KT
    raw_row["flags"] = flags


def extract_flow_rates_from_entry(entry: Any) -> dict[str, float]:
    """Read and clamp the 10 manifold rotameter flow rates from `entry.options`.

    Args:
        entry: Home Assistant `ConfigEntry` (or test double with `.options`).

    Returns:
        Dictionary mapping every group in `ALL_HYDRAULIC_GROUPS` to a finite
        flow rate clamped to `[FLOW_RATE_MIN_L_MIN, FLOW_RATE_MAX_L_MIN]`.
    """
    flows: dict[str, float] = {}
    opts = getattr(entry, "options", None) or {}
    for group in ALL_HYDRAULIC_GROUPS:
        opt_key = FLOW_OPTION_KEYS[group]
        raw = opts.get(opt_key, DEFAULT_FLOW_RATES_L_MIN[group])
        try:
            if isinstance(raw, bool):
                raise TypeError("Boolean flow rate is invalid")
            val = float(raw)
            if not math.isfinite(val):
                raise ValueError("Non-finite flow rate")
            flows[group] = max(
                FLOW_RATE_MIN_L_MIN, min(FLOW_RATE_MAX_L_MIN, val)
            )
        except (TypeError, ValueError):
            flows[group] = float(DEFAULT_FLOW_RATES_L_MIN[group])
    return flows
