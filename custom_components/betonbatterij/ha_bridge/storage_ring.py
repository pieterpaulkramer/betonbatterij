"""Columnar 90-day RingBuffer and rotating weekly Store chunks (Choice F3).

`ColumnarRingBuffer` uses CPython `array('I')` and `array('f')` buffers with
zero top-level `homeassistant.*` imports (`~4.35 MB` flat RAM, zero PyObject
overhead, and `zlib` + `crc32` + `base64` chunk serialization).
"""

from __future__ import annotations

import base64
import binascii
import logging
import sys
import zlib
from array import array
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..const import (
    RINGBUFFER_ALL_COLS,
    RINGBUFFER_CHUNK_ROWS,
    RINGBUFFER_FLOAT_COLS,
    RINGBUFFER_INT_COLS,
    RINGBUFFER_NUM_CHUNKS,
    RINGBUFFER_ROWS,
    STORAGE_KEY_RING_PREFIX,
    STORAGE_VERSION,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.storage import Store

_LOGGER = logging.getLogger(__name__)

CHUNK_FORMAT_VERSION: int = 1
ZLIB_COMPRESSION_LEVEL: int = 3


def _coerce_uint32(val: Any) -> int:
    if val is None or isinstance(val, bool):
        return 0
    try:
        ival = int(val)
    except (TypeError, ValueError):
        return 0
    return max(0, ival) & 0xFFFFFFFF


def _coerce_float32(val: Any) -> float:
    if val is None or isinstance(val, bool):
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


class ColumnarRingBuffer:
    """Fixed-capacity columnar ring buffer backed by 14 weekly chunk slots.

    Attributes:
        chunk_rows: Number of 5-minute rows per weekly chunk (`2,016`).
        num_chunks: Number of rotating weekly chunk slots (`14`).
        capacity: Total circular row capacity (`28,224` rows).
        dirty_slots: Set of chunk indices modified since last disk save.
        corrupt_chunks_recovered: Counter of corrupted chunks zero-recovered.
    """

    def __init__(
        self,
        chunk_rows: int = RINGBUFFER_CHUNK_ROWS,
        num_chunks: int = RINGBUFFER_NUM_CHUNKS,
    ) -> None:
        """Initialize the columnar C-array ring buffer."""
        if chunk_rows <= 0 or num_chunks <= 0:
            raise ValueError("chunk_rows and num_chunks must be positive")
        self.chunk_rows: int = int(chunk_rows)
        self.num_chunks: int = int(num_chunks)
        self.capacity: int = self.chunk_rows * self.num_chunks

        self._int_cols: dict[str, array] = {
            col: array("I", [0]) * self.capacity for col in RINGBUFFER_INT_COLS
        }
        self._float_cols: dict[str, array] = {
            col: array("f", [0.0]) * self.capacity
            for col in RINGBUFFER_FLOAT_COLS
        }
        if (
            self._int_cols["ts_utc"].itemsize != 4
            or self._float_cols[RINGBUFFER_FLOAT_COLS[0]].itemsize != 4
        ):
            raise RuntimeError("Expected 4-byte uint32 and float32 array items")

        self._write_idx: int = 0
        self._last_written_idx: int | None = None
        self._row_count: int = 0
        self.dirty_slots: set[int] = set()
        self.corrupt_chunks_recovered: int = 0

    @property
    def row_count(self) -> int:
        """Total number of valid (`ts_utc > 0`) rows currently stored."""
        return self._row_count

    @property
    def latest_ts_utc(self) -> int:
        """UTC epoch timestamp of the most recently written row, or `0`."""
        if self._last_written_idx is None:
            return 0
        return int(self._int_cols["ts_utc"][self._last_written_idx])

    @property
    def oldest_ts_utc(self) -> int:
        """UTC epoch timestamp of the oldest valid row, or `0`."""
        if self._row_count == 0:
            return 0
        ts_arr = self._int_cols["ts_utc"]
        cap = self.capacity
        w_idx = self._write_idx
        for step in range(cap):
            ts_val = ts_arr[(w_idx + step) % cap]
            if ts_val > 0:
                return int(ts_val)
        return 0

    def append_row(self, row: Mapping[str, float | int | None]) -> int:
        """Append one 5-minute row and mark its weekly chunk slot dirty.

        Args:
            row: Mapping of column names to `uint32` or `float32` values.

        Returns:
            Weekly chunk slot index (`0 .. num_chunks - 1`) written to.
        """
        idx = self._write_idx
        ts_arr = self._int_cols["ts_utc"]
        was_valid = ts_arr[idx] > 0

        for col in RINGBUFFER_INT_COLS:
            self._int_cols[col][idx] = _coerce_uint32(row.get(col, 0))
        for col in RINGBUFFER_FLOAT_COLS:
            self._float_cols[col][idx] = _coerce_float32(row.get(col))

        is_now_valid = ts_arr[idx] > 0
        if is_now_valid and not was_valid:
            self._row_count += 1
        elif not is_now_valid and was_valid:
            self._row_count = max(0, self._row_count - 1)

        if is_now_valid:
            self._last_written_idx = idx
        self._write_idx = (idx + 1) % self.capacity

        slot = idx // self.chunk_rows
        self.dirty_slots.add(slot)
        return slot

    def update_last_row_outputs(
        self, outputs: Mapping[str, float | int | None]
    ) -> None:
        """Update computed estimator/hindcast columns on the latest row."""
        if self._last_written_idx is None:
            return
        idx = self._last_written_idx
        modified = False
        for col, val in outputs.items():
            if col in self._float_cols:
                self._float_cols[col][idx] = _coerce_float32(val)
                modified = True
            elif col in self._int_cols and col != "ts_utc":
                self._int_cols[col][idx] = _coerce_uint32(val)
                modified = True
        if modified:
            self.dirty_slots.add(idx // self.chunk_rows)

    def get_columns(
        self, max_rows: int = RINGBUFFER_ROWS
    ) -> dict[str, list[float | int]]:
        """Extract up to `max_rows` rows ordered chronologically oldest->newest.

        Args:
            max_rows: Maximum number of most-recent valid rows to return.

        Returns:
            Dictionary mapping all 58 `RINGBUFFER_ALL_COLS` names to lists.
        """
        if max_rows <= 0 or self._row_count == 0:
            return {col: [] for col in RINGBUFFER_ALL_COLS}

        ts_arr = self._int_cols["ts_utc"]
        cap = self.capacity
        w_idx = self._write_idx

        valid_indices: list[int] = []
        is_monotonic = True
        prev_ts = 0
        for step in range(cap):
            pos = (w_idx + step) % cap
            ts_val = ts_arr[pos]
            if ts_val > 0:
                if ts_val < prev_ts:
                    is_monotonic = False
                prev_ts = ts_val
                valid_indices.append(pos)

        if not valid_indices:
            return {col: [] for col in RINGBUFFER_ALL_COLS}

        if not is_monotonic:
            valid_indices.sort(key=lambda i: ts_arr[i])

        if len(valid_indices) > max_rows:
            valid_indices = valid_indices[-max_rows:]

        ranges = _coalesce_contiguous_ranges(valid_indices)
        result: dict[str, list[float | int]] = {}
        if len(ranges) <= 2:
            for col, arr in self._int_cols.items():
                out_i: list[int] = []
                for r_start, r_end in ranges:
                    out_i.extend(arr[r_start:r_end].tolist())
                result[col] = out_i
            for col, arr in self._float_cols.items():
                out_f: list[float] = []
                for r_start, r_end in ranges:
                    out_f.extend(arr[r_start:r_end].tolist())
                result[col] = out_f
        else:
            for col, arr in self._int_cols.items():
                result[col] = [int(arr[i]) for i in valid_indices]
            for col, arr in self._float_cols.items():
                result[col] = [float(arr[i]) for i in valid_indices]

        return result

    def encode_chunk(self, slot: int) -> dict[str, Any]:
        """Compress and checksum one weekly chunk (`slot` in `0..num_chunks-1`)."""
        if not 0 <= slot < self.num_chunks:
            raise ValueError(f"Invalid chunk slot {slot}")
        start = slot * self.chunk_rows
        end = start + self.chunk_rows

        parts: list[bytes] = []
        for col in RINGBUFFER_INT_COLS:
            sub = self._int_cols[col][start:end]
            if sys.byteorder != "little":
                sub = array("I", sub)
                sub.byteswap()
            parts.append(sub.tobytes())

        for col in RINGBUFFER_FLOAT_COLS:
            sub = self._float_cols[col][start:end]
            if sys.byteorder != "little":
                sub = array("f", sub)
                sub.byteswap()
            parts.append(sub.tobytes())

        raw_bytes = b"".join(parts)
        compressed = zlib.compress(raw_bytes, level=ZLIB_COMPRESSION_LEVEL)
        crc32_val = zlib.crc32(compressed) & 0xFFFFFFFF
        raw_crc32_val = zlib.crc32(raw_bytes) & 0xFFFFFFFF

        return {
            "format_version": CHUNK_FORMAT_VERSION,
            "slot": slot,
            "chunk_rows": self.chunk_rows,
            "n_int_cols": len(RINGBUFFER_INT_COLS),
            "n_float_cols": len(RINGBUFFER_FLOAT_COLS),
            "raw_nbytes": len(raw_bytes),
            "crc32": crc32_val,
            "raw_crc32": raw_crc32_val,
            "data_b64": base64.b64encode(compressed).decode("ascii"),
        }

    def decode_chunk(self, slot: int, payload: Mapping[str, Any]) -> bool:
        """Decode one weekly chunk, zero-recovering gracefully on corruption.

        Args:
            slot: Target weekly chunk index (`0 .. num_chunks - 1`).
            payload: JSON dictionary loaded from `Store`.

        Returns:
            `True` if decoded cleanly, `False` if corrupted and zero-recovered.
        """
        if not 0 <= slot < self.num_chunks:
            raise ValueError(f"Invalid chunk slot {slot}")
        start = slot * self.chunk_rows
        end = start + self.chunk_rows
        bytes_per_col = self.chunk_rows * 4
        expected_raw_nbytes = len(RINGBUFFER_ALL_COLS) * bytes_per_col

        try:
            if not isinstance(payload, Mapping):
                raise ValueError("Payload is not a mapping")
            if payload.get("format_version") != CHUNK_FORMAT_VERSION:
                raise ValueError("Unsupported chunk format_version")
            if payload.get("chunk_rows") != self.chunk_rows:
                raise ValueError("Chunk row count mismatch")
            if payload.get("n_int_cols") != len(RINGBUFFER_INT_COLS) or (
                payload.get("n_float_cols") != len(RINGBUFFER_FLOAT_COLS)
            ):
                raise ValueError("Chunk column count mismatch")

            b64_str = payload["data_b64"]
            if not isinstance(b64_str, str):
                raise ValueError("data_b64 must be an ASCII string")
            compressed = base64.b64decode(
                b64_str.encode("ascii"), validate=True
            )

            expected_crc = int(payload["crc32"]) & 0xFFFFFFFF
            actual_crc = zlib.crc32(compressed) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ValueError(
                    f"CRC32 mismatch on chunk {slot}: "
                    f"expected {expected_crc:#010x}, got {actual_crc:#010x}"
                )

            raw_bytes = zlib.decompress(compressed)
            if len(raw_bytes) != expected_raw_nbytes:
                raise ValueError(
                    f"Decompressed byte length mismatch on chunk {slot}: "
                    f"expected {expected_raw_nbytes}, got {len(raw_bytes)}"
                )

            if "raw_crc32" in payload:
                expected_raw_crc = int(payload["raw_crc32"]) & 0xFFFFFFFF
                actual_raw_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
                if actual_raw_crc != expected_raw_crc:
                    raise ValueError("Decompressed raw_crc32 mismatch")

            offset = 0
            decoded_ints: dict[str, array] = {}
            for col in RINGBUFFER_INT_COLS:
                arr = array("I")
                arr.frombytes(raw_bytes[offset : offset + bytes_per_col])
                if sys.byteorder != "little":
                    arr.byteswap()
                decoded_ints[col] = arr
                offset += bytes_per_col

            decoded_floats: dict[str, array] = {}
            for col in RINGBUFFER_FLOAT_COLS:
                arr = array("f")
                arr.frombytes(raw_bytes[offset : offset + bytes_per_col])
                if sys.byteorder != "little":
                    arr.byteswap()
                decoded_floats[col] = arr
                offset += bytes_per_col

        except (
            KeyError,
            TypeError,
            ValueError,
            binascii.Error,
            zlib.error,
        ) as err:
            _LOGGER.warning(
                "Corrupt BetonBatterij ringbuffer chunk slot %02d recovered "
                "(zeroed %d rows): %s",
                slot,
                self.chunk_rows,
                err,
            )
            self._zero_slot(slot)
            self.corrupt_chunks_recovered += 1
            self.dirty_slots.add(slot)
            self._recompute_cursors()
            return False

        for col, arr in decoded_ints.items():
            self._int_cols[col][start:end] = arr
        for col, arr in decoded_floats.items():
            self._float_cols[col][start:end] = arr

        self._recompute_cursors()
        return True

    def _zero_slot(self, slot: int) -> None:
        start = slot * self.chunk_rows
        end = start + self.chunk_rows
        zero_i = array("I", [0]) * self.chunk_rows
        zero_f = array("f", [0.0]) * self.chunk_rows
        for arr in self._int_cols.values():
            arr[start:end] = zero_i
        for arr in self._float_cols.values():
            arr[start:end] = zero_f

    def _recompute_cursors(self) -> None:
        ts_arr = self._int_cols["ts_utc"]
        count = 0
        max_ts = 0
        max_idx: int | None = None
        for idx, ts_val in enumerate(ts_arr):
            if ts_val > 0:
                count += 1
                if ts_val >= max_ts:
                    max_ts = ts_val
                    max_idx = idx

        self._row_count = count
        self._last_written_idx = max_idx
        self._write_idx = (
            0 if max_idx is None else (max_idx + 1) % self.capacity
        )


def _coalesce_contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    run_start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((run_start, prev + 1))
            run_start = idx
            prev = idx
    ranges.append((run_start, prev + 1))
    return ranges


class HARingBufferStore:
    """Home Assistant `Store` persistence wrapper for the 14 weekly chunks."""

    def __init__(
        self, hass: HomeAssistant, ringbuffer: ColumnarRingBuffer
    ) -> None:
        """Initialize the 14 rotating weekly `Store` instances."""
        from homeassistant.helpers.storage import Store

        self._hass = hass
        self.ringbuffer = ringbuffer
        self._stores: list[Store[dict[str, Any]]] = [
            Store(
                hass,
                STORAGE_VERSION,
                f"{STORAGE_KEY_RING_PREFIX}{slot:02d}",
                atomic_writes=True,
                serialize_in_event_loop=False,
            )
            for slot in range(ringbuffer.num_chunks)
        ]

    async def async_load_all(self) -> int:
        """Load all 14 weekly chunk files from `.storage` and return row count."""
        for slot, store in enumerate(self._stores):
            try:
                payload = await store.async_load()
            except Exception as err:  # noqa: BLE001
                _LOGGER.warning(
                    "Failed loading ringbuffer store slot %02d: %s", slot, err
                )
                self.ringbuffer.corrupt_chunks_recovered += 1
                self.ringbuffer.dirty_slots.add(slot)
                continue

            if payload is not None and self.ringbuffer.decode_chunk(
                slot, payload
            ):
                self.ringbuffer.dirty_slots.discard(slot)
        return self.ringbuffer.row_count

    async def async_save_dirty(self) -> int:
        """Persist only modified weekly chunk slots to disk."""
        slots_to_save = sorted(self.ringbuffer.dirty_slots)
        saved_count = 0
        for slot in slots_to_save:
            payload = self.ringbuffer.encode_chunk(slot)
            await self._stores[slot].async_save(payload)
            self.ringbuffer.dirty_slots.discard(slot)
            saved_count += 1
        return saved_count
