"""Home Assistant 3-store persistence lifecycle for BetonBatterij.

Encapsulates the state store (`betonbatterij.state`), episode bank store
(`betonbatterij.episode_bank`), and 14-chunk rotating ringbuffer store
(`betonbatterij.ringbuffer_w00..13`) cleanly behind `BetonBatterijPersistence`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..const import (
    STORAGE_KEY_EPISODE_BANK,
    STORAGE_KEY_STATE,
    STORAGE_VERSION,
)
from ._ha_compat import HomeAssistant, Store
from .storage_ring import ColumnarRingBuffer, HARingBufferStore


class _OfflineRingStoreStub:
    """In-memory no-op ring store stub when Home Assistant is not installed."""

    async def async_load_all(self) -> int:
        return 0

    async def async_save_dirty(self) -> int:
        return 0


class BetonBatterijPersistence:
    """Manage loading and atomic saving across the 3 BetonBatterij HA stores.

    Attributes:
        ringbuffer: In-memory 90-day columnar ring buffer.
        ring_store: Weekly chunk `HARingBufferStore` (or offline stub).
        state_store: Engine parameters, Kalman states, and preferences store.
        episode_store: Dedicated store for the episode bank payload.
    """

    def __init__(
        self, hass: HomeAssistant, ringbuffer: ColumnarRingBuffer
    ) -> None:
        """Initialize the ringbuffer, state, and episode-bank stores."""
        self.ringbuffer = ringbuffer
        try:
            self.ring_store: HARingBufferStore = HARingBufferStore(
                hass, ringbuffer
            )
        except ModuleNotFoundError:
            self.ring_store = _OfflineRingStoreStub()  # type: ignore[assignment]

        self.state_store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            STORAGE_KEY_STATE,
            atomic_writes=True,
            serialize_in_event_loop=False,
        )
        self.episode_store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            STORAGE_KEY_EPISODE_BANK,
            atomic_writes=True,
            serialize_in_event_loop=False,
        )

    async def async_load_all(
        self,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, int]:
        """Load state, episode bank, and all 14 weekly ringbuffer chunks.

        Returns:
            Tuple `(state_payload, episode_payload, ringbuffer_row_count)`.
        """
        state_payload = await self.state_store.async_load()
        episode_payload = await self.episode_store.async_load()
        row_count = await self.ring_store.async_load_all()
        return state_payload, episode_payload, row_count

    async def async_save_state_and_episodes(
        self,
        engine: Any,
        *,
        last_ts_utc: int,
        auto_calibrate_enabled: bool,
    ) -> None:
        """Persist engine state, preferences, and optional episode bank."""
        engine_dict = dict(engine.to_state_dict())
        episode_bank = engine_dict.pop("episode_bank", None)
        if episode_bank is None and hasattr(engine, "episode_bank"):
            eb_attr = getattr(engine, "episode_bank")
            if hasattr(eb_attr, "to_dict"):
                episode_bank = eb_attr.to_dict()
            elif isinstance(eb_attr, (dict, list)):
                episode_bank = eb_attr

        state_doc: dict[str, Any] = {
            "last_ts_utc": int(last_ts_utc),
            "auto_calibrate_enabled": bool(auto_calibrate_enabled),
            "engine": engine_dict,
        }
        await self.state_store.async_save(state_doc)
        if episode_bank is not None:
            await self.episode_store.async_save({"episode_bank": episode_bank})

    async def async_flush_all(
        self,
        engine: Any,
        *,
        last_ts_utc: int,
        auto_calibrate_enabled: bool,
    ) -> None:
        """Persist dirty weekly ringbuffer chunks, engine state, and episodes."""
        await self.ring_store.async_save_dirty()
        await self.async_save_state_and_episodes(
            engine,
            last_ts_utc=last_ts_utc,
            auto_calibrate_enabled=auto_calibrate_enabled,
        )

    @staticmethod
    def build_snapshot_payload(
        engine: Any,
        *,
        fit_sensor_generation: int,
        auto_calibrate_enabled: bool,
        ringbuffer_rows: int,
    ) -> dict[str, Any]:
        """Build the immutable coordinator data payload for platform entities."""
        return {
            "wk": engine.get_zone_snapshot("wk"),
            "kt": engine.get_zone_snapshot("kt"),
            "fit_sensor_generation": int(fit_sensor_generation),
            "auto_calibrate_enabled": bool(auto_calibrate_enabled),
            "ringbuffer_rows": int(ringbuffer_rows),
        }
