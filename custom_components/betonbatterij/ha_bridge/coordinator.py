"""Five-minute tick coordinator, storage persistence, and background jobs."""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.event import (
    async_track_time_change,
    async_track_utc_time_change,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from ..const import (
    ALL_HYDRAULIC_GROUPS,
    DEFAULT_FLOW_RATES_L_MIN,
    DOMAIN,
    DT_STEP_S,
    EXPORT_DIR_DEFAULT,
    FLAG_GAP_BRIDGED,
    FLAG_OPEN_WIN_KT,
    FLAG_OPEN_WIN_WK,
    FLOW_OPTION_KEYS,
    RINGBUFFER_ROWS,
    STORAGE_KEY_EPISODE_BANK,
    STORAGE_KEY_STATE,
    STORAGE_VERSION,
)
from ..domain.engine import DigitalTwinEngine
from ..house_config import (
    HOUSE_ENTITIES,
    LOCAL_CONFIG_FILENAME,
    ROOM_SENSORS,
    HouseEntityConfig,
    RoomSensorConfig,
    load_house_config,
)
from .accumulator import ClockJumpGuard, HAWindowAccumulator, WindowCloseResult
from .backfill import async_run_cold_start_backfill
from .storage_ring import ColumnarRingBuffer, HARingBufferStore

_LOGGER = logging.getLogger(__name__)

MAX_RESTART_GAP_BRIDGE_S: int = 10_800  # 3 hours


def _load_local_config_sync(
    cfg_path: Path,
) -> tuple[HouseEntityConfig, dict[str, RoomSensorConfig], bool]:
    """Load optional `/config/betonbatterij_local.json` off the event loop."""
    if not cfg_path.is_file():
        return HOUSE_ENTITIES, dict(ROOM_SENSORS), False
    entities, room_sensors = load_house_config(cfg_path)
    return entities, room_sensors, True


def extract_flow_rates_from_entry(entry: ConfigEntry) -> dict[str, float]:
    """Read the 10 manifold rotameter flow rates [L/min] from `entry.options`."""
    flows: dict[str, float] = {}
    opts = entry.options or {}
    for group in ALL_HYDRAULIC_GROUPS:
        opt_key = FLOW_OPTION_KEYS[group]
        raw = opts.get(opt_key, DEFAULT_FLOW_RATES_L_MIN[group])
        try:
            flows[group] = max(0.1, min(25.0, float(raw)))
        except (TypeError, ValueError):
            flows[group] = float(DEFAULT_FLOW_RATES_L_MIN[group])
    return flows


class BetonBatterijCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Central Home Assistant coordinator for the Layer 1 Digital Twin."""

    config_entry: ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize stores, accumulator, clock guard, and DigitalTwinEngine."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            config_entry=entry,
            update_interval=None,
            always_update=False,
        )
        self.entry = entry
        self.flow_rates_l_min = extract_flow_rates_from_entry(entry)
        raw_lat = getattr(hass.config, "latitude", None)
        raw_lon = getattr(hass.config, "longitude", None)
        lat_deg = (
            float(raw_lat)
            if isinstance(raw_lat, (int, float))
            and math.isfinite(raw_lat)
            and float(raw_lat) != 0.0
            else None
        )
        lon_deg = (
            float(raw_lon)
            if isinstance(raw_lon, (int, float))
            and math.isfinite(raw_lon)
            and float(raw_lon) != 0.0
            else None
        )
        self.engine = DigitalTwinEngine(
            flow_rates_l_min=self.flow_rates_l_min,
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
        )
        self.ringbuffer = ColumnarRingBuffer()
        self.ring_store = HARingBufferStore(hass, self.ringbuffer)

        self._state_store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            STORAGE_KEY_STATE,
            atomic_writes=True,
            serialize_in_event_loop=False,
        )
        self._episode_store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            STORAGE_KEY_EPISODE_BANK,
            atomic_writes=True,
            serialize_in_event_loop=False,
        )

        self.entities: HouseEntityConfig = HOUSE_ENTITIES
        self.room_sensors: dict[str, RoomSensorConfig] = dict(ROOM_SENSORS)
        self.local_config_loaded: bool = False
        self.accumulator = HAWindowAccumulator(
            hass, self.entities, self.room_sensors
        )
        self.clock_guard = ClockJumpGuard()

        self.auto_calibrate_enabled: bool = True
        self.has_persisted_auto_cal_pref: bool = False
        self.fit_sensor_generation: int = 0
        self.last_raw_row: dict[str, float | int] = {}

        self._processing_lock = asyncio.Lock()
        self._unsubs: list[Callable[[], None]] = []
        self._stop_unsub: Callable[[], None] | None = None
        self._backfill_task: asyncio.Task[None] | None = None
        self._is_setup_done: bool = False
        self._is_shut_down: bool = False

    async def async_setup(self) -> None:
        """Set up stores, listeners, and initial coordinator data snapshot."""
        await self._async_setup()
        self.async_set_updated_data(await self._async_update_data())

    async def _async_setup(self) -> None:
        """Load state and ringbuffer once during `async_config_entry_first_refresh`."""
        if self._is_setup_done:
            return
        self._is_setup_done = True

        local_cfg_path = Path(self.hass.config.path(LOCAL_CONFIG_FILENAME))
        try:
            (
                self.entities,
                self.room_sensors,
                self.local_config_loaded,
            ) = await self.hass.async_add_executor_job(
                _load_local_config_sync, local_cfg_path
            )
            if self.local_config_loaded:
                _LOGGER.info(
                    "Loaded BetonBatterij local entity config from %s",
                    local_cfg_path,
                )
            else:
                _LOGGER.warning(
                    "Local entity config %s not found; generic placeholder "
                    "entity IDs are active",
                    local_cfg_path,
                )
        except (OSError, ValueError) as err:
            self.entities = HOUSE_ENTITIES
            self.room_sensors = dict(ROOM_SENSORS)
            self.local_config_loaded = False
            _LOGGER.warning(
                "Failed loading local entity config %s (%s); falling back to "
                "generic placeholder entity IDs",
                local_cfg_path,
                err,
            )
        self.accumulator.configure(self.entities, self.room_sensors)

        state_payload = await self._state_store.async_load()
        episode_payload = await self._episode_store.async_load()
        await self.ring_store.async_load_all()

        saved_ts_utc = 0
        engine_dict: dict[str, Any] = {}
        if isinstance(state_payload, Mapping):
            saved_ts_utc = int(state_payload.get("last_ts_utc", 0) or 0)
            if "auto_calibrate_enabled" in state_payload:
                self.auto_calibrate_enabled = bool(
                    state_payload["auto_calibrate_enabled"]
                )
                self.has_persisted_auto_cal_pref = True
            engine_dict = dict(state_payload.get("engine") or {})

        if (
            isinstance(episode_payload, Mapping)
            and episode_payload.get("episode_bank") is not None
        ):
            engine_dict["episode_bank"] = episode_payload["episode_bank"]

        if engine_dict:
            try:
                self.engine.load_state_dict(engine_dict)
            except Exception as err:  # noqa: BLE001
                _LOGGER.warning(
                    "Failed restoring DigitalTwinEngine state; continuing "
                    "with priors: %s",
                    err,
                )

        self.engine.update_flow_rates(self.flow_rates_l_min)

        t_mono_now = time.monotonic()
        anchor_utc = max(saved_ts_utc, self.ringbuffer.latest_ts_utc)
        self.clock_guard = ClockJumpGuard(
            saved_wall_utc=anchor_utc,
            t_mono_start=t_mono_now,
        )

        self.accumulator.start()

        now_utc = dt_util.utcnow().timestamp()
        aligned_now = (int(now_utc) // DT_STEP_S) * DT_STEP_S

        if self.ringbuffer.row_count == 0:
            self._backfill_task = self.hass.async_create_background_task(
                self._async_background_cold_start(),
                f"{DOMAIN}_cold_start_backfill",
            )
        else:
            last_ts = self.ringbuffer.latest_ts_utc
            gap_s = aligned_now - last_ts
            if (
                aligned_now >= self.clock_guard.min_valid_utc
                and DT_STEP_S <= gap_s <= MAX_RESTART_GAP_BRIDGE_S
            ):
                self._bridge_restart_gap(aligned_now, gap_s, t_mono_now)
            elif not self.last_raw_row:
                self._seed_last_raw_row_from_ring()

            if (
                self.engine.eval_report_wk is None
                or self.engine.eval_report_kt is None
            ):
                self.hass.async_create_background_task(
                    self._async_refresh_eval_only(),
                    f"{DOMAIN}_initial_eval_refresh",
                )
            else:
                self.fit_sensor_generation += 1

        self._unsubs.append(
            async_track_utc_time_change(
                self.hass,
                self._async_on_5m_tick,
                minute=range(0, 60, 5),
                second=0,
            )
        )
        self._unsubs.append(
            async_track_time_change(
                self.hass,
                self._async_on_6h_maintenance,
                hour=[0, 6, 12, 18],
                minute=15,
                second=0,
            )
        )
        self._unsubs.append(
            async_track_time_change(
                self.hass,
                self._async_on_weekly_calibration_slot,
                hour=3,
                minute=15,
                second=0,
            )
        )
        self._stop_unsub = self.hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP, self._async_on_ha_stop
        )

    async def _async_update_data(self) -> dict[str, Any]:
        """Return current zone snapshots and metadata for entities."""
        return self._build_snapshot_payload()

    def _bridge_restart_gap(
        self, aligned_now: int, gap_s: int, t_mono_now: float
    ) -> None:
        closed = self.accumulator.core.close_window(
            ts_utc=aligned_now,
            t_mono=t_mono_now,
            window_dt=float(gap_s),
        )
        raw_row = closed.raw_row
        raw_row["flags"] = int(raw_row["flags"]) | FLAG_GAP_BRIDGED
        try:
            outputs = self.engine.process_5m_window(
                raw_row,
                temp_changed_wk=closed.temp_changed_wk,
                hum_changed_wk=closed.hum_changed_wk,
                stale_wk=closed.stale_wk,
                temp_changed_kt=closed.temp_changed_kt,
                hum_changed_kt=closed.hum_changed_kt,
                stale_kt=closed.stale_kt,
                dt_s=float(gap_s),
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Failed bridging restart gap (%d s): %s", gap_s, err)
            return

        self._apply_engine_outputs_to_row(raw_row, outputs)
        self.ringbuffer.append_row(raw_row)
        self.ringbuffer.update_last_row_outputs(outputs)
        self.last_raw_row = raw_row
        self.clock_guard.record_bridged_utc(aligned_now, t_mono_now)
        _LOGGER.info(
            "Bridged BetonBatterij restart gap of %d s at ts_utc=%d.",
            gap_s,
            aligned_now,
        )

    async def _async_background_cold_start(self) -> None:
        async with self._processing_lock:
            now_utc = dt_util.utcnow().timestamp()
            rows = await async_run_cold_start_backfill(
                hass=self.hass,
                engine=self.engine,
                ringbuffer=self.ringbuffer,
                ring_store=self.ring_store,
                now_utc=now_utc,
                entities=self.entities,
                room_sensors=self.room_sensors,
            )
            if rows > 0:
                latest_ts = self.ringbuffer.latest_ts_utc
                self.clock_guard.record_bridged_utc(latest_ts, time.monotonic())
                self._seed_last_raw_row_from_ring()
                await self.async_flush_storage()
                self.fit_sensor_generation += 1
                self.async_set_updated_data(self._build_snapshot_payload())

    async def _async_refresh_eval_only(self) -> None:
        async with self._processing_lock:
            if self.ringbuffer.row_count == 0:
                return
            cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
            await self.hass.async_add_executor_job(
                self.engine.refresh_14d_evaluation, cols
            )
            self.fit_sensor_generation += 1
            self.async_set_updated_data(self._build_snapshot_payload())

    async def _async_on_5m_tick(self, now: datetime) -> None:
        async with self._processing_lock:
            if self._is_shut_down:
                return

            t_mono = time.monotonic()
            aligned_utc = int(round(now.timestamp() / DT_STEP_S)) * DT_STEP_S
            accepted, ts_utc, dt_mono = self.clock_guard.check_and_advance(
                aligned_utc, t_mono
            )
            if not accepted:
                self.accumulator.close_window(ts_utc=ts_utc, t_mono=t_mono)
                _LOGGER.warning(
                    "ClockJumpGuard skipped pre-NTP or non-monotonic 5m tick "
                    "(wall_utc=%d, last_accepted=%d).",
                    ts_utc,
                    self.clock_guard.last_accepted_utc,
                )
                return

            closed: WindowCloseResult = self.accumulator.close_window(
                ts_utc=ts_utc, t_mono=t_mono
            )
            prev_ring_ts = self.ringbuffer.latest_ts_utc
            if prev_ring_ts > 0 and ts_utc > prev_ring_ts:
                wall_dt = ts_utc - prev_ring_ts
                step_dt_s = (
                    float(DT_STEP_S)
                    if wall_dt <= 330
                    else float(wall_dt)
                )
            elif dt_mono <= 330.0:
                step_dt_s = float(DT_STEP_S)
            elif dt_mono <= MAX_RESTART_GAP_BRIDGE_S:
                step_dt_s = float(round(dt_mono))
            else:
                step_dt_s = dt_mono
            raw_row = closed.raw_row
            if step_dt_s > 450.0 and step_dt_s <= MAX_RESTART_GAP_BRIDGE_S:
                raw_row["flags"] = int(raw_row["flags"]) | FLAG_GAP_BRIDGED

            outputs = self.engine.process_5m_window(
                raw_row,
                temp_changed_wk=closed.temp_changed_wk,
                hum_changed_wk=closed.hum_changed_wk,
                stale_wk=closed.stale_wk,
                temp_changed_kt=closed.temp_changed_kt,
                hum_changed_kt=closed.hum_changed_kt,
                stale_kt=closed.stale_kt,
                dt_s=step_dt_s,
            )
            self._apply_engine_outputs_to_row(raw_row, outputs)
            self.ringbuffer.append_row(raw_row)
            self.ringbuffer.update_last_row_outputs(outputs)
            self.last_raw_row = raw_row

            self.async_set_updated_data(self._build_snapshot_payload())

    def _seed_last_raw_row_from_ring(self) -> None:
        """Populate `last_raw_row` from the most recent stored ringbuffer row."""
        if self.ringbuffer.row_count == 0:
            return
        tail = self.ringbuffer.get_columns(max_rows=1)
        if tail.get("ts_utc"):
            self.last_raw_row = {
                col: seq[-1] for col, seq in tail.items() if seq
            }

    def _apply_engine_outputs_to_row(
        self,
        raw_row: dict[str, float | int],
        outputs: Mapping[str, float | int | None],
    ) -> None:
        raw_row.update(outputs)  # type: ignore[arg-type]
        snap_wk = self.engine.get_zone_snapshot("wk")
        snap_kt = self.engine.get_zone_snapshot("kt")
        flags = int(raw_row.get("flags", 0))
        if snap_wk.open_window_active:
            flags |= FLAG_OPEN_WIN_WK
        if snap_kt.open_window_active:
            flags |= FLAG_OPEN_WIN_KT
        raw_row["flags"] = flags

    async def _async_on_6h_maintenance(self, now: datetime) -> None:
        del now  # Unused.
        async with self._processing_lock:
            if self._is_shut_down:
                return
            if self.ringbuffer.row_count > 0:
                cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
                await self.hass.async_add_executor_job(
                    self.engine.refresh_14d_evaluation, cols
                )
            await self.async_flush_storage()
            self.fit_sensor_generation += 1
            self.async_set_updated_data(self._build_snapshot_payload())

    async def _async_on_weekly_calibration_slot(self, now: datetime) -> None:
        local_now = dt_util.as_local(now)
        if local_now.weekday() != 6:  # 6 == Sunday
            return
        if not self.auto_calibrate_enabled:
            _LOGGER.debug(
                "Skipping Sunday 03:15 auto-calibration (switch is off)."
            )
            return

        async with self._processing_lock:
            if self._is_shut_down or self.ringbuffer.row_count == 0:
                return
            cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
            await self.hass.async_add_executor_job(
                self._sync_calibrate_and_eval, cols, "auto_weekly"
            )
            await self.async_flush_storage()
            self.fit_sensor_generation += 1
            self.async_set_updated_data(self._build_snapshot_payload())

    def _sync_calibrate_and_eval(
        self, cols: dict[str, list[float | int]], mode: str
    ) -> dict[str, Any]:
        outcome = self.engine.run_calibration(cols, mode=mode)  # type: ignore[arg-type]
        if outcome.get("blocked_by_seasonal_gate") and cols.get("ts_utc"):
            self.engine.refresh_14d_evaluation(cols)
        return outcome

    def _sync_rollback_and_eval(
        self, cols: dict[str, list[float | int]]
    ) -> bool:
        return self.engine.rollback_parameters(cols)

    def _sync_reset_and_eval(
        self, cols: dict[str, list[float | int]]
    ) -> None:
        self.engine.reset_parameters(cols)

    async def async_set_auto_calibrate(
        self, enabled: bool, *, persist: bool = True
    ) -> None:
        """Toggle automatic Sunday `03:15` MAP calibration."""
        self.auto_calibrate_enabled = bool(enabled)
        self.has_persisted_auto_cal_pref = True
        if persist:
            await self._async_save_state_only()
        self.async_set_updated_data(self._build_snapshot_payload())

    async def async_trigger_manual_calibration(self) -> dict[str, Any]:
        """Run manual MAP calibration in executor and refresh fit sensors."""
        async with self._processing_lock:
            cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
            outcome = await self.hass.async_add_executor_job(
                self._sync_calibrate_and_eval, cols, "manual"
            )
            await self.async_flush_storage()
            self.fit_sensor_generation += 1
            self.async_set_updated_data(self._build_snapshot_payload())
            return outcome

    async def async_trigger_rollback(self) -> bool:
        """Restore previous parameter snapshot from stack."""
        async with self._processing_lock:
            cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
            rolled_back = await self.hass.async_add_executor_job(
                self._sync_rollback_and_eval, cols
            )
            await self.async_flush_storage()
            self.fit_sensor_generation += 1
            self.async_set_updated_data(self._build_snapshot_payload())
            return rolled_back

    async def async_trigger_reset(self) -> None:
        """Reset parameters to Day-1 building priors."""
        async with self._processing_lock:
            cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
            await self.hass.async_add_executor_job(
                self._sync_reset_and_eval, cols
            )
            await self.async_flush_storage()
            self.fit_sensor_generation += 1
            self.async_set_updated_data(self._build_snapshot_payload())

    async def async_trigger_export_csv(
        self, export_dir: str = EXPORT_DIR_DEFAULT
    ) -> tuple[str, str]:
        """Atomically export `telemetry_5m.csv` and `fit_report.json`."""
        async with self._processing_lock:
            cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
        csv_path, json_path = await self.hass.async_add_executor_job(
            self.engine.export_diagnostics, export_dir, cols
        )
        _LOGGER.info(
            "Exported BetonBatterij diagnostics to %s and %s",
            csv_path,
            json_path,
        )
        return csv_path, json_path

    async def async_update_flow_rates(
        self, flow_rates_l_min: Mapping[str, float]
    ) -> None:
        """Apply updated manifold rotameter flow rates from OptionsFlow."""
        async with self._processing_lock:
            self.flow_rates_l_min = {
                group: float(
                    flow_rates_l_min.get(group, DEFAULT_FLOW_RATES_L_MIN[group])
                )
                for group in ALL_HYDRAULIC_GROUPS
            }
            self.engine.update_flow_rates(self.flow_rates_l_min)
            await self._async_save_state_only()
            self.async_set_updated_data(self._build_snapshot_payload())

    async def _async_save_state_only(self) -> None:
        engine_dict = dict(self.engine.to_state_dict())
        episode_bank = engine_dict.pop("episode_bank", None)
        if episode_bank is None and hasattr(self.engine, "episode_bank"):
            eb_attr = getattr(self.engine, "episode_bank")
            if hasattr(eb_attr, "to_dict"):
                episode_bank = eb_attr.to_dict()
            elif isinstance(eb_attr, (dict, list)):
                episode_bank = eb_attr

        state_doc: dict[str, Any] = {
            "last_ts_utc": max(
                self.clock_guard.last_accepted_utc,
                self.ringbuffer.latest_ts_utc,
            ),
            "auto_calibrate_enabled": self.auto_calibrate_enabled,
            "engine": engine_dict,
        }
        await self._state_store.async_save(state_doc)
        if episode_bank is not None:
            await self._episode_store.async_save({"episode_bank": episode_bank})

    async def async_flush_storage(self) -> None:
        """Persist dirty weekly ringbuffer chunks, engine state, and episodes."""
        await self.ring_store.async_save_dirty()
        await self._async_save_state_only()

    async def _async_on_ha_stop(self, event: Event) -> None:
        del event  # Unused.
        self._stop_unsub = None
        await self.async_shutdown()

    async def async_shutdown(self) -> None:
        """Unsubscribe listeners and flush all in-memory state cleanly."""
        if self._is_shut_down:
            return
        self._is_shut_down = True

        self.accumulator.stop()
        while self._unsubs:
            unsub = self._unsubs.pop()
            unsub()
        if self._stop_unsub is not None:
            self._stop_unsub()
            self._stop_unsub = None
        if self._backfill_task is not None and not self._backfill_task.done():
            self._backfill_task.cancel()

        try:
            await self.async_flush_storage()
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning(
                "Error flushing BetonBatterij storage on shutdown: %s", err
            )

    @callback
    def _build_snapshot_payload(self) -> dict[str, Any]:
        return {
            "wk": self.engine.get_zone_snapshot("wk"),
            "kt": self.engine.get_zone_snapshot("kt"),
            "fit_sensor_generation": self.fit_sensor_generation,
            "auto_calibrate_enabled": self.auto_calibrate_enabled,
            "ringbuffer_rows": self.ringbuffer.row_count,
        }
