"""Five-minute tick coordinator, storage persistence, and background jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
import contextlib
from datetime import datetime
import logging
import math
from pathlib import Path
import time
from typing import Any, Literal

from ..const import (
    ALL_HYDRAULIC_GROUPS,
    DEFAULT_FLOW_RATES_L_MIN,
    DOMAIN,
    DT_STEP_S,
    EXPORT_DIR_DEFAULT,
    FLAG_GAP_BRIDGED,
    MAX_RESTART_BRIDGE_GAP_S,
    RINGBUFFER_ROWS,
)
from ..domain.contracts import (
    BufferSensorOffsets,
    ZoneEvaluationReport,
    ZoneParameters,
)
from ..domain.engine import DigitalTwinEngine
from ..domain.evaluation import SPINUP_48H_STEPS, evaluate_14d_hindcast
from ..domain.priors import (
    DAY1_BUFFER_OFFSETS,
    DAY1_PRIORS_KT,
    DAY1_PRIORS_WK,
)
from ..house_config import (
    HOUSE_ENTITIES,
    LOCAL_CONFIG_FILENAME,
    ROOM_SENSORS,
    HouseEntityConfig,
    RoomSensorConfig,
    load_house_config,
)
from ._ha_compat import (
    ConfigEntry,
    DataUpdateCoordinator,
    EVENT_HOMEASSISTANT_STOP,
    Event,
    HomeAssistant,
    async_track_time_change,
    async_track_utc_time_change,
    callback,
    dt_util,
)
from .accumulator import HAWindowAccumulator, WindowCloseResult
from .backfill import async_run_cold_start_backfill
from .clock import ClockJumpGuard
from .persistence import BetonBatterijPersistence
from .row_utils import (
    apply_engine_outputs_to_row,
    extract_flow_rates_from_entry,
)
from .storage_ring import ColumnarRingBuffer

_LOGGER = logging.getLogger(__name__)

__all__ = ["BetonBatterijCoordinator"]


def _load_local_config_sync(
    cfg_path: Path,
) -> tuple[HouseEntityConfig, dict[str, RoomSensorConfig], bool]:
    """Load optional `/config/betonbatterij_local.json` off the event loop."""
    if not cfg_path.is_file():
        return HOUSE_ENTITIES, dict(ROOM_SENSORS), False
    entities, room_sensors = load_house_config(cfg_path)
    return entities, room_sensors, True


def _compute_14d_eval_reports_sync(
    cols: Mapping[str, Sequence[float | int]],
    params_wk: ZoneParameters,
    params_kt: ZoneParameters,
    flow_rates_l_min: Mapping[str, float],
    lat_deg: float,
    lon_deg: float,
    k_hx_by_group: Mapping[str, float] | None = None,
    buffer_offsets: BufferSensorOffsets = DAY1_BUFFER_OFFSETS,
) -> tuple[ZoneEvaluationReport, ZoneEvaluationReport]:
    """Compute 14-day hindcast evaluation reports without mutating `engine`."""
    k_hx_map = dict(k_hx_by_group) if k_hx_by_group else {}
    k_hx_map["wk"] = params_wk.k_hx_w_k
    k_hx_map["kantoor"] = params_kt.k_hx_w_k
    rep_wk = evaluate_14d_hindcast(
        cols,
        "wk",
        params_wk,
        DAY1_PRIORS_WK,
        flow_rates_l_min=flow_rates_l_min,
        k_hx_by_group=k_hx_map,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        buffer_offsets=buffer_offsets,
    )
    rep_kt = evaluate_14d_hindcast(
        cols,
        "kt",
        params_kt,
        DAY1_PRIORS_KT,
        flow_rates_l_min=flow_rates_l_min,
        k_hx_by_group=k_hx_map,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        buffer_offsets=buffer_offsets,
    )
    return rep_wk, rep_kt


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
        self.persistence = BetonBatterijPersistence(hass, self.ringbuffer)
        self.ring_store = self.persistence.ring_store
        self._state_store = self.persistence.state_store
        self._episode_store = self.persistence.episode_store

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
        self._calibration_lock = asyncio.Lock()
        self._unsubs: list[Callable[[], None]] = []
        self._stop_unsub: Callable[[], None] | None = None
        self._backfill_task: asyncio.Task[None] | None = None
        self._eval_init_task: asyncio.Task[None] | None = None
        self._is_setup_done: bool = False
        self._is_shut_down: bool = False

    async def async_setup(self) -> None:
        """Set up stores, listeners, and initial coordinator data snapshot."""
        await self._async_setup()
        self.async_set_updated_data(await self._async_update_data())

    async def _async_setup(self) -> None:
        """Load state and ringbuffer once during first refresh."""
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

        state_payload, episode_payload, _ = (
            await self.persistence.async_load_all()
        )

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
                and DT_STEP_S <= gap_s <= MAX_RESTART_BRIDGE_GAP_S
            ):
                self._bridge_restart_gap(aligned_now, gap_s, t_mono_now)
            elif not self.last_raw_row:
                self._seed_last_raw_row_from_ring()

            if (
                self.engine.eval_report_wk is None
                or self.engine.eval_report_kt is None
            ):
                self._eval_init_task = self.hass.async_create_background_task(
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
        closed = self.accumulator.close_window(
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
            _LOGGER.warning(
                "Failed bridging restart gap (%d s): %s", gap_s, err
            )
            return

        apply_engine_outputs_to_row(raw_row, outputs, engine=self.engine)
        self.ringbuffer.append_row(raw_row)
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

    def _capture_eval_snapshot_unlocked(self) -> dict[str, Any]:
        """Snapshot immutable evaluation inputs while holding `_processing_lock`."""
        is_real_engine = isinstance(self.engine, DigitalTwinEngine)
        has_pure_eval = hasattr(self.engine, "compute_14d_evaluation_reports")
        ctx: dict[str, Any] = {}
        params_wk: ZoneParameters | None = None
        params_kt: ZoneParameters | None = None
        flows: dict[str, float] = {}
        k_hx_map: dict[str, float] = {}
        lat_deg = 0.0
        lon_deg = 0.0

        raw_buf = getattr(self.engine, "buffer_offsets", DAY1_BUFFER_OFFSETS)
        buf_offsets = (
            raw_buf
            if isinstance(raw_buf, BufferSensorOffsets)
            else BufferSensorOffsets.from_dict(
                raw_buf if isinstance(raw_buf, Mapping) else None,
                defaults=DAY1_BUFFER_OFFSETS,
            )
        )

        if has_pure_eval and hasattr(
            self.engine, "capture_calibration_context"
        ):
            ctx = self.engine.capture_calibration_context()
        elif is_real_engine:
            ctx = self.engine.capture_calibration_context()
            params_wk = ctx["params_wk"]
            params_kt = ctx["params_kt"]
            flows = dict(ctx["flow_rates_l_min"])
            k_hx_map = dict(ctx["k_hx_by_group_w_k"])
            lat_deg = float(ctx["latitude_deg"])
            lon_deg = float(ctx["longitude_deg"])

        return {
            "is_real_engine": is_real_engine,
            "has_pure_eval": has_pure_eval,
            "ctx": ctx,
            "params_wk": params_wk,
            "params_kt": params_kt,
            "flows": flows,
            "k_hx_map": k_hx_map,
            "lat_deg": lat_deg,
            "lon_deg": lon_deg,
            "buffer_offsets": buf_offsets,
        }

    async def _async_compute_eval_reports_in_executor(
        self,
        cols: Mapping[str, Sequence[float | int]],
        snap: Mapping[str, Any],
    ) -> tuple[ZoneEvaluationReport | None, ZoneEvaluationReport | None]:
        """Execute pure 14d hindcast evaluation off the event loop."""
        if snap["has_pure_eval"]:
            ctx = snap["ctx"]
            eval_res = await self.hass.async_add_executor_job(
                self.engine.compute_14d_evaluation_reports,
                cols,
                ctx if ctx else None,
            )
            if isinstance(eval_res, Mapping):
                return eval_res.get("wk"), eval_res.get("kt")
            rep_wk, rep_kt = eval_res
            return rep_wk, rep_kt

        if (
            snap["is_real_engine"]
            and snap["params_wk"] is not None
            and snap["params_kt"] is not None
        ):
            return await self.hass.async_add_executor_job(
                _compute_14d_eval_reports_sync,
                cols,
                snap["params_wk"],
                snap["params_kt"],
                snap["flows"],
                snap["lat_deg"],
                snap["lon_deg"],
                snap["k_hx_map"],
                snap.get("buffer_offsets", DAY1_BUFFER_OFFSETS),
            )

        await self.hass.async_add_executor_job(
            self.engine.refresh_14d_evaluation, cols
        )
        return None, None

    def _apply_eval_reports(
        self,
        rep_wk: ZoneEvaluationReport | None,
        rep_kt: ZoneEvaluationReport | None,
    ) -> None:
        """Apply computed 14-day evaluation reports onto `self.engine`."""
        if rep_wk is None or rep_kt is None:
            return
        self.engine.apply_14d_evaluation_reports(rep_wk, rep_kt)

    async def _async_refresh_14d_eval(self, *, flush: bool) -> None:
        """Refresh 14-day hindcast evaluation reports outside `_processing_lock`."""
        async with self._calibration_lock:
            async with self._processing_lock:
                if self._is_shut_down:
                    return
                if self.ringbuffer.row_count == 0:
                    if flush:
                        await self.async_flush_storage()
                        self.fit_sensor_generation += 1
                        self.async_set_updated_data(
                            self._build_snapshot_payload()
                        )
                    return
                cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
                eval_snap = self._capture_eval_snapshot_unlocked()

            rep_wk, rep_kt = (
                await self._async_compute_eval_reports_in_executor(
                    cols, eval_snap
                )
            )

            async with self._processing_lock:
                if self._is_shut_down:
                    return
                self._apply_eval_reports(rep_wk, rep_kt)
                if flush:
                    await self.async_flush_storage()
                self.fit_sensor_generation += 1
                self.async_set_updated_data(self._build_snapshot_payload())

    async def _async_refresh_eval_only(self) -> None:
        await self._async_refresh_14d_eval(flush=False)

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
            elif dt_mono <= MAX_RESTART_BRIDGE_GAP_S:
                step_dt_s = float(round(dt_mono))
            else:
                step_dt_s = dt_mono
            raw_row = closed.raw_row
            if 450.0 < step_dt_s <= MAX_RESTART_BRIDGE_GAP_S:
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
            apply_engine_outputs_to_row(raw_row, outputs, engine=self.engine)
            self.ringbuffer.append_row(raw_row)
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

    async def _async_on_6h_maintenance(self, now: datetime) -> None:
        del now  # Unused.
        await self._async_refresh_14d_eval(flush=True)

    async def _async_on_weekly_calibration_slot(self, now: datetime) -> None:
        local_now = dt_util.as_local(now)
        if local_now.weekday() != 6:  # 6 == Sunday
            return
        if not self.auto_calibrate_enabled:
            _LOGGER.debug(
                "Skipping Sunday 03:15 auto-calibration (switch is off)."
            )
            return
        await self._async_run_calibration("auto_weekly")

    async def _async_run_calibration(
        self, mode: Literal["manual", "auto_weekly"]
    ) -> dict[str, Any]:
        """Run 3-phase MAP calibration with solver outside `_processing_lock`.

        Args:
            mode: Trigger mode (`"manual"` or `"auto_weekly"`).

        Returns:
            Calibration outcome dictionary from `DigitalTwinEngine`.
        """
        async with self._calibration_lock:
            async with self._processing_lock:
                if self._is_shut_down:
                    return {
                        "accepted_wk": False,
                        "accepted_kt": False,
                        "reason": "Coordinator shut down.",
                    }
                if mode == "auto_weekly" and self.ringbuffer.row_count == 0:
                    return {
                        "accepted_wk": False,
                        "accepted_kt": False,
                        "reason": "Empty ringbuffer.",
                    }
                cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
                use_three_phase = (
                    hasattr(self.engine, "capture_calibration_context")
                    and hasattr(self.engine, "compute_calibration_plan")
                    and hasattr(self.engine, "apply_calibration_plan")
                )
                ctx: dict[str, Any] = (
                    self.engine.capture_calibration_context()
                    if use_three_phase
                    else {}
                )

            t0 = time.perf_counter()
            plan: dict[str, Any] = {}
            fallback_outcome: dict[str, Any] = {}
            if use_three_phase:
                plan = await self.hass.async_add_executor_job(
                    self.engine.compute_calibration_plan, cols, ctx, mode
                )
            else:
                fallback_outcome = await self.hass.async_add_executor_job(
                    self._sync_calibrate_and_eval, cols, mode
                )
            _LOGGER.info(
                "BetonBatterij %s calibration solver completed in %.2f s",
                mode,
                time.perf_counter() - t0,
            )

            async with self._processing_lock:
                if self._is_shut_down:
                    return {
                        "accepted_wk": False,
                        "accepted_kt": False,
                        "reason": "Coordinator shut down.",
                    }
                if use_three_phase:
                    recent_cols = self.ringbuffer.get_columns(SPINUP_48H_STEPS)
                    outcome = self.engine.apply_calibration_plan(
                        plan, recent_cols
                    )
                else:
                    outcome = fallback_outcome
                await self.async_flush_storage()
                self.fit_sensor_generation += 1
                self.async_set_updated_data(self._build_snapshot_payload())
                return outcome

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
        async with self._processing_lock:
            self.auto_calibrate_enabled = bool(enabled)
            self.has_persisted_auto_cal_pref = True
            if persist:
                await self._async_save_state_only()
            self.async_set_updated_data(self._build_snapshot_payload())

    async def async_trigger_manual_calibration(self) -> dict[str, Any]:
        """Run manual MAP calibration in executor and refresh fit sensors."""
        return await self._async_run_calibration("manual")

    async def async_trigger_rollback(self) -> bool:
        """Restore previous parameter snapshot with executor eval unlocked."""
        async with self._calibration_lock:
            async with self._processing_lock:
                if self._is_shut_down:
                    return False
                cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
                supports_pure_eval = isinstance(
                    self.engine, DigitalTwinEngine
                ) or hasattr(self.engine, "compute_14d_evaluation_reports")
                if supports_pure_eval:
                    rolled_back = self.engine.rollback_parameters(columns=None)
                    if not rolled_back:
                        return False
                    recent_cols = self.ringbuffer.get_columns(SPINUP_48H_STEPS)
                    if recent_cols.get("ts_utc") and hasattr(
                        self.engine, "replay_48h_from_columns"
                    ):
                        self.engine.replay_48h_from_columns(recent_cols)
                    eval_snap = self._capture_eval_snapshot_unlocked()
                else:
                    eval_snap = {}

            rep_wk: ZoneEvaluationReport | None = None
            rep_kt: ZoneEvaluationReport | None = None
            if supports_pure_eval:
                if cols.get("ts_utc"):
                    rep_wk, rep_kt = (
                        await self._async_compute_eval_reports_in_executor(
                            cols, eval_snap
                        )
                    )
            else:
                rolled_back = await self.hass.async_add_executor_job(
                    self._sync_rollback_and_eval, cols
                )

            async with self._processing_lock:
                if self._is_shut_down or not rolled_back:
                    return False
                if supports_pure_eval:
                    self._apply_eval_reports(rep_wk, rep_kt)
                await self.async_flush_storage()
                self.fit_sensor_generation += 1
                self.async_set_updated_data(self._build_snapshot_payload())
                return rolled_back

    async def async_trigger_reset(self) -> None:
        """Reset parameters to Day-1 priors with executor eval unlocked."""
        async with self._calibration_lock:
            async with self._processing_lock:
                if self._is_shut_down:
                    return
                cols = self.ringbuffer.get_columns(RINGBUFFER_ROWS)
                supports_pure_eval = isinstance(
                    self.engine, DigitalTwinEngine
                ) or hasattr(self.engine, "compute_14d_evaluation_reports")
                if supports_pure_eval:
                    self.engine.reset_parameters(columns=None)
                    recent_cols = self.ringbuffer.get_columns(SPINUP_48H_STEPS)
                    if recent_cols.get("ts_utc") and hasattr(
                        self.engine, "replay_48h_from_columns"
                    ):
                        self.engine.replay_48h_from_columns(recent_cols)
                    eval_snap = self._capture_eval_snapshot_unlocked()
                else:
                    eval_snap = {}

            rep_wk: ZoneEvaluationReport | None = None
            rep_kt: ZoneEvaluationReport | None = None
            if supports_pure_eval:
                if cols.get("ts_utc"):
                    rep_wk, rep_kt = (
                        await self._async_compute_eval_reports_in_executor(
                            cols, eval_snap
                        )
                    )
            else:
                await self.hass.async_add_executor_job(
                    self._sync_reset_and_eval, cols
                )

            async with self._processing_lock:
                if self._is_shut_down:
                    return
                if supports_pure_eval:
                    self._apply_eval_reports(rep_wk, rep_kt)
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

    def _latest_ts_utc(self) -> int:
        return max(
            self.clock_guard.last_accepted_utc,
            self.ringbuffer.latest_ts_utc,
        )

    async def _async_save_state_only(self) -> None:
        await self.persistence.async_save_state_and_episodes(
            self.engine,
            last_ts_utc=self._latest_ts_utc(),
            auto_calibrate_enabled=self.auto_calibrate_enabled,
        )

    async def async_flush_storage(self) -> None:
        """Persist dirty weekly ringbuffer chunks, engine state, and episodes."""
        await self.persistence.async_flush_all(
            self.engine,
            last_ts_utc=self._latest_ts_utc(),
            auto_calibrate_enabled=self.auto_calibrate_enabled,
        )

    async def _async_on_ha_stop(self, event: Event) -> None:
        del event  # Unused.
        self._stop_unsub = None
        await self.async_shutdown()

    async def async_shutdown(self) -> None:
        """Unsubscribe listeners, cancel background tasks, and flush storage."""
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

        bg_tasks = [
            task
            for task in (self._backfill_task, self._eval_init_task)
            if task is not None and not task.done()
        ]
        for task in bg_tasks:
            task.cancel()
        for task in bg_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

        async with self._processing_lock:
            try:
                await self.async_flush_storage()
            except Exception as err:  # noqa: BLE001
                _LOGGER.warning(
                    "Error flushing BetonBatterij storage on shutdown: %s", err
                )

    @callback
    def _build_snapshot_payload(self) -> dict[str, Any]:
        return self.persistence.build_snapshot_payload(
            self.engine,
            fit_sensor_generation=self.fit_sensor_generation,
            auto_calibrate_enabled=self.auto_calibrate_enabled,
            ringbuffer_rows=self.ringbuffer.row_count,
        )
