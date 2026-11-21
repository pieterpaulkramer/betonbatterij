"""Cold-start historical backfill from Home Assistant Recorder & LTS.

Populates an empty `ColumnarRingBuffer` on first installation without blocking
Home Assistant startup by replaying up to 90 days of Long-Term Statistics and
10 days of raw state changes onto a strict 300 s UTC grid.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..const import (
    AC_MODE_OFF,
    DT_STEP_S,
    FLAG_AANWEZIG,
    FLAG_OPEN_WIN_KT,
    FLAG_OPEN_WIN_WK,
    FLAG_PUMP_SEC_ON,
    FLAG_VALID_KT,
    FLAG_VALID_WK,
    RINGBUFFER_DAYS,
    RINGBUFFER_ROWS,
)
from ..house_config import (
    HOUSE_ENTITIES,
    ROOM_SENSORS,
    HouseEntityConfig,
    RoomSensorConfig,
)
from .accumulator import WindowAccumulatorCore, convert_power_to_w
from .storage_ring import ColumnarRingBuffer

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from ..domain.engine import DigitalTwinEngine
    from .storage_ring import HARingBufferStore

_LOGGER = logging.getLogger(__name__)

RAW_HISTORY_DAYS: int = 10


@dataclass(frozen=True, slots=True)
class HistoricalStateEvent:
    """Lightweight pure-Python state event extracted from HA Recorder."""

    ts_utc: float
    entity_id: str
    state_str: str | None
    attributes: Mapping[str, Any]
    last_changed_utc: float
    last_reported_utc: float


class _LtsSeriesLookup:
    """Sorted `(ts_utc, value)` lookup with linear/step-hold interpolation."""

    __slots__ = ("ts_list", "val_list")

    def __init__(self, points: list[tuple[int, float]]) -> None:
        points.sort(key=lambda item: item[0])
        dedup_ts: list[int] = []
        dedup_val: list[float] = []
        for ts_i, val_f in points:
            if not math.isfinite(val_f):
                continue
            if dedup_ts and dedup_ts[-1] == ts_i:
                dedup_val[-1] = val_f
            else:
                dedup_ts.append(ts_i)
                dedup_val.append(val_f)
        self.ts_list = dedup_ts
        self.val_list = dedup_val

    def empty(self) -> bool:
        return not self.ts_list

    def earliest_ts(self) -> int | None:
        return self.ts_list[0] if self.ts_list else None

    def sample(self, ts_utc: int, max_hold_s: int = 7200) -> float | None:
        if not self.ts_list:
            return None
        idx = bisect.bisect_right(self.ts_list, ts_utc)
        if idx == 0:
            return None
        t0 = self.ts_list[idx - 1]
        v0 = self.val_list[idx - 1]
        if ts_utc - t0 > max_hold_s:
            return None
        if idx < len(self.ts_list):
            t1 = self.ts_list[idx]
            v1 = self.val_list[idx]
            span = t1 - t0
            if 0 < span <= max_hold_s:
                frac = (ts_utc - t0) / span
                return v0 + frac * (v1 - v0)
        return v0


def replay_backfill_into_engine(
    engine: DigitalTwinEngine,
    ringbuffer: ColumnarRingBuffer,
    lts_series: Mapping[str, list[tuple[int, float]]],
    lts_units: Mapping[str, str | None],
    raw_events: Sequence[HistoricalStateEvent],
    lts_start_ts: int,
    raw_start_ts: int,
    end_ts: int,
    entities: HouseEntityConfig = HOUSE_ENTITIES,
    room_sensors: Mapping[str, RoomSensorConfig] = ROOM_SENSORS,
) -> int:
    """Replay LTS and raw historical state events onto a 300 s UTC grid.

    Args:
        engine: Target `DigitalTwinEngine` instance to step forward.
        ringbuffer: Target `ColumnarRingBuffer` to populate.
        lts_series: Mapping of entity ID to `(ts_utc, mean_value)` points.
        lts_units: Mapping of entity ID to `unit_of_measurement`.
        raw_events: Raw recorder state events over `[raw_start_ts, end_ts]`.
        lts_start_ts: Earliest 300 s UTC timestamp for Long-Term Statistics.
        raw_start_ts: Boundary UTC timestamp where raw state replay begins.
        end_ts: Final 300 s UTC timestamp to reconstruct.
        entities: Configured Home Assistant entity mapping.
        room_sensors: Per-room ZHA sensor mapping including swap history.

    Returns:
        Total number of 5-minute rows appended to `ringbuffer`.
    """
    lookups: dict[str, _LtsSeriesLookup] = {
        eid: _LtsSeriesLookup(list(pts))
        for eid, pts in lts_series.items()
        if pts
    }

    rows_added = 0

    earliest_candidates = [
        lkp.earliest_ts() for lkp in lookups.values() if not lkp.empty()
    ]
    earliest_lts = min(
        (ts for ts in earliest_candidates if ts is not None), default=None
    )
    if earliest_lts is not None and earliest_lts < raw_start_ts:
        grid_start = max(
            lts_start_ts, ((earliest_lts // DT_STEP_S) + 1) * DT_STEP_S
        )
        prev_buf_top: float | None = None
        prev_buf_bot: float | None = None
        prev_wk_temp: float | None = None
        prev_kt_temp: float | None = None

        for ts in range(grid_start, raw_start_ts, DT_STEP_S):
            row = _build_lts_row_at(
                ts=ts,
                lookups=lookups,
                lts_units=lts_units,
                entities=entities,
                room_sensors=room_sensors,
                prev_buf_top=prev_buf_top,
                prev_buf_bot=prev_buf_bot,
            )
            if row is None:
                continue

            if math.isfinite(float(row["t_buf_top_end_c"])):
                prev_buf_top = float(row["t_buf_top_end_c"])
            if math.isfinite(float(row["t_buf_bot_end_c"])):
                prev_buf_bot = float(row["t_buf_bot_end_c"])

            t_wk = float(row["t_air_wk_c"])
            t_kt = float(row["t_air_kantoor_c"])
            valid_wk = bool(int(row["flags"]) & FLAG_VALID_WK)
            valid_kt = bool(int(row["flags"]) & FLAG_VALID_KT)
            temp_chg_wk = valid_wk and (
                prev_wk_temp is None or abs(t_wk - prev_wk_temp) > 1e-4
            )
            temp_chg_kt = valid_kt and (
                prev_kt_temp is None or abs(t_kt - prev_kt_temp) > 1e-4
            )
            if valid_wk:
                prev_wk_temp = t_wk
            if valid_kt:
                prev_kt_temp = t_kt

            outputs = engine.process_5m_window(
                row,
                temp_changed_wk=temp_chg_wk,
                hum_changed_wk=valid_wk,
                stale_wk=not valid_wk,
                temp_changed_kt=temp_chg_kt,
                hum_changed_kt=valid_kt,
                stale_kt=not valid_kt,
                dt_s=float(DT_STEP_S),
            )
            _apply_backfill_outputs(engine, row, outputs)
            ringbuffer.append_row(row)
            rows_added += 1

    if raw_events and raw_start_ts < end_ts:
        sorted_events = sorted(raw_events, key=lambda ev: ev.ts_utc)
        first_ev_ts = int(sorted_events[0].ts_utc)
        hist_grid_start = max(
            raw_start_ts, (first_ev_ts // DT_STEP_S) * DT_STEP_S
        )
        base_ts = float(hist_grid_start)
        acc = WindowAccumulatorCore(
            t_mono=0.0, entities=entities, room_sensors=room_sensors
        )

        ev_idx = 0
        n_ev = len(sorted_events)

        while ev_idx < n_ev and sorted_events[ev_idx].ts_utc <= base_ts:
            ev = sorted_events[ev_idx]
            acc.ingest_state(
                entity_id=ev.entity_id,
                state_str=ev.state_str,
                attributes=ev.attributes,
                t_mono=0.0,
                last_changed_utc=ev.last_changed_utc,
                last_reported_utc=ev.last_reported_utc,
                is_initial_seed=True,
                event_wall_utc=ev.ts_utc,
            )
            ev_idx += 1

        for window_end_ts in range(
            hist_grid_start + DT_STEP_S, end_ts + DT_STEP_S, DT_STEP_S
        ):
            while ev_idx < n_ev and sorted_events[ev_idx].ts_utc <= window_end_ts:
                ev = sorted_events[ev_idx]
                t_rel = max(0.0, ev.ts_utc - base_ts)
                acc.ingest_state(
                    entity_id=ev.entity_id,
                    state_str=ev.state_str,
                    attributes=ev.attributes,
                    t_mono=t_rel,
                    last_changed_utc=ev.last_changed_utc,
                    last_reported_utc=ev.last_reported_utc,
                    is_initial_seed=False,
                    event_wall_utc=ev.ts_utc,
                )
                ev_idx += 1

            win_mono = float(window_end_ts) - base_ts
            closed = acc.close_window(
                ts_utc=window_end_ts,
                t_mono=win_mono,
                window_dt=float(DT_STEP_S),
            )
            row = closed.raw_row
            if (
                not int(row["flags"]) & (FLAG_VALID_WK | FLAG_VALID_KT)
                and rows_added == 0
            ):
                continue

            outputs = engine.process_5m_window(
                row,
                temp_changed_wk=closed.temp_changed_wk,
                hum_changed_wk=closed.hum_changed_wk,
                stale_wk=closed.stale_wk,
                temp_changed_kt=closed.temp_changed_kt,
                hum_changed_kt=closed.hum_changed_kt,
                stale_kt=closed.stale_kt,
                dt_s=float(DT_STEP_S),
            )
            _apply_backfill_outputs(engine, row, outputs)
            ringbuffer.append_row(row)
            rows_added += 1

    if ringbuffer.row_count > 0:
        cols = ringbuffer.get_columns(RINGBUFFER_ROWS)
        engine.refresh_14d_evaluation(cols)

    return rows_added


def _apply_backfill_outputs(
    engine: DigitalTwinEngine,
    row: dict[str, float | int],
    outputs: Mapping[str, float | int],
) -> None:
    """Merge engine outputs and open-window flags into a backfilled row."""
    row.update(outputs)
    if hasattr(engine, "get_zone_snapshot"):
        flags = int(row.get("flags", 0))
        if engine.get_zone_snapshot("wk").open_window_active:
            flags |= FLAG_OPEN_WIN_WK
        if engine.get_zone_snapshot("kt").open_window_active:
            flags |= FLAG_OPEN_WIN_KT
        row["flags"] = flags


def _build_lts_row_at(
    ts: int,
    lookups: Mapping[str, _LtsSeriesLookup],
    lts_units: Mapping[str, str | None],
    entities: HouseEntityConfig,
    room_sensors: Mapping[str, RoomSensorConfig],
    prev_buf_top: float | None,
    prev_buf_bot: float | None,
) -> dict[str, float | int] | None:
    def _s(eid: str | None, default: float = float("nan")) -> float:
        if not eid:
            return default
        lkp = lookups.get(eid)
        if lkp is None:
            return default
        val = lkp.sample(ts)
        return val if val is not None else default

    def _room_t(room_key: str) -> float:
        cfg = room_sensors.get(room_key)
        if cfg is None:
            return float("nan")
        if cfg.swap_timestamp_utc is not None and ts < cfg.swap_timestamp_utc:
            if cfg.pre_swap_temp_entity_id:
                return _s(cfg.pre_swap_temp_entity_id)
            return float("nan")
        return _s(cfg.temp_entity_id)

    t_wk = _room_t("wk")
    t_kt = _room_t("kantoor")
    t_dak = _s(entities.outdoor_temp_dak)
    t_hm = _s(entities.wp_outside_temp)
    any_out = math.isfinite(t_dak) or math.isfinite(t_hm)
    if not any_out or (not math.isfinite(t_wk) and not math.isfinite(t_kt)):
        return None

    v_prim = max(0.0, _s(entities.wp_pump_flow, 0.0))
    t_wp_out = _s(entities.wp_outlet_temp)
    t_wp_in = _s(entities.wp_inlet_temp)
    if math.isfinite(t_wp_out) and math.isfinite(t_wp_in):
        p_prim_th_w = 69.73 * v_prim * (t_wp_out - t_wp_in)
    else:
        p_prim_th_w = 0.0

    wp_el_raw = _s(entities.wp_power_consumption, 0.0)
    p_wp_el_w = max(
        0.0,
        convert_power_to_w(
            wp_el_raw, lts_units.get(entities.wp_power_consumption), "W"
        ),
    )

    p1_cons_raw = _s(entities.p1_power_consumed_kw, float("nan"))
    p1_prod_raw = _s(entities.p1_power_produced_kw, float("nan"))
    if math.isfinite(p1_cons_raw) or math.isfinite(p1_prod_raw):
        cons_w = (
            convert_power_to_w(
                p1_cons_raw,
                lts_units.get(entities.p1_power_consumed_kw),
                "kW",
            )
            if math.isfinite(p1_cons_raw)
            else 0.0
        )
        prod_w = (
            convert_power_to_w(
                p1_prod_raw,
                lts_units.get(entities.p1_power_produced_kw),
                "kW",
            )
            if math.isfinite(p1_prod_raw)
            else 0.0
        )
        p_p1_net_w = cons_w - prod_w
    else:
        p_p1_net_w = float("nan")

    pv_raw = _s(entities.pv_power_w, 0.0)
    p_pv_w = max(
        0.0, convert_power_to_w(pv_raw, lts_units.get(entities.pv_power_w), "W")
    )
    ac_raw = _s(entities.ac_kantoor_power_w, 0.0)
    p_ac_w = max(
        0.0,
        convert_power_to_w(
            ac_raw, lts_units.get(entities.ac_kantoor_power_w), "W"
        ),
    )
    grow_raw = _s(entities.growlights_kantoor_power_w, 0.0)
    p_grow_w = max(
        0.0,
        convert_power_to_w(
            grow_raw, lts_units.get(entities.growlights_kantoor_power_w), "W"
        ),
    )
    desk_raw = _s(entities.desk_kantoor_power_w)
    p_desk_w = (
        max(
            0.0,
            convert_power_to_w(
                desk_raw,
                lts_units.get(entities.desk_kantoor_power_w or ""),
                "W",
            ),
        )
        if math.isfinite(desk_raw)
        else float("nan")
    )

    buf_top_end = _s(entities.buf_top_temp)
    buf_top_start = (
        prev_buf_top if prev_buf_top is not None else buf_top_end
    )
    buf_bot_end = _s(entities.buf_bot_temp)
    buf_bot_start = (
        prev_buf_bot if prev_buf_bot is not None else buf_bot_end
    )

    heating_active = p_prim_th_w > 250.0
    u_pump = 1.0 if heating_active else 0.0
    u_bg = 1.0 if heating_active else 0.0

    flags = FLAG_AANWEZIG
    if math.isfinite(t_wk):
        flags |= FLAG_VALID_WK
    if math.isfinite(t_kt):
        flags |= FLAG_VALID_KT
    if u_pump > 0.01:
        flags |= FLAG_PUMP_SEC_ON

    return {
        "ts_utc": ts,
        "flags": flags,
        "u_eff_bg": u_bg,
        "u_eff_slaapkamer": 0.0,
        "u_eff_badkamer": 0.0,
        "u_eff_werkkamer_1e": 0.0,
        "u_eff_kamer_1e_midden": 0.0,
        "u_eff_kamer_1e_zuid": 0.0,
        "u_eff_kantoor": 0.0,
        "u_eff_badkamer_boven": 0.0,
        "u_eff_logeerkamer": 0.0,
        "u_pump_sec": u_pump,
        "t_air_wk_c": t_wk,
        "t_air_slaapkamer_c": _room_t("slaapkamer"),
        "t_air_badkamer_c": _room_t("badkamer"),
        "t_air_werkkamer_1e_c": _room_t("werkkamer_1e"),
        "t_air_kamer_1e_midden_c": _room_t("kamer_1e_midden"),
        "t_air_kamer_1e_zuid_c": _room_t("kamer_1e_zuid"),
        "t_air_kantoor_c": t_kt,
        "t_air_badkamer_boven_c": _room_t("badkamer_boven"),
        "t_air_logeerkamer_c": _room_t("logeerkamer"),
        "v_prim_l_min": v_prim,
        "t_wp_out_c": t_wp_out,
        "t_wp_in_c": t_wp_in,
        "p_prim_th_w": p_prim_th_w,
        "f_comp_hz": max(0.0, _s(entities.wp_compressor_freq, 0.0)),
        "p_wp_raw_el_w": p_wp_el_w,
        "t_buf_top_start_c": buf_top_start,
        "t_buf_top_end_c": buf_top_end,
        "t_buf_bot_start_c": buf_bot_start,
        "t_buf_bot_end_c": buf_bot_end,
        "t_dhw_top_c": _s(entities.dhw_top_temp),
        "t_dhw_bot_c": _s(entities.dhw_bot_temp),
        "p_p1_net_w": p_p1_net_w,
        "t_out_dak_c": t_dak,
        "t_out_weather_c": float("nan"),
        "t_out_hm_c": t_hm,
        "rh_out_pct": _s(entities.outdoor_humidity_dak, 80.0),
        "v_wind_ms": 3.0,
        "p_pv_w": p_pv_w,
        "p_pv_forecast_w": max(0.0, _s(entities.pv_forecast_now_w, 0.0)),
        "epex_eur_kwh": _s(entities.epex_price_eur_kwh),
        "screen_small_pos": 100.0,
        "screen_large_pos": 100.0,
        "p_el_ac_kt_w": p_ac_w,
        "ac_mode_code": 1.0 if p_ac_w > 40.0 else AC_MODE_OFF,
        "p_desk_kt_w": p_desk_w,
        "pir_occ_kt": 0.0,
        "p_growlights_kt_w": p_grow_w,
        "t_bg_sup_c": _s(entities.bg_manifold_supply_temp),
        "t_bg_ret_c": (
            _s(entities.bg_manifold_return_temp)
            + entities.bg_manifold_return_offset_c
        ),
        "t_floor_surf_meas_wk_c": _s(entities.floor_surface_sensor_wk),
        "t_floor_wk_c": float("nan"),
        "d_est_wk_w": float("nan"),
        "q_fh_wk_w": float("nan"),
        "t_hind6h_wk_c": float("nan"),
        "t_floor_kt_c": float("nan"),
        "d_est_kt_w": float("nan"),
        "q_fh_kt_w": float("nan"),
        "t_hind6h_kt_c": float("nan"),
    }


def _extract_lts_point_ts(row: Mapping[str, Any]) -> int | None:
    start_val = row.get("start")
    if isinstance(start_val, (int, float)):
        return int(start_val)
    if isinstance(start_val, datetime):
        if start_val.tzinfo is None:
            return int(start_val.replace(tzinfo=timezone.utc).timestamp())
        return int(start_val.timestamp())
    return None


async def async_run_cold_start_backfill(
    hass: HomeAssistant,
    engine: DigitalTwinEngine,
    ringbuffer: ColumnarRingBuffer,
    ring_store: HARingBufferStore,
    now_utc: float,
    entities: HouseEntityConfig = HOUSE_ENTITIES,
    room_sensors: Mapping[str, RoomSensorConfig] = ROOM_SENSORS,
) -> int:
    """Query HA Recorder and LTS on executor threads to backfill `ringbuffer`."""
    if ringbuffer.row_count > 0:
        return 0

    if "recorder" not in hass.config.components:
        _LOGGER.warning(
            "Home Assistant 'recorder' integration is not active; "
            "skipping BetonBatterij cold-start backfill."
        )
        return 0

    try:
        from homeassistant.components.recorder import (
            get_instance,
            history as recorder_history,
            statistics as recorder_statistics,
        )
    except ImportError as err:
        _LOGGER.warning(
            "Recorder imports unavailable; skipping cold-start backfill: %s",
            err,
        )
        return 0

    end_ts = (int(now_utc) // DT_STEP_S) * DT_STEP_S
    raw_start_ts = end_ts - RAW_HISTORY_DAYS * 86400
    lts_start_ts = end_ts - RINGBUFFER_DAYS * 86400

    lts_start_dt = datetime.fromtimestamp(lts_start_ts, tz=timezone.utc)
    raw_start_dt = datetime.fromtimestamp(raw_start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    probe_acc = WindowAccumulatorCore(
        t_mono=0.0, entities=entities, room_sensors=room_sensors
    )
    watched_ids = probe_acc.watched_entity_ids()

    def _query_recorder_db() -> tuple[
        dict[str, list[tuple[int, float]]],
        dict[str, str | None],
        list[HistoricalStateEvent],
    ]:
        lts_points: dict[str, list[tuple[int, float]]] = {}
        lts_units: dict[str, str | None] = {}

        for eid in watched_ids:
            st = hass.states.get(eid)
            if st is not None:
                lts_units[eid] = st.attributes.get("unit_of_measurement")

        for period in ("hour", "5minute"):
            try:
                stats_map = recorder_statistics.statistics_during_period(
                    hass,
                    lts_start_dt,
                    raw_start_dt,
                    set(watched_ids),
                    period,
                    None,
                    {"mean", "state"},
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.debug(
                    "LTS query (%s) skipped during backfill: %s", period, exc
                )
                continue

            for eid, stat_rows in (stats_map or {}).items():
                bucket = lts_points.setdefault(eid, [])
                for srow in stat_rows:
                    ts_i = _extract_lts_point_ts(srow)
                    val = srow.get("mean")
                    if val is None:
                        val = srow.get("state")
                    if ts_i is not None and isinstance(val, (int, float)):
                        bucket.append((ts_i, float(val)))

        raw_events: list[HistoricalStateEvent] = []
        try:
            states_map = recorder_history.get_significant_states(
                hass,
                raw_start_dt,
                end_dt,
                entity_ids=watched_ids,
                include_start_time_state=True,
                significant_changes_only=False,
                minimal_response=False,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "Recorder raw history query failed during backfill: %s", exc
            )
            states_map = {}

        for eid, state_list in (states_map or {}).items():
            for st in state_list:
                lc_dt = getattr(st, "last_changed", None) or getattr(
                    st, "last_updated", None
                )
                lr_dt = (
                    getattr(st, "last_reported", None)
                    or getattr(st, "last_updated", None)
                    or lc_dt
                )
                if lc_dt is None:
                    continue
                lc_ts = (
                    lc_dt.replace(tzinfo=timezone.utc).timestamp()
                    if lc_dt.tzinfo is None
                    else lc_dt.timestamp()
                )
                lr_ts = (
                    lr_dt.replace(tzinfo=timezone.utc).timestamp()
                    if lr_dt is not None and lr_dt.tzinfo is None
                    else (lr_dt.timestamp() if lr_dt is not None else lc_ts)
                )
                attrs = dict(getattr(st, "attributes", None) or {})
                raw_events.append(
                    HistoricalStateEvent(
                        ts_utc=lr_ts,
                        entity_id=eid,
                        state_str=getattr(st, "state", None),
                        attributes=attrs,
                        last_changed_utc=lc_ts,
                        last_reported_utc=lr_ts,
                    )
                )

        return lts_points, lts_units, raw_events

    try:
        rec_instance = get_instance(hass)
        lts_points, lts_units, raw_events = (
            await rec_instance.async_add_executor_job(_query_recorder_db)
        )
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning(
            "BetonBatterij cold-start recorder query failed gracefully: %s",
            err,
        )
        return 0

    if not lts_points and not raw_events:
        _LOGGER.warning(
            "Recorder returned no historical data for BetonBatterij entities; "
            "starting fresh in shadow mode."
        )
        return 0

    try:
        rows_added = await hass.async_add_executor_job(
            replay_backfill_into_engine,
            engine,
            ringbuffer,
            lts_points,
            lts_units,
            raw_events,
            lts_start_ts,
            raw_start_ts,
            end_ts,
            entities,
            room_sensors,
        )
        if rows_added > 0:
            await ring_store.async_save_dirty()
            _LOGGER.info(
                "BetonBatterij cold-start backfill completed: %d 5-minute rows "
                "reconstructed.",
                rows_added,
            )
        return rows_added
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning(
            "BetonBatterij cold-start backfill replay failed gracefully: %s",
            err,
        )
        return 0
