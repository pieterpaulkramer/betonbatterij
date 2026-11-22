"""Offline pytest suite for accumulator, ringbuffer, backfill, and throttling.

Runs 100% offline on standard Python without `homeassistant` installed.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.betonbatterij.const import (
    AC_MODE_COOL,
    AC_MODE_HEAT,
    AC_MODE_OFF,
    AC_MODE_OTHER,
    DEFAULT_FLOW_RATES_L_MIN,
    DT_STEP_S,
    FLAG_AC_COOL_KT,
    FLAG_DHW_TANK,
    FLAG_LTS_BACKFILL,
    FLAG_VALID_KT,
    FLAG_VALID_WK,
    RINGBUFFER_ALL_COLS,
    RINGBUFFER_CHUNK_ROWS,
    RINGBUFFER_NUM_CHUNKS,
    RINGBUFFER_ROWS,
)
from custom_components.betonbatterij.ha_bridge.accumulator import (
    ClockJumpGuard,
    HydraulicPowerAccumulator,
    StepSignalAccumulator,
    WaxValveTracker,
    WindowAccumulatorCore,
    convert_power_to_w,
    convert_wind_speed_to_ms,
)
from custom_components.betonbatterij.ha_bridge.backfill import (
    HistoricalStateEvent,
    _LtsSeriesLookup,
    _build_lts_row_at,
    replay_backfill_into_engine,
)
from custom_components.betonbatterij.ha_bridge.coordinator import (
    BetonBatterijCoordinator,
)
from custom_components.betonbatterij.ha_bridge.storage_ring import (
    ColumnarRingBuffer,
)
from custom_components.betonbatterij.house_config import (
    HOUSE_ENTITIES,
    LATITUDE_DEG,
    LOCAL_CONFIG_FILENAME,
    LONGITUDE_DEG,
    ROOM_SENSORS,
    RoomSensorConfig,
    load_house_config,
)
from custom_components.betonbatterij.domain.engine import DigitalTwinEngine
from custom_components.betonbatterij import sensor as sensor_mod
from custom_components.betonbatterij.sensor import (
    BetonBatterijCopFitSensor,
    BetonBatterijZoneFitSensor,
    SmartThrottledSensor,
)


def test_benchmark_1_ac_power_step_exact_600w() -> None:
    """Scenario 1: Attic AC modulates from 1200 W to 200 W at 12:02:00 (t=120s)."""
    acc = StepSignalAccumulator(initial_value=1200.0, t_mono=0.0, is_valid=True)
    acc.update(200.0, t_mono=120.0, is_valid=True)
    mean_w, coverage = acc.close_window(t_mono=300.0, window_dt=300.0)

    assert mean_w is not None
    assert math.isclose(mean_w, 600.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(coverage, 1.0, rel_tol=0.0, abs_tol=1e-12)

    mean_w_next, cov_next = acc.close_window(t_mono=600.0, window_dt=300.0)
    assert mean_w_next is not None
    assert math.isclose(mean_w_next, 200.0, abs_tol=1e-12)
    assert math.isclose(cov_next, 1.0, abs_tol=1e-12)


def test_step_accumulator_unavailable_gap_pauses_valid_dt() -> None:
    acc = StepSignalAccumulator(initial_value=1000.0, t_mono=0.0, is_valid=True)
    acc.update(None, t_mono=90.0, is_valid=False)
    acc.update(500.0, t_mono=240.0, is_valid=True)
    mean_val, coverage = acc.close_window(t_mono=300.0, window_dt=300.0)

    assert mean_val is not None
    assert math.isclose(mean_val, 800.0, abs_tol=1e-12)
    assert math.isclose(coverage, 0.50, abs_tol=1e-12)


def test_benchmark_2_heishamon_late_spike_immunized() -> None:
    """Scenario 2: HeishaMon outlet jumps +0.25 C at 12:04:55 (t=295s) @ 15.2 L/min."""
    hyd = HydraulicPowerAccumulator(t_mono=0.0)
    hyd.update_threeway_valve("Room", t_mono=0.0)
    hyd.update_flow(15.2, t_mono=0.0)
    hyd.update_inlet_temp(27.00, t_mono=0.0)
    hyd.update_outlet_temp(30.00, t_mono=0.0)
    hyd.update_buffer_top(29.40)
    hyd.update_buffer_bot(27.80)

    hyd.update_outlet_temp(30.25, t_mono=295.0)
    hyd.update_buffer_top(29.85)
    hyd.update_buffer_bot(28.10)

    res = hyd.close_window(t_mono=300.0, window_dt=300.0)

    naive_snapshot_w = 69.73 * 15.2 * (30.25 - 27.00)
    expected_true_w = (
        295.0 * (69.73 * 15.2 * 3.00) + 5.0 * naive_snapshot_w
    ) / 300.0

    assert math.isclose(res.p_prim_th_w, expected_true_w, abs_tol=1e-9)
    assert math.isclose(res.p_prim_th_w, 3184.1, abs_tol=0.05)
    assert math.isclose(naive_snapshot_w - res.p_prim_th_w, 260.6, abs_tol=0.1)
    assert math.isclose(res.coverage, 1.0, abs_tol=1e-12)

    assert math.isclose(res.t_buf_top_start_c, 29.40, abs_tol=1e-9)
    assert math.isclose(res.t_buf_top_end_c, 29.85, abs_tol=1e-9)
    assert math.isclose(res.t_buf_bot_start_c, 27.80, abs_tol=1e-9)
    assert math.isclose(res.t_buf_bot_end_c, 28.10, abs_tol=1e-9)

    hyd.update_buffer_top(30.10)
    res2 = hyd.close_window(t_mono=600.0, window_dt=300.0)
    assert math.isclose(res2.t_buf_top_start_c, 29.85, abs_tol=1e-9)
    assert math.isclose(res2.t_buf_top_end_c, 30.10, abs_tol=1e-9)


def test_hydraulic_accumulator_defrost_negative_and_dhw_tank_gate() -> None:
    hyd = HydraulicPowerAccumulator(t_mono=0.0)
    hyd.update_threeway_valve("Room", t_mono=0.0)
    hyd.update_flow(20.0, t_mono=0.0)
    hyd.update_inlet_temp(28.0, t_mono=0.0)
    hyd.update_outlet_temp(24.0, t_mono=0.0)

    hyd.update_threeway_valve("Tank", t_mono=150.0)
    hyd.update_outlet_temp(48.0, t_mono=150.0)
    hyd.update_inlet_temp(43.0, t_mono=150.0)

    res = hyd.close_window(t_mono=300.0, window_dt=300.0)
    expected_room_w = (150.0 * (69.73 * 20.0 * -4.0) + 150.0 * 0.0) / 300.0
    assert res.p_prim_th_w < 0.0
    assert math.isclose(res.p_prim_th_w, expected_room_w, abs_tol=1e-9)
    assert math.isclose(res.dhw_duty, 0.5, abs_tol=1e-9)


def test_benchmark_3_wax_valve_close_and_mid_stroke_reversals() -> None:
    """Scenario 3: Wax valve (228s) turns off at 12:01:00 -> exact u_eff = 0.580."""
    valve = WaxValveTracker(
        initial_on=True,
        t_mono=0.0,
        open_time_s=228.0,
        close_time_s=228.0,
        has_state=True,
    )
    valve.set_switch(False, t_mono=60.0)
    u_eff = valve.close_window(t_mono=300.0, window_dt=300.0)

    assert math.isclose(u_eff, 0.580, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(valve.position, 0.0, abs_tol=1e-12)

    valve2 = WaxValveTracker(
        initial_on=False,
        t_mono=0.0,
        open_time_s=228.0,
        close_time_s=228.0,
        has_state=True,
    )
    valve2.set_switch(True, t_mono=0.0)
    valve2.set_switch(False, t_mono=114.0)
    assert math.isclose(valve2.position, 0.5, abs_tol=1e-12)
    valve2.set_switch(True, t_mono=171.0)
    assert math.isclose(valve2.position, 0.25, abs_tol=1e-12)
    u_eff_rev = valve2.close_window(t_mono=300.0, window_dt=300.0)

    dt_ref = 0.001
    pos_ref = 0.0
    area_ref = 0.0
    steps = int(300.0 / dt_ref)
    for i in range(steps):
        t = i * dt_ref
        target_on = (t < 114.0) or (t >= 171.0)
        pos_next = (
            min(1.0, pos_ref + dt_ref / 228.0)
            if target_on
            else max(0.0, pos_ref - dt_ref / 228.0)
        )
        area_ref += 0.5 * (pos_ref + pos_next) * dt_ref
        pos_ref = pos_next

    assert math.isclose(u_eff_rev, area_ref / 300.0, abs_tol=1e-6)


@pytest.mark.parametrize(
    ("raw_val", "unit", "default_unit", "expected_w"),
    [
        pytest.param(1.5, None, "kW", 1500.0, id="p1_default_kw"),
        pytest.param(1.5, "kW", "W", 1500.0, id="explicit_kw"),
        pytest.param(450.0, "W", "kW", 450.0, id="explicit_w_override"),
    ],
)
def test_power_unit_conversions(
    raw_val: float, unit: str | None, default_unit: str, expected_w: float
) -> None:
    assert math.isclose(
        convert_power_to_w(raw_val, unit, default_unit),
        expected_w,
        abs_tol=1e-9,
    )


@pytest.mark.parametrize(
    ("raw_speed", "unit", "expected_ms"),
    [
        pytest.param(18.0, "km/h", 5.0, id="kmh"),
        pytest.param(10.0, "mph", 4.4704, id="mph"),
        pytest.param(10.0, "kn", 5.14444, id="knots"),
        pytest.param(4.2, "m/s", 4.2, id="ms"),
    ],
)
def test_wind_speed_unit_conversions(
    raw_speed: float, unit: str, expected_ms: float
) -> None:
    assert math.isclose(
        convert_wind_speed_to_ms(raw_speed, unit), expected_ms, abs_tol=1e-9
    )


def test_window_accumulator_core_schema_p1_valves_and_zha_staleness() -> None:
    core = WindowAccumulatorCore(t_mono=0.0)
    t0_utc = 1_763_670_000.0

    core.ingest_state(
        HOUSE_ENTITIES.p1_power_consumed_kw,
        "1.85",
        {"unit_of_measurement": "kW"},
        t_mono=0.0,
        is_initial_seed=True,
    )
    core.ingest_state(
        HOUSE_ENTITIES.p1_power_produced_kw,
        "0.35",
        {"unit_of_measurement": "kW"},
        t_mono=0.0,
        is_initial_seed=True,
    )
    core.ingest_state(
        HOUSE_ENTITIES.weather_entity,
        "cloudy",
        {
            "temperature": 7.2,
            "humidity": None,
            "wind_speed": 18.0,
            "wind_speed_unit": "km/h",
        },
        t_mono=0.0,
        is_initial_seed=True,
    )
    core.ingest_state(
        HOUSE_ENTITIES.outdoor_humidity_dak,
        "84.5",
        {"unit_of_measurement": "%"},
        t_mono=0.0,
        is_initial_seed=True,
    )
    core.ingest_state(
        HOUSE_ENTITIES.valve_slaapkamer,
        "unavailable",
        {},
        t_mono=0.0,
        is_initial_seed=True,
    )
    core.ingest_state(
        HOUSE_ENTITIES.valve_slaapkamer_voor,
        "on",
        {},
        t_mono=0.0,
        is_initial_seed=True,
    )
    core.ingest_state(
        HOUSE_ENTITIES.valve_slaapkamer_achter,
        "off",
        {},
        t_mono=0.0,
        is_initial_seed=True,
    )

    wk_cfg = ROOM_SENSORS["wk"]
    core.ingest_state(
        wk_cfg.temp_entity_id,
        "20.15",
        {},
        t_mono=0.0,
        last_changed_utc=t0_utc - 10_000.0,
        last_reported_utc=t0_utc - 10_000.0,
        is_initial_seed=True,
    )
    assert wk_cfg.humidity_entity_id is not None
    core.ingest_state(
        wk_cfg.humidity_entity_id,
        "49.2",
        {},
        t_mono=100.0,
        last_changed_utc=t0_utc + 100.0,
        last_reported_utc=t0_utc + 100.0,
        is_initial_seed=False,
    )

    kt_cfg = ROOM_SENSORS["kantoor"]
    core.ingest_state(
        kt_cfg.temp_entity_id,
        "18.40",
        {},
        t_mono=0.0,
        last_changed_utc=t0_utc - 9000.0,
        last_reported_utc=t0_utc - 9000.0,
        is_initial_seed=True,
    )
    assert kt_cfg.humidity_entity_id is not None
    core.ingest_state(
        kt_cfg.humidity_entity_id,
        "52.0",
        {},
        t_mono=0.0,
        last_changed_utc=t0_utc - 8000.0,
        last_reported_utc=t0_utc - 8000.0,
        is_initial_seed=True,
    )

    res = core.close_window(
        ts_utc=int(t0_utc + 300), t_mono=300.0, window_dt=300.0
    )
    row = res.raw_row

    assert set(row.keys()) == set(RINGBUFFER_ALL_COLS)
    assert math.isclose(float(row["p_p1_net_w"]), 1500.0, abs_tol=1e-6)
    assert math.isclose(float(row["v_wind_ms"]), 5.0, abs_tol=1e-6)
    assert math.isclose(float(row["rh_out_pct"]), 84.5, abs_tol=1e-6)
    assert math.isclose(float(row["u_eff_slaapkamer"]), 0.5, abs_tol=1e-6)

    assert res.stale_wk is False
    assert res.temp_changed_wk is False
    assert res.hum_changed_wk is True
    assert bool(int(row["flags"]) & FLAG_VALID_WK) is True

    assert res.stale_kt is True
    assert bool(int(row["flags"]) & FLAG_VALID_KT) is False


def test_clock_jump_guard_ntp_immunity() -> None:
    saved_utc = 1_763_670_000
    guard = ClockJumpGuard(saved_wall_utc=saved_utc, t_mono_start=1000.0)

    ok1, _, dt1 = guard.check_and_advance(saved_utc - 600, t_mono=1300.0)
    assert ok1 is False
    assert math.isclose(dt1, 300.0, abs_tol=1e-9)
    assert guard.rejected_ticks == 1

    ok2, _, dt2 = guard.check_and_advance(saved_utc, t_mono=1600.0)
    assert ok2 is False
    assert math.isclose(dt2, 300.0, abs_tol=1e-9)

    ntp_synced_utc = saved_utc + 86_400
    ok3, ts3, dt3 = guard.check_and_advance(ntp_synced_utc, t_mono=1900.0)
    assert ok3 is True
    assert ts3 == ntp_synced_utc
    assert math.isclose(dt3, 300.0, abs_tol=1e-9)

    ok4, _, _ = guard.check_and_advance(ntp_synced_utc, t_mono=2200.0)
    assert ok4 is False


def _sample_row(ts_utc: int, idx: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {col: 0.0 for col in RINGBUFFER_ALL_COLS}
    row["ts_utc"] = ts_utc
    row["flags"] = FLAG_VALID_WK | FLAG_VALID_KT
    row["t_air_wk_c"] = 19.5 + 0.001 * (idx % 1000)
    row["t_air_kantoor_c"] = 18.0 + 0.001 * (idx % 500)
    row["p_prim_th_w"] = float(idx % 4000)
    row["t_bg_sup_c"] = float("nan")
    return row


def test_ringbuffer_14_week_rotation_and_90d_chronological_slice() -> None:
    rb = ColumnarRingBuffer()
    assert rb.capacity == RINGBUFFER_CHUNK_ROWS * RINGBUFFER_NUM_CHUNKS == 28_224
    assert rb.row_count == 0
    assert rb.latest_ts_utc == 0

    base_ts = 1_760_000_100
    total_rows = 15 * RINGBUFFER_CHUNK_ROWS
    for i in range(total_rows):
        ts = base_ts + i * DT_STEP_S
        slot = rb.append_row(_sample_row(ts, i))
        assert 0 <= slot < RINGBUFFER_NUM_CHUNKS

    rb.update_last_row_outputs({"t_floor_wk_c": 21.25, "q_fh_wk_w": 1850.0})

    assert rb.row_count == rb.capacity
    expected_latest_ts = base_ts + (total_rows - 1) * DT_STEP_S
    assert rb.latest_ts_utc == expected_latest_ts

    cols_90d = rb.get_columns(max_rows=RINGBUFFER_ROWS)
    ts_col = cols_90d["ts_utc"]
    assert len(ts_col) == RINGBUFFER_ROWS == 25_920
    assert ts_col[-1] == expected_latest_ts
    assert ts_col[0] == expected_latest_ts - (RINGBUFFER_ROWS - 1) * DT_STEP_S

    diffs = {ts_col[k + 1] - ts_col[k] for k in range(len(ts_col) - 1)}
    assert diffs == {DT_STEP_S}

    assert math.isclose(float(cols_90d["t_floor_wk_c"][-1]), 21.25, rel_tol=1e-6)
    assert math.isclose(float(cols_90d["q_fh_wk_w"][-1]), 1850.0, rel_tol=1e-6)
    assert math.isnan(float(cols_90d["t_bg_sup_c"][-1]))


def test_ringbuffer_zlib_crc32_roundtrip_and_1byte_corruption_recovery() -> None:
    rb_src = ColumnarRingBuffer(chunk_rows=64, num_chunks=14)
    base_ts = 1_762_000_200
    total_rows = 14 * 64 + 20
    for i in range(total_rows):
        rb_src.append_row(_sample_row(base_ts + i * DT_STEP_S, i))

    encoded_chunks = [
        rb_src.encode_chunk(slot) for slot in range(rb_src.num_chunks)
    ]

    rb_restored = ColumnarRingBuffer(chunk_rows=64, num_chunks=14)
    for slot, payload in enumerate(encoded_chunks):
        assert rb_restored.decode_chunk(slot, payload) is True

    assert rb_restored.corrupt_chunks_recovered == 0
    assert rb_restored.row_count == rb_src.row_count
    assert rb_restored.latest_ts_utc == rb_src.latest_ts_utc
    assert (
        rb_restored.get_columns(500)["ts_utc"]
        == rb_src.get_columns(500)["ts_utc"]
    )

    rb_corrupt = ColumnarRingBuffer(chunk_rows=64, num_chunks=14)
    for slot, payload in enumerate(encoded_chunks):
        if slot == 5:
            raw_comp = bytearray(base64.b64decode(payload["data_b64"]))
            raw_comp[len(raw_comp) // 2] ^= 0xFF
            bad_payload = dict(payload)
            bad_payload["data_b64"] = base64.b64encode(bytes(raw_comp)).decode(
                "ascii"
            )
            assert rb_corrupt.decode_chunk(slot, bad_payload) is False
        else:
            assert rb_corrupt.decode_chunk(slot, payload) is True

    assert rb_corrupt.corrupt_chunks_recovered == 1
    assert rb_corrupt.row_count == (13 * 64)
    cols_after = rb_corrupt.get_columns(1000)["ts_utc"]
    assert len(cols_after) == 13 * 64
    assert all(
        cols_after[k] < cols_after[k + 1] for k in range(len(cols_after) - 1)
    )


class _StubEngine:
    def __init__(self) -> None:
        self.calls: int = 0
        self.eval_refreshed: bool = False

    def process_5m_window(
        self, raw_row: dict[str, Any], **kwargs: Any
    ) -> dict[str, float]:
        del kwargs  # Unused.
        self.calls += 1
        t_wk = float(raw_row["t_air_wk_c"])
        t_kt = float(raw_row["t_air_kantoor_c"])
        return {
            "t_floor_wk_c": t_wk if math.isfinite(t_wk) else float("nan"),
            "d_est_wk_w": 0.0,
            "q_fh_wk_w": 0.0,
            "t_hind6h_wk_c": float("nan"),
            "t_floor_kt_c": t_kt if math.isfinite(t_kt) else float("nan"),
            "d_est_kt_w": 0.0,
            "q_fh_kt_w": 0.0,
            "t_hind6h_kt_c": float("nan"),
        }

    def refresh_14d_evaluation(self, cols: dict[str, Any]) -> dict[str, Any]:
        del cols  # Unused.
        self.eval_refreshed = True
        return {}


def test_backfill_swap_timestamp_utc_and_300s_alignment() -> None:
    swap_ts = 1_763_000_600
    custom_rooms = dict(ROOM_SENSORS)
    custom_rooms["wk"] = RoomSensorConfig(
        temp_entity_id="sensor.new_wk_temp",
        humidity_entity_id="sensor.new_wk_hum",
        pre_swap_temp_entity_id="sensor.old_wk_temp",
        swap_timestamp_utc=swap_ts,
    )

    engine = _StubEngine()
    rb = ColumnarRingBuffer()

    raw_start = 1_763_000_000
    end_ts = 1_763_001_200
    events = [
        HistoricalStateEvent(
            ts_utc=raw_start,
            entity_id=HOUSE_ENTITIES.outdoor_temp_dak,
            state_str="6.5",
            attributes={},
            last_changed_utc=raw_start,
            last_reported_utc=raw_start,
        ),
        HistoricalStateEvent(
            ts_utc=raw_start,
            entity_id="sensor.old_wk_temp",
            state_str="19.2",
            attributes={},
            last_changed_utc=raw_start,
            last_reported_utc=raw_start,
        ),
        HistoricalStateEvent(
            ts_utc=raw_start + 100,
            entity_id="sensor.new_wk_temp",
            state_str="24.9",
            attributes={},
            last_changed_utc=raw_start + 100,
            last_reported_utc=raw_start + 100,
        ),
        HistoricalStateEvent(
            ts_utc=swap_ts + 50,
            entity_id="sensor.new_wk_temp",
            state_str="20.4",
            attributes={},
            last_changed_utc=swap_ts + 50,
            last_reported_utc=swap_ts + 50,
        ),
    ]

    added = replay_backfill_into_engine(
        engine=engine,  # type: ignore[arg-type]
        ringbuffer=rb,
        lts_series={},
        lts_units={},
        raw_events=events,
        lts_start_ts=raw_start - 3600,
        raw_start_ts=raw_start,
        end_ts=end_ts,
        room_sensors=custom_rooms,
    )

    assert added == 4
    assert engine.eval_refreshed is True
    cols = rb.get_columns()
    assert cols["ts_utc"] == [
        raw_start + 300,
        raw_start + 600,
        raw_start + 900,
        raw_start + 1200,
    ]
    assert math.isclose(float(cols["t_air_wk_c"][0]), 19.2, abs_tol=1e-5)
    assert math.isclose(float(cols["t_air_wk_c"][1]), 19.2, abs_tol=1e-5)
    assert math.isclose(float(cols["t_air_wk_c"][2]), 20.4, abs_tol=1e-5)
    assert math.isclose(float(cols["t_air_wk_c"][3]), 20.4, abs_tol=1e-5)


def test_smart_throttled_sensor_deadband_and_heartbeats() -> None:
    temp_sensor = SmartThrottledSensor(
        coordinator=None,  # type: ignore[arg-type]
        entry_id="test_entry",
        object_id="woonkamer_geschatte_vloertemperatuur",
        translation_key="woonkamer_geschatte_vloertemperatuur",
        deadband=0.05,
        max_interval_s=3600.0,
        precision=1,
    )
    assert temp_sensor.entity_id == "sensor.woonkamer_geschatte_vloertemperatuur"
    assert temp_sensor.evaluate_update(20.00, t_mono=0.0) is True
    assert temp_sensor.native_value == pytest.approx(20.0)
    assert temp_sensor.available is True

    assert temp_sensor.evaluate_update(20.03, t_mono=300.0) is False
    assert temp_sensor.evaluate_update(19.97, t_mono=1800.0) is False
    assert temp_sensor.evaluate_update(20.02, t_mono=3600.0) is True
    assert temp_sensor.evaluate_update(20.08, t_mono=3900.0) is True
    assert temp_sensor.native_value == pytest.approx(20.1)

    assert temp_sensor.evaluate_update(None, t_mono=4000.0) is True
    assert temp_sensor.available is False
    assert temp_sensor.evaluate_update(None, t_mono=4300.0) is False
    assert temp_sensor.evaluate_update(20.08, t_mono=4600.0) is True
    assert temp_sensor.available is True

    dist_sensor = SmartThrottledSensor(
        coordinator=None,  # type: ignore[arg-type]
        entry_id="test_entry",
        object_id="woonkamer_ongeplande_warmtelast",
        translation_key="woonkamer_ongeplande_warmtelast",
        deadband=25.0,
        max_interval_s=3600.0,
        precision=0,
        zero_clamp_threshold=15.0,
        zero_transition_immediate=True,
    )
    assert dist_sensor.evaluate_update(12.0, t_mono=0.0) is True
    assert dist_sensor.native_value == 0
    assert dist_sensor.evaluate_update(-14.5, t_mono=300.0) is False
    assert dist_sensor.evaluate_update(18.0, t_mono=600.0) is True
    assert dist_sensor.native_value == 18

    q_sensor = SmartThrottledSensor(
        coordinator=None,  # type: ignore[arg-type]
        entry_id="test_entry",
        object_id="woonkamer_vloer_afgiftevermogen",
        translation_key="woonkamer_vloer_afgiftevermogen",
        deadband=50.0,
        max_interval_s=900.0,
        idle_zero_max_interval_s=21600.0,
        precision=0,
        zero_transition_immediate=True,
    )
    assert q_sensor.evaluate_update(0.0, t_mono=0.0) is True
    assert q_sensor.evaluate_update(0.0, t_mono=900.0) is False
    assert q_sensor.evaluate_update(0.0, t_mono=10_800.0) is False
    assert q_sensor.evaluate_update(0.0, t_mono=21_600.0) is True
    assert q_sensor.evaluate_update(35.0, t_mono=21_900.0) is True
    assert q_sensor.native_value == 35
    assert q_sensor.evaluate_update(40.0, t_mono=22_200.0) is False
    assert q_sensor.evaluate_update(40.0, t_mono=22_800.0) is True
    assert q_sensor.evaluate_update(0.0, t_mono=23_100.0) is True
    assert q_sensor.native_value == 0
    assert len(DEFAULT_FLOW_RATES_L_MIN) == 10


def test_cross_module_pir_fallback_sink_mapping_and_fit_sensor() -> None:
    """Verify accumulator -> engine PIR fallback, sink mapping, and fit sensor."""
    core_occ = WindowAccumulatorCore(t_mono=0.0)
    core_empty = WindowAccumulatorCore(t_mono=0.0)
    t0 = 1_763_670_000.0

    for acc in (core_occ, core_empty):
        acc.ingest_state(
            HOUSE_ENTITIES.outdoor_temp_dak,
            "5.0",
            {},
            t_mono=0.0,
            is_initial_seed=True,
        )
        acc.ingest_state(
            ROOM_SENSORS["kantoor"].temp_entity_id,
            "19.0",
            {},
            t_mono=0.0,
            last_changed_utc=t0,
            last_reported_utc=t0,
            is_initial_seed=True,
        )
    core_occ.ingest_state(
        HOUSE_ENTITIES.ac_kantoor_occupancy,
        "on",
        {},
        t_mono=0.0,
        is_initial_seed=True,
    )

    row_occ = core_occ.close_window(
        ts_utc=int(t0 + 300), t_mono=300.0, window_dt=300.0
    ).raw_row
    row_empty = core_empty.close_window(
        ts_utc=int(t0 + 300), t_mono=300.0, window_dt=300.0
    ).raw_row
    assert math.isnan(float(row_occ["p_desk_kt_w"]))

    eng_occ = DigitalTwinEngine()
    eng_empty = DigitalTwinEngine()
    eng_occ.process_5m_window(row_occ, stale_kt=True)
    eng_empty.process_5m_window(row_empty, stale_kt=True)
    # Active PIR radar (+140 W workplace gain) must warm `kt` air vs unoccupied
    assert (
        eng_occ.get_zone_snapshot("kt").state.t_air_c
        > eng_empty.get_zone_snapshot("kt").state.t_air_c
    )

    # Mapping `{zone: u_eff}` in `estimate_total_sink_capacity_w` respects weights
    cap_bg_only = eng_occ.estimate_total_sink_capacity_w(
        32.0, {"bg": 1.0, "kt": 0.0}
    )
    cap_bg_list = eng_occ.estimate_total_sink_capacity_w(32.0, ["bg"])
    cap_half_bg = eng_occ.estimate_total_sink_capacity_w(32.0, {"bg": 0.5})
    assert math.isclose(cap_bg_only, cap_bg_list, rel_tol=1e-9)
    assert math.isclose(cap_half_bg, 0.5 * cap_bg_list, rel_tol=1e-9)

    # ZoneFitSensor reads `get_zone_fit_diagnostics` & combined `calibration_history`
    eng_occ._last_posterior_std_wk = {"h_loss_w_k": 3.25}
    eng_occ._last_corr_wk = {"h_loss_w_k": {"h_loss_w_k": 1.0}}
    eng_occ.calibration_history.append(
        {
            "ts_utc": int(t0 + 300),
            "event": "calibration_manual",
            "mode": "manual",
            "accepted_wk": True,
            "reason_wk": "Accepted (manual)",
            "holdout_j_wk_before": 0.22,
            "holdout_j_wk_after": 0.14,
            "accepted_kt": False,
            "reason_kt": "Skipped",
        }
    )

    class _StubCoord:
        fit_sensor_generation = 1
        engine = eng_occ
        ringbuffer = ColumnarRingBuffer()

    fit_sensor = BetonBatterijZoneFitSensor(
        coordinator=_StubCoord(),  # type: ignore[arg-type]
        entry_id="test_entry",
        zone="wk",
        object_id="betonbatterij_fit_woonkamer",
        translation_key="fit_woonkamer",
    )
    fit_sensor._refresh_from_engine(force=True)
    attrs = fit_sensor._attr_extra_state_attributes
    assert attrs is not None
    json.dumps(attrs, allow_nan=False)
    assert attrs["fisher_1sigma_std"] == {"h_loss_w_k": 3.25}
    assert attrs["last_calibration"]["accepted"] is True
    assert attrs["last_calibration"]["mode"] == "manual"


def test_load_house_config_defaults_example_and_override(
    tmp_path: Path,
) -> None:
    """Verify `load_house_config()` on None, missing, example, override, & local."""
    repo_root = Path(__file__).resolve().parents[1]

    # 1. None or missing file returns generic defaults cleanly.
    ent_none, rooms_none = load_house_config(None)
    assert ent_none == HOUSE_ENTITIES
    assert rooms_none == ROOM_SENSORS

    ent_missing, rooms_missing = load_house_config(
        tmp_path / "does_not_exist.json"
    )
    assert ent_missing == HOUSE_ENTITIES
    assert rooms_missing == ROOM_SENSORS

    # 2. Tracked `betonbatterij_local.example.json` parses & matches defaults.
    example_path = repo_root / "betonbatterij_local.example.json"
    assert example_path.is_file()
    ent_ex, rooms_ex = load_house_config(example_path)
    assert ent_ex == HOUSE_ENTITIES
    assert rooms_ex == ROOM_SENSORS

    # 3. Temporary override JSON merges partial entities + room sensors cleanly.
    override_path = tmp_path / LOCAL_CONFIG_FILENAME
    override_payload = {
        "_comment": "Partial local override test",
        "entities": {
            "buf_top_temp": "sensor.custom_buffer_top",
            "valve_werkkamer_1e": "switch.custom_valve_werkkamer",
            "desk_kantoor_power_w": "sensor.custom_desk_power",
            "bg_manifold_return_offset_c": -0.15,
        },
        "room_sensors": {
            "werkkamer_1e": {
                "temp_entity_id": "sensor.custom_werkkamer_temp",
                "humidity_entity_id": None,
                "pre_swap_temp_entity_id": "sensor.old_werkkamer_temp",
                "swap_timestamp_utc": 1_760_000_000,
            },
            "kamer_1e_zuid": "sensor.shorthand_zuid_temp",
        },
    }
    override_path.write_text(json.dumps(override_payload), encoding="utf-8")

    ent_ov, rooms_ov = load_house_config(override_path)
    assert ent_ov.buf_top_temp == "sensor.custom_buffer_top"
    assert ent_ov.valve_werkkamer_1e == "switch.custom_valve_werkkamer"
    assert ent_ov.desk_kantoor_power_w == "sensor.custom_desk_power"
    assert math.isclose(ent_ov.bg_manifold_return_offset_c, -0.15, abs_tol=1e-9)
    assert ent_ov.wp_pump_flow == HOUSE_ENTITIES.wp_pump_flow

    assert rooms_ov["werkkamer_1e"] == RoomSensorConfig(
        temp_entity_id="sensor.custom_werkkamer_temp",
        humidity_entity_id=None,
        pre_swap_temp_entity_id="sensor.old_werkkamer_temp",
        swap_timestamp_utc=1_760_000_000,
    )
    assert (
        rooms_ov["kamer_1e_zuid"].temp_entity_id
        == "sensor.shorthand_zuid_temp"
    )
    assert (
        rooms_ov["kamer_1e_zuid"].humidity_entity_id
        == ROOM_SENSORS["kamer_1e_zuid"].humidity_entity_id
    )
    assert rooms_ov["wk"] == ROOM_SENSORS["wk"]

    core_ov = WindowAccumulatorCore(
        t_mono=0.0, entities=ent_ov, room_sensors=rooms_ov
    )
    watched_ov = core_ov.watched_entity_ids()
    assert "sensor.custom_buffer_top" in watched_ov
    assert "switch.custom_valve_werkkamer" in watched_ov
    assert "sensor.custom_desk_power" in watched_ov
    assert "sensor.custom_werkkamer_temp" in watched_ov
    assert "sensor.old_werkkamer_temp" in watched_ov

    # 4. Gitignored `betonbatterij_local.json` on disk parses cleanly if present.
    local_path = repo_root / LOCAL_CONFIG_FILENAME
    if local_path.is_file():
        ent_local, rooms_local = load_house_config(local_path)
        assert set(rooms_local.keys()) == set(ROOM_SENSORS.keys())
        core_local = WindowAccumulatorCore(
            t_mono=0.0, entities=ent_local, room_sensors=rooms_local
        )
        assert len(core_local.watched_entity_ids()) >= 40

    # 5. Runtime coordinates propagate into DigitalTwinEngine with 0.0 fallback.
    eng_default = DigitalTwinEngine(latitude_deg=0.0, longitude_deg=0.0)
    assert eng_default.latitude_deg == LATITUDE_DEG
    assert eng_default.longitude_deg == LONGITUDE_DEG

    eng_custom = DigitalTwinEngine(latitude_deg=52.37, longitude_deg=4.89)
    assert math.isclose(eng_custom.latitude_deg, 52.37, abs_tol=1e-9)
    assert math.isclose(eng_custom.longitude_deg, 4.89, abs_tol=1e-9)

    # 6. `strings.json` and `translations/en.json` stay byte-identical.
    cc_dir = repo_root / "custom_components" / "betonbatterij"
    assert (cc_dir / "strings.json").read_bytes() == (
        cc_dir / "translations" / "en.json"
    ).read_bytes()


def test_cop_fit_sensor_dhw_tank_and_post_dhw_idle_supply_temp() -> None:
    """Verify CV-COP supply temperature selection during & right after DHW runs."""
    eng = DigitalTwinEngine()

    class _CopCoordStub:
        def __init__(self, raw_row: dict[str, float | int]) -> None:
            self.fit_sensor_generation = 1
            self.engine = eng
            self.last_raw_row = raw_row

    def _eval_cop(raw_row: dict[str, float | int]) -> tuple[float, float]:
        coord = _CopCoordStub(raw_row)
        sensor = BetonBatterijCopFitSensor(
            coordinator=coord,  # type: ignore[arg-type]
            entry_id="test_entry",
        )
        sensor._refresh_from_engine(force=True)
        live = sensor._compute_live_expected_cop({}, {})
        assert sensor.native_value == round(live, 2)
        return live, float(sensor.native_value)

    # 1. Normal CV-run: uses t_wp_out_c (29.0 °C) over t_buf_top_end_c (28.0 °C).
    cop_cv, _ = _eval_cop(
        {
            "flags": FLAG_VALID_WK,
            "t_out_dak_c": 7.0,
            "rh_out_pct": 80.0,
            "f_comp_hz": 24.0,
            "t_wp_out_c": 29.0,
            "t_buf_top_end_c": 28.0,
        }
    )
    assert math.isclose(
        cop_cv, eng.predict_cop_cv(7.0, 29.0, 80.0), rel_tol=1e-9
    )

    # 2. Active DHW/Tank-run: ignores 58.5 °C boiler water, uses 28.5 °C buffer.
    cop_dhw_buf, _ = _eval_cop(
        {
            "flags": FLAG_VALID_WK | FLAG_DHW_TANK,
            "t_out_dak_c": 7.0,
            "rh_out_pct": 80.0,
            "f_comp_hz": 48.0,
            "t_wp_out_c": 58.5,
            "t_buf_top_end_c": 28.5,
        }
    )
    cop_at_boiler_58_5 = eng.predict_cop_cv(7.0, 58.5, 80.0)
    assert math.isclose(
        cop_dhw_buf, eng.predict_cop_cv(7.0, 28.5, 80.0), rel_tol=1e-9
    )
    assert cop_dhw_buf > cop_at_boiler_58_5 + 1.5

    # 3. Active DHW/Tank-run with NaN buffer sensors: falls back to 30.0 °C.
    cop_dhw_nan, _ = _eval_cop(
        {
            "flags": FLAG_VALID_WK | FLAG_DHW_TANK,
            "t_out_dak_c": 7.0,
            "rh_out_pct": 80.0,
            "f_comp_hz": 48.0,
            "t_wp_out_c": 58.5,
            "t_buf_top_end_c": float("nan"),
            "t_buf_top_start_c": float("nan"),
        }
    )
    assert math.isclose(
        cop_dhw_nan, eng.predict_cop_cv(7.0, 30.0, 80.0), rel_tol=1e-9
    )
    assert not math.isclose(cop_dhw_nan, cop_at_boiler_58_5, abs_tol=0.5)

    # 4. Post-DHW stationary WP (f_comp_hz <= 0.5, hot t_wp_out_c > 40.0):
    #    prefers buffer top (28.0 °C) over stagnant 52.0 °C outlet sensor.
    cop_post_dhw, _ = _eval_cop(
        {
            "flags": FLAG_VALID_WK,
            "t_out_dak_c": 7.0,
            "rh_out_pct": 80.0,
            "f_comp_hz": 0.0,
            "t_wp_out_c": 52.0,
            "t_buf_top_end_c": 28.0,
        }
    )
    assert math.isclose(
        cop_post_dhw, eng.predict_cop_cv(7.0, 28.0, 80.0), rel_tol=1e-9
    )

    # 5. Post-DHW stationary WP with NaN buffer sensors: falls back to 30.0 °C
    #    and never leaks hot 52.0 °C outlet temperature into CV COP.
    cop_post_dhw_nan, _ = _eval_cop(
        {
            "flags": FLAG_VALID_WK,
            "t_out_dak_c": 7.0,
            "rh_out_pct": 80.0,
            "f_comp_hz": 0.0,
            "t_wp_out_c": 52.0,
            "t_buf_top_end_c": float("nan"),
            "t_buf_top_start_c": float("nan"),
        }
    )
    assert math.isclose(
        cop_post_dhw_nan, eng.predict_cop_cv(7.0, 30.0, 80.0), rel_tol=1e-9
    )
    assert cop_post_dhw_nan > eng.predict_cop_cv(7.0, 52.0, 80.0) + 1.0

    # 6. Running compressor (22 Hz) right after DHW with residual t_wp_out_c =
    #    45.0 °C (> 40.0 °C): still uses buffer top (or 30.0 °C fallback).
    cop_hot_running_buf, _ = _eval_cop(
        {
            "flags": FLAG_VALID_WK,
            "t_out_dak_c": 7.0,
            "rh_out_pct": 80.0,
            "f_comp_hz": 22.0,
            "t_wp_out_c": 45.0,
            "t_buf_top_end_c": 29.0,
        }
    )
    assert math.isclose(
        cop_hot_running_buf, eng.predict_cop_cv(7.0, 29.0, 80.0), rel_tol=1e-9
    )

    cop_hot_running_nan, _ = _eval_cop(
        {
            "flags": FLAG_VALID_WK,
            "t_out_dak_c": 7.0,
            "rh_out_pct": 80.0,
            "f_comp_hz": 22.0,
            "t_wp_out_c": 45.0,
            "t_buf_top_end_c": float("nan"),
            "t_buf_top_start_c": float("nan"),
        }
    )
    assert math.isclose(
        cop_hot_running_nan, eng.predict_cop_cv(7.0, 30.0, 80.0), rel_tol=1e-9
    )


def test_cop_fit_sensor_live_5m_deadband_heartbeat_and_gen_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify live 5m COP recomputation, 0.05 deadband, 60m heartbeat, & gen."""
    eng = DigitalTwinEngine()
    eng.cop_diagnostics = {"n_cv_samples": 12}
    mono_now = 1000.0
    monkeypatch.setattr(sensor_mod.time, "monotonic", lambda: mono_now)

    class _LiveCopCoordStub:
        def __init__(self) -> None:
            self.fit_sensor_generation = 1
            self.engine = eng
            self.last_raw_row: dict[str, float | int] = {
                "flags": FLAG_VALID_WK,
                "t_out_dak_c": 7.0,
                "rh_out_pct": 80.0,
                "f_comp_hz": 24.0,
                "t_wp_out_c": 29.0,
            }

    coord = _LiveCopCoordStub()
    writes: list[float] = []

    class _TrackedCopSensor(BetonBatterijCopFitSensor):
        def async_write_ha_state(self) -> None:
            assert self._attr_native_value is not None
            writes.append(float(self._attr_native_value))

    sensor = _TrackedCopSensor(
        coordinator=coord,  # type: ignore[arg-type]
        entry_id="test_entry",
    )

    # Initial forced publish
    sensor._refresh_from_engine(force=True)
    assert len(writes) == 1
    initial_cop = float(sensor.native_value)
    assert sensor._attr_extra_state_attributes is not None
    assert sensor._attr_extra_state_attributes["cop_diagnostics"] == {
        "n_cv_samples": 12
    }

    # Tiny outdoor-temp nudge (< 0.05 COP delta), same generation -> suppressed
    mono_now = 1300.0
    coord.last_raw_row = {
        "flags": FLAG_VALID_WK,
        "t_out_dak_c": 7.1,
        "rh_out_pct": 80.0,
        "f_comp_hz": 24.0,
        "t_wp_out_c": 29.0,
    }
    small_cop = round(eng.predict_cop_cv(7.1, 29.0, 80.0), 2)
    assert abs(small_cop - initial_cop) < 0.05
    sensor._refresh_from_engine(force=False)
    assert len(writes) == 1

    # Significant outdoor-temp drop (>= 0.05 COP delta), same gen -> writes now
    mono_now = 1600.0
    coord.last_raw_row = {
        "flags": FLAG_VALID_WK,
        "t_out_dak_c": 4.0,
        "rh_out_pct": 80.0,
        "f_comp_hz": 24.0,
        "t_wp_out_c": 29.0,
    }
    expected_drop_cop = round(eng.predict_cop_cv(4.0, 29.0, 80.0), 2)
    assert abs(expected_drop_cop - initial_cop) >= 0.05
    sensor._refresh_from_engine(force=False)
    assert len(writes) == 2
    assert sensor.native_value == expected_drop_cop
    assert (
        sensor._attr_extra_state_attributes["live_expected_cop"]
        == expected_drop_cop
    )

    # Unchanged COP before 3600 s -> still suppressed; at >= 3600 s -> heartbeat
    mono_now = 5199.0
    sensor._refresh_from_engine(force=False)
    assert len(writes) == 2

    mono_now = 5200.0  # 5200 - 1600 == 3600 s
    sensor._refresh_from_engine(force=False)
    assert len(writes) == 3

    # Diagnostics mutated without generation increment remain cached until gen++
    eng.cop_diagnostics = {"n_cv_samples": 48, "cop_cv_7c_27c": 3.82}
    mono_now = 5500.0
    sensor._refresh_from_engine(force=False)
    assert len(writes) == 3
    assert sensor._attr_extra_state_attributes["cop_diagnostics"] == {
        "n_cv_samples": 12
    }

    coord.fit_sensor_generation = 2
    sensor._refresh_from_engine(force=False)
    assert len(writes) == 4
    assert sensor._attr_extra_state_attributes["cop_diagnostics"] == {
        "n_cv_samples": 48,
        "cop_cv_7c_27c": 3.82,
    }


def test_zone_fit_sensor_exposes_bound_warnings() -> None:
    """Verify `BetonBatterijZoneFitSensor` forwards `bound_warnings`."""
    base_eng = DigitalTwinEngine()

    class _BoundWarnEngine:
        def __init__(self) -> None:
            self.eval_report_wk = None
            self.eval_report_kt = None
            self.calibration_history: list[dict[str, Any]] = []

        def get_zone_snapshot(self, zone: str) -> Any:
            return base_eng.get_zone_snapshot(zone)  # type: ignore[arg-type]

        def get_zone_fit_diagnostics(self, zone: str) -> dict[str, Any]:
            assert zone == "kt"
            return {
                "posterior_std": {"h_loss_w_k": 2.1},
                "correlation_matrix": {"h_loss_w_k": {"h_loss_w_k": 1.0}},
                "collinearity_warnings": [],
                "bound_warnings": ["h_loss_w_k at lower bound (25.0 W/K)"],
            }

    class _StubCoord:
        fit_sensor_generation = 1
        engine = _BoundWarnEngine()
        ringbuffer = ColumnarRingBuffer()

    sensor = BetonBatterijZoneFitSensor(
        coordinator=_StubCoord(),  # type: ignore[arg-type]
        entry_id="test_entry",
        zone="kt",
        object_id="betonbatterij_fit_kantoor",
        translation_key="fit_kantoor",
    )
    sensor._refresh_from_engine(force=True)
    attrs = sensor._attr_extra_state_attributes
    assert attrs is not None
    assert "bound_warnings" in BetonBatterijZoneFitSensor._unrecorded_attributes
    assert attrs["bound_warnings"] == ["h_loss_w_k at lower bound (25.0 W/K)"]
    json.dumps(attrs, allow_nan=False)


def test_build_lts_row_at_flag_and_p_prim_th_nan_vs_finite() -> None:
    """Verify LTS backfill rows carry `FLAG_LTS_BACKFILL` and honest `p_prim_th_w`."""
    ts = 1_763_000_300
    wk_eid = ROOM_SENSORS["wk"].temp_entity_id
    dak_eid = HOUSE_ENTITIES.outdoor_temp_dak

    base_lookups = {
        wk_eid: _LtsSeriesLookup([(ts - 300, 19.8), (ts, 19.8)]),
        dak_eid: _LtsSeriesLookup([(ts - 300, 6.0), (ts, 6.0)]),
    }

    # Missing WP flow / outlet / inlet -> FLAG_LTS_BACKFILL set, p_prim_th_w is NaN
    row_missing_all = _build_lts_row_at(
        ts=ts,
        lookups=base_lookups,
        lts_units={},
        entities=HOUSE_ENTITIES,
        room_sensors=ROOM_SENSORS,
        prev_buf_top=None,
        prev_buf_bot=None,
    )
    assert row_missing_all is not None
    assert bool(int(row_missing_all["flags"]) & FLAG_LTS_BACKFILL) is True
    assert math.isnan(float(row_missing_all["p_prim_th_w"]))

    # Missing wp_inlet_temp -> p_prim_th_w must still be NaN
    partial_inlet_missing = dict(base_lookups)
    partial_inlet_missing[HOUSE_ENTITIES.wp_pump_flow] = _LtsSeriesLookup(
        [(ts, 15.0)]
    )
    partial_inlet_missing[HOUSE_ENTITIES.wp_outlet_temp] = _LtsSeriesLookup(
        [(ts, 30.5)]
    )
    row_missing_inlet = _build_lts_row_at(
        ts=ts,
        lookups=partial_inlet_missing,
        lts_units={},
        entities=HOUSE_ENTITIES,
        room_sensors=ROOM_SENSORS,
        prev_buf_top=None,
        prev_buf_bot=None,
    )
    assert row_missing_inlet is not None
    assert bool(int(row_missing_inlet["flags"]) & FLAG_LTS_BACKFILL) is True
    assert math.isnan(float(row_missing_inlet["p_prim_th_w"]))

    # Missing wp_pump_flow while temps present -> p_prim_th_w must be NaN
    partial_flow_missing = dict(base_lookups)
    partial_flow_missing[HOUSE_ENTITIES.wp_outlet_temp] = _LtsSeriesLookup(
        [(ts, 30.5)]
    )
    partial_flow_missing[HOUSE_ENTITIES.wp_inlet_temp] = _LtsSeriesLookup(
        [(ts, 27.5)]
    )
    row_missing_flow = _build_lts_row_at(
        ts=ts,
        lookups=partial_flow_missing,
        lts_units={},
        entities=HOUSE_ENTITIES,
        room_sensors=ROOM_SENSORS,
        prev_buf_top=None,
        prev_buf_bot=None,
    )
    assert row_missing_flow is not None
    assert bool(int(row_missing_flow["flags"]) & FLAG_LTS_BACKFILL) is True
    assert math.isnan(float(row_missing_flow["p_prim_th_w"]))

    # All three primary sensors present -> finite p_prim_th_w & FLAG_LTS_BACKFILL
    complete_lookups = dict(partial_inlet_missing)
    complete_lookups[HOUSE_ENTITIES.wp_inlet_temp] = _LtsSeriesLookup(
        [(ts, 27.5)]
    )
    row_complete = _build_lts_row_at(
        ts=ts,
        lookups=complete_lookups,
        lts_units={},
        entities=HOUSE_ENTITIES,
        room_sensors=ROOM_SENSORS,
        prev_buf_top=28.0,
        prev_buf_bot=26.5,
    )
    assert row_complete is not None
    assert bool(int(row_complete["flags"]) & FLAG_LTS_BACKFILL) is True
    expected_w = 69.73 * 15.0 * (30.5 - 27.5)
    assert math.isfinite(float(row_complete["p_prim_th_w"]))
    assert math.isclose(
        float(row_complete["p_prim_th_w"]), expected_w, rel_tol=1e-9
    )

    # End-to-end backfill replay preserves FLAG_LTS_BACKFILL only on LTS rows
    rb = ColumnarRingBuffer()
    stub_eng = _StubEngine()
    lts_start = 1_763_000_100
    raw_start = 1_763_000_700
    end_ts = 1_763_001_000
    added = replay_backfill_into_engine(
        engine=stub_eng,  # type: ignore[arg-type]
        ringbuffer=rb,
        lts_series={
            wk_eid: [(lts_start, 19.5), (raw_start, 19.5)],
            dak_eid: [(lts_start, 6.0), (raw_start, 6.0)],
        },
        lts_units={},
        raw_events=[
            HistoricalStateEvent(
                ts_utc=raw_start,
                entity_id=dak_eid,
                state_str="6.0",
                attributes={},
                last_changed_utc=raw_start,
                last_reported_utc=raw_start,
            ),
            HistoricalStateEvent(
                ts_utc=raw_start,
                entity_id=wk_eid,
                state_str="19.6",
                attributes={},
                last_changed_utc=raw_start,
                last_reported_utc=raw_start,
            ),
        ],
        lts_start_ts=lts_start,
        raw_start_ts=raw_start,
        end_ts=end_ts,
    )
    assert added == 2
    cols = rb.get_columns()
    assert bool(int(cols["flags"][0]) & FLAG_LTS_BACKFILL) is True
    assert math.isnan(float(cols["p_prim_th_w"][0]))
    assert bool(int(cols["flags"][1]) & FLAG_LTS_BACKFILL) is False


def test_build_lts_row_at_ac_mode_summer_winter_and_shoulder() -> None:
    """Verify LTS backfill AC mode distinguishes cooling, heating, & shoulder."""
    ts = 1_763_000_300
    kt_eid = ROOM_SENSORS["kantoor"].temp_entity_id
    dak_eid = HOUSE_ENTITIES.outdoor_temp_dak
    ac_eid = HOUSE_ENTITIES.ac_kantoor_power_w

    def _row_for(t_out: float, t_kt: float, p_ac_w: float) -> dict[str, Any]:
        lookups = {
            kt_eid: _LtsSeriesLookup([(ts - 300, t_kt), (ts, t_kt)]),
            dak_eid: _LtsSeriesLookup([(ts - 300, t_out), (ts, t_out)]),
            ac_eid: _LtsSeriesLookup([(ts - 300, p_ac_w), (ts, p_ac_w)]),
        }
        row = _build_lts_row_at(
            ts=ts,
            lookups=lookups,
            lts_units={},
            entities=HOUSE_ENTITIES,
            room_sensors=ROOM_SENSORS,
            prev_buf_top=None,
            prev_buf_bot=None,
        )
        assert row is not None
        return row

    # Summer cooling: t_out=26.0, t_kt=24.5, p_ac_w=450.0 -> COOL + flag
    row_summer = _row_for(26.0, 24.5, 450.0)
    assert row_summer["ac_mode_code"] == AC_MODE_COOL
    assert bool(int(row_summer["flags"]) & FLAG_AC_COOL_KT) is True

    # Winter heating: t_out=5.0, t_kt=18.0, p_ac_w=450.0 -> HEAT, no cool flag
    row_winter = _row_for(5.0, 18.0, 450.0)
    assert row_winter["ac_mode_code"] == AC_MODE_HEAT
    assert bool(int(row_winter["flags"]) & FLAG_AC_COOL_KT) is False

    # Shoulder / uncertain: t_out=16.0, t_kt=21.8, p_ac_w=450.0 -> OTHER
    row_shoulder = _row_for(16.0, 21.8, 450.0)
    assert row_shoulder["ac_mode_code"] == AC_MODE_OTHER
    assert bool(int(row_shoulder["flags"]) & FLAG_AC_COOL_KT) is False

    # Idle standby (<= 40 W) -> OFF
    row_idle = _row_for(26.0, 24.5, 15.0)
    assert row_idle["ac_mode_code"] == AC_MODE_OFF
    assert bool(int(row_idle["flags"]) & FLAG_AC_COOL_KT) is False


def test_coordinator_manual_calibration_releases_processing_lock() -> None:
    """Prove `_processing_lock` is free during executor solver & 48h sees live rows."""
    real_eng_for_snaps = DigitalTwinEngine()

    class _DelayedThreePhaseEngine:
        def __init__(self) -> None:
            self.solver_started = threading.Event()
            self.allow_solver_finish = threading.Event()
            self.context_calls = 0
            self.solver_input_ts: list[int] = []
            self.solver_modes: list[str] = []
            self.applied_recent_ts: list[int] = []
            self.applied_plans: list[Mapping[str, Any]] = []

        def capture_calibration_context(self) -> dict[str, Any]:
            self.context_calls += 1
            return {"ctx_version": self.context_calls}

        def compute_calibration_plan(
            self,
            columns: Mapping[str, Sequence[float | int]],
            context: Mapping[str, Any],
            mode: str = "manual",
        ) -> dict[str, Any]:
            self.solver_input_ts = [int(ts) for ts in columns["ts_utc"]]
            self.solver_modes.append(mode)
            self.solver_started.set()
            finished = self.allow_solver_finish.wait(timeout=5.0)
            assert finished, "Timed out waiting for allow_solver_finish"
            return {
                "ctx_version": context["ctx_version"],
                "mode": mode,
                "n_train_rows": len(self.solver_input_ts),
            }

        def apply_calibration_plan(
            self,
            plan: Mapping[str, Any],
            recent_columns: Mapping[str, Sequence[float | int]],
        ) -> dict[str, Any]:
            self.applied_plans.append(dict(plan))
            self.applied_recent_ts = [
                int(ts) for ts in recent_columns["ts_utc"]
            ]
            return {
                "accepted_any": True,
                "accepted_wk": True,
                "accepted_kt": True,
                "mode": plan["mode"],
            }

        def get_zone_snapshot(self, zone: str) -> Any:
            return real_eng_for_snaps.get_zone_snapshot(zone)  # type: ignore[arg-type]

        def to_state_dict(self) -> dict[str, Any]:
            return {"episode_bank": {}}

    class _ExecutorHassStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(latitude=51.4, longitude=5.5)

        async def async_add_executor_job(
            self, target: Callable[..., Any], *args: Any
        ) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, target, *args)

    async def _run_scenario() -> None:
        hass_stub = _ExecutorHassStub()
        entry_stub = SimpleNamespace(entry_id="test_entry", options={})
        coord = BetonBatterijCoordinator(
            hass_stub,  # type: ignore[arg-type]
            entry_stub,  # type: ignore[arg-type]
        )

        fake_engine = _DelayedThreePhaseEngine()
        coord.engine = fake_engine  # type: ignore[assignment]

        # Sunday 03:15 auto_weekly with empty ringbuffer exits cleanly
        sunday_dt = datetime(2026, 11, 22, 3, 15, 0, tzinfo=timezone.utc)
        assert sunday_dt.weekday() == 6
        await coord._async_on_weekly_calibration_slot(sunday_dt)
        assert fake_engine.context_calls == 0
        assert coord.fit_sensor_generation == 0

        # Seed 3 initial 5-minute rows
        t0 = 1_763_670_000
        for idx in range(3):
            coord.ringbuffer.append_row(_sample_row(t0 + idx * DT_STEP_S, idx))

        cal_task = asyncio.create_task(coord.async_trigger_manual_calibration())

        # Wait until compute_calibration_plan is actively running in executor
        started = await asyncio.get_running_loop().run_in_executor(
            None, fake_engine.solver_started.wait, 5.0
        )
        assert started is True
        assert coord._calibration_lock.locked() is True
        assert coord._processing_lock.locked() is False
        assert cal_task.done() is False

        # Simulate a live 5-minute tick acquiring _processing_lock immediately
        ts_live_tick = t0 + 3 * DT_STEP_S

        async def _concurrent_5m_tick() -> None:
            async with coord._processing_lock:
                coord.ringbuffer.append_row(_sample_row(ts_live_tick, 3))

        await asyncio.wait_for(_concurrent_5m_tick(), timeout=0.5)
        assert cal_task.done() is False

        # Unblock solver and verify Phase 3 receives the live tick in recent_cols
        fake_engine.allow_solver_finish.set()
        outcome = await asyncio.wait_for(cal_task, timeout=5.0)

        assert outcome["accepted_wk"] is True
        assert outcome["accepted_kt"] is True
        assert fake_engine.solver_input_ts == [
            t0,
            t0 + DT_STEP_S,
            t0 + 2 * DT_STEP_S,
        ]
        assert fake_engine.applied_recent_ts == [
            t0,
            t0 + DT_STEP_S,
            t0 + 2 * DT_STEP_S,
            ts_live_tick,
        ]
        assert coord.fit_sensor_generation == 1
        assert coord.data is not None
        assert coord.data["ringbuffer_rows"] == 4

    asyncio.run(_run_scenario())


def test_coordinator_refresh_14d_eval_preserves_gang_k_hx() -> None:
    """Verify 6h evaluation forwards full context including calibrated `gang`."""
    base_eng = DigitalTwinEngine(k_hx_by_group_w_k={"gang": 142.5})

    class _PureEvalEngineStub:
        def __init__(self) -> None:
            self.eval_report_wk: Any = None
            self.eval_report_kt: Any = None
            self.captured_ctx: Mapping[str, Any] | None = None
            self.refreshed_snapshots = False

        def capture_calibration_context(self) -> dict[str, Any]:
            return base_eng.capture_calibration_context()

        def compute_14d_evaluation_reports(
            self,
            cols: Mapping[str, Sequence[float | int]],
            ctx: Mapping[str, Any] | None = None,
        ) -> tuple[Any, Any]:
            del cols  # Unused in stub.
            self.captured_ctx = ctx
            return ("rep_wk_stub", "rep_kt_stub")

        def _refresh_cached_snapshots(self) -> None:
            self.refreshed_snapshots = True

        def get_zone_snapshot(self, zone: str) -> Any:
            return base_eng.get_zone_snapshot(zone)  # type: ignore[arg-type]

        def to_state_dict(self) -> dict[str, Any]:
            return {"episode_bank": {}}

    class _ExecutorHassStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(latitude=51.4, longitude=5.5)

        async def async_add_executor_job(
            self, target: Callable[..., Any], *args: Any
        ) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, target, *args)

    async def _run() -> None:
        coord = BetonBatterijCoordinator(
            _ExecutorHassStub(),  # type: ignore[arg-type]
            SimpleNamespace(entry_id="test_entry", options={}),  # type: ignore[arg-type]
        )
        stub_eng = _PureEvalEngineStub()
        coord.engine = stub_eng  # type: ignore[assignment]
        coord.ringbuffer.append_row(_sample_row(1_763_670_000, 0))

        await coord._async_refresh_14d_eval(flush=False)

        assert stub_eng.captured_ctx is not None
        assert math.isclose(
            float(stub_eng.captured_ctx["k_hx_by_group_w_k"]["gang"]),
            142.5,
            abs_tol=1e-9,
        )
        assert stub_eng.eval_report_wk == "rep_wk_stub"
        assert stub_eng.eval_report_kt == "rep_kt_stub"
        assert stub_eng.refreshed_snapshots is True
        assert coord.fit_sensor_generation == 1

    asyncio.run(_run())
