# Home Assistant Bridge (`ha_bridge/`)

## Purpose & Layer Boundary

`ha_bridge/` connects Home Assistant event streams, Recorder/Long-Term
Statistics (LTS) history, and `.storage` persistence to the pure-Python
`domain/` layer (`DigitalTwinEngine` and frozen contracts in
`domain/contracts.py`).

- **Owns:** Home Assistant async I/O, state-change subscriptions, sub-5-minute
  time-weighted signal integration, wax-valve ramp tracking, wall-clock jump
  protection, 3-store persistence, cold-start Recorder/LTS backfill, coordinator
  locking, and periodic task scheduling.
- **Does NOT own:** Thermal RC physics, state-space discretization, Kalman
  filtering, Carnot COP equations, MAP parameter optimization, 14-day hindcast
  scoring, or PV curtailment heuristics. All domain calculations live strictly
  in `domain/`.

Core pure-Python primitives (`ClockJumpGuard`, `ColumnarRingBuffer`,
`StepSignalAccumulator`, `WaxValveTracker`, `HydraulicPowerAccumulator`,
`WindowAccumulatorCore`, and `row_utils`) avoid top-level `homeassistant.*`
imports so offline `pytest` suites run without Home Assistant Core installed.

______________________________________________________________________

## Module Responsibilities

| Module            | Owns                                                                                                                                                                                                                                                                                                                                                                                           | Does NOT Own                                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`     | Public re-export surface for pure-Python bridge classes and row helpers only, with zero `homeassistant.*` import attempts at package import time (`BetonBatterijPersistence` is imported directly from `ha_bridge.persistence`).                                                                                                                                                               | Coordinator setup, HA persistence imports, HA event listeners, or domain physics.                                     |
| `_ha_compat.py`   | Guarded `homeassistant.*` imports (`HA_AVAILABLE`) and minimal offline fallback stubs (`DataUpdateCoordinator`, `CoordinatorEntity`, `Store`, `SensorEntity`, `dt_util`, `callback`, event trackers, enum constants).                                                                                                                                                                          | Business logic, persistence policy, or runtime integration state.                                                     |
| `clock.py`        | `ClockJumpGuard` and `MIN_PLAUSIBLE_UTC_EPOCH`: rejects pre-NTP boot epochs and backwards/duplicate UTC ticks on RTC-less hardware while measuring elapsed durations from `time.monotonic()`.                                                                                                                                                                                                  | Event scheduling, restart-gap engine stepping, or timezone formatting.                                                |
| `row_utils.py`    | Canonical 60-column row creation (`make_empty_raw_row`, `2` uint32 + `58` float32), merging engine outputs + open-window flags (`apply_engine_outputs_to_row`), `orjson`-safe sanitization (`sanitize_json_value`, `to_plain_dict`), and rotameter flow-rate clamping (`extract_flow_rates_from_entry`).                                                                                       | RingBuffer storage layout, signal integration, or HA `ConfigEntry` mutation.                                          |
| `storage_ring.py` | `ColumnarRingBuffer` (`array('I')`/`array('f')` C-arrays, `~6.46 MiB` (`6.77 MB`) RAM for `28,224 × 60 × 4 bytes`, `zlib` + dual `crc32` + `base64` chunk codec with zero-recovery on corruption) and `HARingBufferStore` (14 rotating weekly `Store` files).                                                                                                                                  | Engine state/episode storage, Recorder backfill, or coordinator locking.                                              |
| `persistence.py`  | `BetonBatterijPersistence`: 3-store persistence lifecycle (`betonbatterij.state`, `betonbatterij.episode_bank`, `betonbatterij.ringbuffer_w00..13`), offline ring-store fallback stub, and immutable coordinator snapshot payload construction.                                                                                                                                                | Scheduling flushes, lock acquisition, or ringbuffer binary encoding.                                                  |
| `accumulator.py`  | Sub-5m ZOH step integration (`StepSignalAccumulator`), analytical wax-valve stroke integration (`WaxValveTracker`), coupled primary hydraulic power + buffer boundary tracking (`HydraulicPowerAccumulator`), full signal aggregation (`WindowAccumulatorCore`), unit conversions (`W`/`kW`, wind to `m/s`), ZHA change/staleness detection, and HA event-bus adapter (`HAWindowAccumulator`). | Buffer sensor offset correction (`-0.33 °C`), PV curtailment detection, hydraulic flow sharing, or Kalman estimation. |
| `backfill.py`     | Non-blocking cold-start Recorder & LTS backfill (`async_run_cold_start_backfill`), pure-Python replay onto a 300 s UTC grid (`replay_backfill_into_engine`), LTS interpolation (`_LtsSeriesLookup`), and 10-day raw state replay via `WindowAccumulatorCore`.                                                                                                                                  | Live state-change subscriptions, coordinator lock orchestration, or calibration execution.                            |
| `coordinator.py`  | `BetonBatterijCoordinator`: local JSON override loading, store restoration, restart-gap bridging (`FLAG_GAP_BRIDGED`), UTC 5-minute tick loop, 6-hour 14d evaluation + flush, Sunday 03:15 auto-calibration, manual calibration/rollback/reset/export actions, lock discipline, and clean shutdown.                                                                                            | Thermal modeling, optimizer math, UI entity definitions, or direct Recorder SQL queries.                              |

______________________________________________________________________

## Non-Obvious Cross-File Correlations & Invariants

### 1. Locking Hierarchy & Executor Discipline (`coordinator.py`)

`BetonBatterijCoordinator` enforces a strict two-lock ordering to prevent
deadlocks while keeping the 5-minute tick loop responsive during multi-minute
executor jobs:

1. **Outer lock — `_calibration_lock`:** Serializes long-running batch jobs
   (`_async_run_calibration`, `_async_refresh_14d_eval`,
   `async_trigger_rollback`, `async_trigger_reset`).
1. **Inner lock — `_processing_lock`:** Protects mutable coordinator/engine
   state (`ringbuffer`, `engine`, `clock_guard`, `last_raw_row`,
   `auto_calibrate_enabled`, `fit_sensor_generation`, and storage flushes).

Key executor and lifecycle invariants:

- **Never hold `_processing_lock` across executor work:** Heavy operations
  follow a 3-phase pattern:
  1. Acquire `_processing_lock` briefly to snapshot columns and immutable
     context (`capture_calibration_context` /
     `_capture_eval_snapshot_unlocked`). For `async_trigger_rollback()` and
     `async_trigger_reset()`, the fast 48-hour Kalman replay
     (`replay_48h_from_columns`) runs immediately inside Phase 1 before
     releasing `_processing_lock` so any 5-minute tick arriving during the
     executor hindcast steps an already-replayed state under the new parameters
     (and an empty-stack rollback exits early with `False` without flushing
     storage or incrementing `fit_sensor_generation`).
  1. Release `_processing_lock` and run the pure CPU solver/evaluator off the
     event loop via `hass.async_add_executor_job(...)` so incoming 5-minute
     ticks (`_async_on_5m_tick`) continue uninterrupted.
  1. Re-acquire `_processing_lock`, verify `not self._is_shut_down`, apply
     results (`apply_calibration_plan` + 48-hour replay or
     `_apply_eval_reports`), flush storage, and bump `fit_sensor_generation`.
- **Short mutations always acquire `_processing_lock`:** `_async_on_5m_tick`,
  `async_set_auto_calibrate`, `async_update_flow_rates`, and
  `_async_background_cold_start` run under `_processing_lock` (and
  `async_trigger_export_csv` snapshots columns under `_processing_lock` before
  exporting in the executor).
- **Prompt background task cancellation on shutdown:** Startup background tasks
  (`_backfill_task`, `_eval_init_task`) are stored on the coordinator.
  `async_shutdown()` sets `_is_shut_down = True`, unsubscribes all listeners,
  and cancels and awaits both tasks (suppressing `asyncio.CancelledError`)
  **before** acquiring `_processing_lock` to flush storage, so shutdown promptly
  cancels an active cold-start backfill holding `_processing_lock`.

### 2. Ring Buffer Persistence Safety (`storage_ring.py`)

`HARingBufferStore.async_save_dirty()` encodes each modified weekly chunk and
removes `slot` from `ringbuffer.dirty_slots` **before** awaiting
`self._stores[slot].async_save(payload)`. If `async_save` raises, `slot` is
re-added to `dirty_slots` before re-raising. Because `dirty_slots.discard(slot)`
happens prior to the `await` suspension point, any concurrent row append or
output update touching that same chunk during disk I/O marks `slot` dirty again
and is guaranteed to be persisted on the next flush.

On load, `ColumnarRingBuffer.decode_chunk()` validates `format_version`,
`chunk_rows`, column counts, Base64 syntax, compressed `crc32`, decompressed
byte length, and `raw_crc32`. Any mismatch zero-recovers only that single weekly
slot, increments `corrupt_chunks_recovered`, marks the slot dirty, and
recomputes ring cursors without failing integration startup.

### 3. Ring Buffer Physical Capacity vs. Logical Query Window

- **Physical capacity:**
  `RINGBUFFER_NUM_CHUNKS × RINGBUFFER_CHUNK_ROWS = 14 × 2,016 = 28,224` rows (98
  days across `betonbatterij.ringbuffer_w00` through
  `betonbatterij.ringbuffer_w13`).
- **Logical query window:** `RINGBUFFER_ROWS = 25,920` rows (90 days), used as
  the default `max_rows` in `ColumnarRingBuffer.get_columns()`.

The extra 14th weekly chunk provides a rolling active-week buffer so writing new
5-minute rows never overwrites the oldest rows of the 90-day domain lookback
window mid-week.

### 4. Recorder Executor Thread Safety (`backfill.py`)

Home Assistant's state machine (`hass.states`) is not thread-safe. In
`async_run_cold_start_backfill()`, entity `unit_of_measurement` metadata
(`lts_units`) is snapshotted via `hass.states.get(eid)` on the asyncio event
loop **before** dispatching `_query_recorder_db()` to the Recorder executor
(`get_instance(hass).async_add_executor_job`). The Recorder thread returns only
pure-Python tuples and `HistoricalStateEvent` objects, which are then replayed
via `replay_backfill_into_engine()` on a standard HA executor thread.

### 5. Four-Way Room / Group / Valve Topology Mapping

The integration intentionally maintains four distinct cardinalities across
`const.py`, `house_config.py`, `accumulator.py`, and `row_utils.py`:

| Representation                                   | Count  | Details & Mapping Rules                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ALL_HYDRAULIC_GROUPS` (`const.py`)              | **10** | `wk`, `gang`, `slaapkamer`, `badkamer`, `werkkamer_1e`, `kamer_1e_midden`, `kamer_1e_zuid`, `kantoor`, `badkamer_boven`, `logeerkamer`. Used for the 10 rotameter flow-rate options (`extract_flow_rates_from_entry()` clamps each to `[0.1, 25.0] L/min`).                                                                                          |
| `ROOM_SENSORS` / `t_air_*_c` (`house_config.py`) | **9**  | Same keys as above **minus `gang`** (no dedicated hallway thermometer). Supports historical sensor swaps via `pre_swap_temp_entity_id` and `swap_timestamp_utc`.                                                                                                                                                                                     |
| Physical wax valves (`accumulator.py`)           | **11** | `bg` (shared relay driving both `wk` and `gang`), 3 bedroom valves (`slaapkamer_main`, `slaapkamer_voor`, `slaapkamer_achter`), plus 7 single-valve rooms (`badkamer`, `werkkamer_1e`, `kamer_1e_midden`, `kamer_1e_zuid`, `kantoor`, `badkamer_boven`, `logeerkamer`). Each valve integrates its `228 s` stroke analytically via `WaxValveTracker`. |
| `u_eff_*` RingBuffer columns (`const.py`)        | **9**  | `u_eff_bg` drives both `wk` and `gang`; `_close_valves()` collapses the 3 `slaapkamer` valves into a single `u_eff_slaapkamer` (preferring `slaapkamer_main` if it has state, else the mean of `slaapkamer_voor` and `slaapkamer_achter`, else whichever single valve has state, else `0.0`).                                                        |

### 6. Asymmetric Raw Column Semantics (`accumulator.py`, `backfill.py`)

Several columns in the 60-column RingBuffer schema (`2` uint32 + `58` float32)
intentionally differ in calibration, aggregation, or gating semantics:

- **`t_buf_bot_start_c` / `t_buf_bot_end_c` (raw/uncalibrated boundary):**
  Stored strictly raw at window boundaries (`t_0` and `t_1`, carrying `t_1`
  forward as the next window's `t_0`). The `-0.33 °C` sensor calibration offset
  (`BUF_BOT_SENSOR_OFFSET_C`) is applied downstream in `domain/`.
- **`t_bg_ret_c` (offset pre-applied):** Unlike the buffer bottom sensor,
  `HouseEntityConfig.bg_manifold_return_offset_c` is added immediately on
  ingestion in both `accumulator.py` and `backfill.py`.
- **`t_dhw_top_c` / `t_dhw_bot_c` (end-of-window instantaneous):** Store the
  latest valid instantaneous value at window close (falling back to the window
  mean only if currently invalid) rather than a 5-minute average, preserving
  sharp DHW charging-run temperature deltas.
- **`p_prim_th_w` (DHW vs. CV gating switch):** `HydraulicPowerAccumulator`
  integrates instantaneous `69.73 * V_prim(t) * (T_out(t) - T_in(t))` into both
  Room-gated (`p_prim_th_w`) and ungated (`p_prim_raw_w`) totals. When
  `FLAG_DHW_TANK` is set (`dhw_duty >= 0.5`), `_assemble_raw_row()` writes
  `hyd.p_prim_raw_w` into `p_prim_th_w` so DHW thermal output is preserved for
  COP analysis while `domain/hydraulics.py` zeroes buffer heat injection under
  `FLAG_DHW_TANK`.
- **`p_pv_w` (raw telemetry only):** Kept raw (`NaN` when unmeasured).
  `FLAG_PV_FALLBACK` is raised solely for genuine PV sensor unavailability or
  `< 50%` window coverage when `p_pv_forecast_w > 0`. Curtailment detection
  lives exclusively in `domain/solar_weather.py`.

### 7. Live vs. 10-Day Raw vs. 90-Day LTS Backfill Parity

- **Live + 10-day raw Recorder backfill share one core:** Both
  `HAWindowAccumulator` and the `[now - 10 d, now]` raw Recorder pass in
  `replay_backfill_into_engine()` feed `WindowAccumulatorCore` (`ingest_state`
  followed by `close_window`), guaranteeing identical wax-valve ramps, coupled
  hydraulic integration, ZHA staleness/change tracking (`7,200 s` threshold),
  and status bitmask derivation.
- **10–90 day LTS backfill (`_build_lts_row_at()`):** Queries `hour` and
  `5minute` Recorder statistics, interpolates via `_LtsSeriesLookup`
  (`max_hold_s = 7,200 s`), and marks every row with `FLAG_LTS_BACKFILL`.
  Because binary/transient sensors lack LTS series, it:
  - Omits transient status flags (`FLAG_DEFROST_WP`, `FLAG_DEFROST_AC`,
    `FLAG_DHW_TANK`, `FLAG_INTERNAL_HEATER`, `FLAG_Z1_OFF_HARVEST`,
    `FLAG_SUMMER`, `FLAG_FLOOR_COOL`, `FLAG_PV_FALLBACK`).
  - Infers `u_pump_sec = u_eff_bg = 1.0` whenever `p_prim_th_w > 250.0 W` (with
    upper-floor valves defaulted to `0.0`).
  - Infers attic split-AC mode when `p_el_ac_kt_w > 40.0 W`: `AC_MODE_COOL` (+
    `FLAG_AC_COOL_KT`) if `t_air_kantoor_c >= 22.5 °C` or outdoor temp
    `>= 17.0 °C`; `AC_MODE_HEAT` if outdoor temp `< 15.0 °C` and
    `t_air_kantoor_c < 21.5 °C`; otherwise `AC_MODE_OTHER` (`AC_MODE_OFF` when
    `<= 40.0 W`).
  - Uses neutral defaults for unrecorded weather/screen signals
    (`v_wind_ms = 3.0`, `rh_out_pct = 80.0`, `screen_*_pos = 100.0`).
- **Schema synchronization (`row_utils.py`):** Every row producer initializes
  dictionaries through `make_empty_raw_row()` and merges engine outputs
  (`t_floor_*`, `d_est_*`, `q_fh_*`, `t_hind6h_*`, plus `FLAG_OPEN_WIN_WK` /
  `FLAG_OPEN_WIN_KT`) through `apply_engine_outputs_to_row()`, ensuring strict
  adherence to the 60-column schema (`2` uint32 + `58` float32) in `const.py`.

### 8. Clock Jumps, Restart Bridging & Offline Testability

- **Clock guard & gap bridging (`clock.py`, `coordinator.py`):**
  `ClockJumpGuard` anchors `min_valid_utc` at
  `max(MIN_PLAUSIBLE_UTC_EPOCH, saved_wall_utc)` and rejects pre-NTP or
  non-monotonic ticks while computing step durations from `time.monotonic()`. On
  restart, gaps in `[300 s, MAX_RESTART_BRIDGE_GAP_S]` (`10,800 s` / 3 h) are
  bridged in one engine step tagged with `FLAG_GAP_BRIDGED`.
- **Offline testability (`_ha_compat.py`):** When Home Assistant Core is not
  installed (`HA_AVAILABLE == False`), `_ha_compat.py` supplies lightweight
  drop-in stubs (`Store`, `DataUpdateCoordinator`, `CoordinatorEntity`,
  `SensorEntity`, `dt_util`, `callback`, event trackers, and Core/Sensor enums)
  and `BetonBatterijPersistence` substitutes `_OfflineRingStoreStub`. This
  allows the full `ha_bridge` unit test suite to execute via `python -m pytest`
  in seconds with zero HA Core runtime dependency.
