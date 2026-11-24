"""Foundation contract tests, boundary AST guards, and TDD bug regressions.

Covers:
1. Foundation contracts (`const.py` bounds, `ZoneParameters` / `CopParameters`
   `to_dict`/`from_dict`, `ZoneSnapshot.warmup_remaining_min`,
   `ThermalTwinProtocol`, `domain/priors.py` purity, and `house_config.py`
   isolation from `domain/`).
2. Deterministic, fast (`< 50 ms`) TDD regression tests for Bugs 1, 3, 4, 5a,
   6, 7, 8a, 8b, and 8c plus coordinator rollback/shutdown hardenings.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Callable, Mapping, Sequence
import math
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from custom_components.betonbatterij import const
from custom_components.betonbatterij import house_config
from custom_components.betonbatterij.domain import calibration
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import cop_model
from custom_components.betonbatterij.domain import engine
from custom_components.betonbatterij.domain import evaluation
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import solar_weather
from custom_components.betonbatterij.ha_bridge import accumulator
from custom_components.betonbatterij.ha_bridge import backfill
from custom_components.betonbatterij.ha_bridge import coordinator as coord_mod
from custom_components.betonbatterij.ha_bridge import storage_ring
from custom_components.betonbatterij.sensor import BetonBatterijCopFitSensor


def _minimal_row(ts_utc: int) -> dict[str, float | int]:
    """Builds a minimal valid 60-column RingBuffer row at `ts_utc`."""
    row: dict[str, float | int] = {col: 0.0 for col in const.RINGBUFFER_ALL_COLS}
    row["ts_utc"] = ts_utc
    row["flags"] = (
        const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
    )
    row["t_air_wk_c"] = 20.0
    row["t_air_kantoor_c"] = 19.0
    row["t_out_dak_c"] = 5.0
    row["rh_out_pct"] = 80.0
    row["v_wind_ms"] = 3.0
    row["screen_small_pos"] = 100.0
    row["screen_large_pos"] = 100.0
    row["t_buf_top_start_c"] = 28.0
    row["t_buf_top_end_c"] = 28.0
    row["t_buf_bot_start_c"] = 26.0
    row["t_buf_bot_end_c"] = 26.0
    row["p_desk_kt_w"] = float("nan")
    row["t_bg_sup_c"] = float("nan")
    row["t_bg_ret_c"] = float("nan")
    row["t_floor_surf_meas_wk_c"] = float("nan")
    return row


def _make_eval_report(
    zone: contracts.ZoneId, j_eval_c: float, ts_utc: int
) -> contracts.ZoneEvaluationReport:
    """Builds a synthetic `ZoneEvaluationReport` with a target `j_eval_c`."""
    metrics = contracts.RegimeHindcastMetrics(
        regime="overall_6h",
        horizon_steps=preprocessing.HORIZON_6H_STEPS,
        n_windows=24,
        rmse_end_c=j_eval_c,
        rmse_path_c=j_eval_c,
        mae_end_c=j_eval_c * 0.8,
        bias_end_c=0.0,
        j_eval_c=j_eval_c,
        prior_rmse_end_c=0.60,
        skill_score_pct=20.0,
    )
    return contracts.ZoneEvaluationReport(
        zone=zone,
        ts_evaluated_utc=ts_utc,
        overall_6h=metrics,
        by_regime={"overall_6h": metrics},
    )


# ============================================================================
# Wave 1 Foundation Contract & Boundary Tests
# ============================================================================
def test_foundation_const_contracts_priors_and_protocol() -> None:
    """Verifies Wave 1 foundation symbols, serialization, and Protocol."""
    assert const.FLOW_RATE_MIN_L_MIN == pytest.approx(0.1)
    assert const.FLOW_RATE_MAX_L_MIN == pytest.approx(25.0)
    assert const.MAX_RESTART_BRIDGE_GAP_S == pytest.approx(10_800.0)

    # ZoneParameters to_dict / from_dict roundtrip + non-finite fallback
    p_dict = priors.DAY1_PRIORS_WK.to_dict()
    assert contracts.ZoneParameters.from_dict(p_dict) == priors.DAY1_PRIORS_WK
    corrupt = dict(p_dict)
    corrupt["h_loss_w_k"] = float("nan")
    corrupt["a_sol_m2"] = "not-a-number"
    corrupt["c_floor_wh_k"] = True
    restored = contracts.ZoneParameters.from_dict(
        corrupt, defaults=priors.DAY1_PRIORS_WK
    )
    assert restored == priors.DAY1_PRIORS_WK

    with pytest.raises(ValueError):
        contracts.ZoneParameters.from_dict({"h_loss_w_k": 50.0})

    # CopParameters to_dict / from_dict roundtrip + fallback
    c_dict = priors.DAY1_COP_PARAMS.to_dict()
    assert contracts.CopParameters.from_dict(c_dict) == priors.DAY1_COP_PARAMS
    assert (
        contracts.CopParameters.from_dict({"eta_carnot_cv": float("inf")})
        == priors.DAY1_COP_PARAMS
    )

    # house_config owns ONLY HA entity defaults — zero domain/priors re-exports
    for prior_name in (
        "DAY1_PRIORS_WK",
        "DAY1_PRIORS_KT",
        "DAY1_COP_PARAMS",
        "PARAM_BOUNDS_WK",
        "DAY1_K_HX_BY_GROUP_W_K",
    ):
        assert not hasattr(house_config, prior_name)

    # ZoneSnapshot default warmup_remaining_min & ThermalTwinProtocol conformance
    eng = engine.DigitalTwinEngine()
    assert isinstance(eng, contracts.ThermalTwinProtocol)
    snap = eng.get_zone_snapshot("wk")
    assert snap.warmup_remaining_min == 0


def test_house_config_optional_entity_field_validation_errors(
    tmp_path: Path,
) -> None:
    """Verifies `_validate_optional_entity_field` preserves exact error paths."""
    bad_ent = tmp_path / "bad_ent.json"
    bad_ent.write_text('{"entities": {"desk_kantoor_power_w": 123}}')
    with pytest.raises(
        ValueError, match="Expected string or null for 'desk_kantoor_power_w'"
    ):
        house_config.load_house_config(bad_ent)

    bad_hum = tmp_path / "bad_hum.json"
    bad_hum.write_text(
        '{"room_sensors": {"wk": {"humidity_entity_id": 42}}}'
    )
    with pytest.raises(
        ValueError,
        match="Expected string or null for room_sensors.wk.humidity_entity_id",
    ):
        house_config.load_house_config(bad_hum)

    bad_pre = tmp_path / "bad_pre.json"
    bad_pre.write_text(
        '{"room_sensors": {"wk": {"pre_swap_temp_entity_id": false}}}'
    )
    with pytest.raises(
        ValueError,
        match=(
            "Expected string or null for "
            "room_sensors.wk.pre_swap_temp_entity_id"
        ),
    ):
        house_config.load_house_config(bad_pre)


# ============================================================================
# TDD Regression Tests for Bugs 1, 3, 4, 5a, 6, 7, 8a, 8b, 8c
# ============================================================================
def test_bug1_auto_weekly_khx_single_damping_consistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug 1: `auto_weekly` must damp `k_hx` ONCE (0.20) for wk, kt, & gang."""
    eng = engine.DigitalTwinEngine()
    ctx = eng.capture_calibration_context()
    old_khx = dict(ctx["k_hx_by_group_w_k"])
    old_wk = old_khx["wk"]
    old_kt = old_khx["kantoor"]
    old_gang = old_khx["gang"]

    # Minimal 4-day + 10-step dataset so train/holdout split activates quickly.
    n_rows = 4 * const.STEPS_PER_DAY + 10
    base_ts = 1_763_000_000
    cols: dict[str, list[float | int]] = {
        "ts_utc": [base_ts + i * const.DT_STEP_S for i in range(n_rows)],
        "flags": [
            const.FLAG_VALID_WK | const.FLAG_VALID_KT | const.FLAG_AANWEZIG
        ]
        * n_rows,
        "t_air_wk_c": [20.0] * n_rows,
        "t_air_kantoor_c": [19.0] * n_rows,
        "t_out_dak_c": [5.0] * n_rows,
        "v_wind_ms": [3.0] * n_rows,
        "p_pv_w": [0.0] * n_rows,
    }

    monkeypatch.setattr(
        calibration,
        "calibrate_k_hx_robust",
        lambda *args, **kwargs: {
            g: old_khx[g] + 100.0 for g in const.ALL_HYDRAULIC_GROUPS
        },
    )
    monkeypatch.setattr(
        calibration,
        "check_seasonal_calibration_gate",
        lambda *args, **kwargs: (True, "ok"),
    )
    monkeypatch.setattr(
        calibration,
        "extract_clean_regime_windows",
        lambda *args, **kwargs: [
            calibration.TrainingWindow(
                regime="night", k0=100, length_steps=36, quality_score=1.0
            )
        ],
    )

    def _stub_solver(
        residual_fn: Callable[[Sequence[float]], list[float]],
        theta0: Sequence[float],
        *args: Any,
        **kwargs: Any,
    ) -> calibration.SolverResult:
        del residual_fn, args, kwargs
        p = len(theta0)
        return calibration.SolverResult(
            theta=list(theta0),
            success=True,
            n_iter=1,
            n_fev=1,
            cost=0.1,
            posterior_std=[1.0] * p,
            posterior_cov=[
                [1.0 if i == j else 0.0 for j in range(p)] for i in range(p)
            ],
            fisher_diag=[1.0] * p,
            correlation_matrix=[
                [1.0 if i == j else 0.0 for j in range(p)] for i in range(p)
            ],
            collinearity_pairs=[],
            message="stubbed",
        )

    monkeypatch.setattr(calibration, "solve_b2_box_huber_lm", _stub_solver)

    def _stub_eval(
        columns: Mapping[str, Sequence[float | int]],
        zone: contracts.ZoneId,
        params_eval: contracts.ZoneParameters,
        *args: Any,
        **kwargs: Any,
    ) -> contracts.ZoneEvaluationReport:
        del columns, args, kwargs
        ref_khx = old_wk if zone == "wk" else old_kt
        is_before = abs(params_eval.k_hx_w_k - ref_khx) <= 1e-6
        return _make_eval_report(
            zone, j_eval_c=0.50 if is_before else 0.30, ts_utc=base_ts
        )

    monkeypatch.setattr(evaluation, "evaluate_14d_hindcast", _stub_eval)
    monkeypatch.setattr(
        cop_model,
        "fit_cop_parameters",
        lambda columns, current_params, memory, **kwargs: (
            current_params,
            memory
            or contracts.ParameterPosteriorMemory(mean={}, precision={}),
            {},
        ),
    )

    plan = eng.compute_calibration_plan(cols, ctx, mode="auto_weekly")
    assert plan["accepted_wk"] is True
    assert plan["accepted_kt"] is True
    assert plan["next_k_hx"]["wk"] == pytest.approx(old_wk + 20.0)
    assert plan["next_k_hx"]["kantoor"] == pytest.approx(old_kt + 20.0)
    assert plan["next_k_hx"]["gang"] == pytest.approx(old_gang + 20.0)


def test_bug3_p_desk_kt_w_negative_glitch_rejected_in_engine() -> None:
    """Bug 3: negative `p_desk_kt_w` glitch must trigger PIR fallback."""
    row = _minimal_row(1_763_670_300)
    row["p_desk_kt_w"] = -50.0
    row["pir_occ_kt"] = 1.0
    row["p_growlights_kt_w"] = 0.0
    row["p_el_ac_kt_w"] = 0.0

    cols = {k: [v] for k, v in row.items()}
    hist = preprocessing.preprocess_history_columns(cols)
    assert hist.q_int_kt_w[0] == pytest.approx(
        solar_weather.PIR_WORKPLACE_GAIN_W
    )

    eng = engine.DigitalTwinEngine()
    eng.process_5m_window(row)
    assert eng._desk_state_kt.get("pir_hold_s", 0.0) > 0.0


def test_bug4_preprocess_and_recompute_use_midpoint_t_sup_sec() -> None:
    """Bug 4: `t_sup_sec_c` must use midpoint `(start+end)/2` in both paths."""
    row = _minimal_row(1_763_670_300)
    row["flags"] = const.FLAG_VALID_WK
    row["t_buf_top_start_c"] = 25.0
    row["t_buf_top_end_c"] = 35.0
    row["t_buf_bot_start_c"] = 23.0
    row["t_buf_bot_end_c"] = 33.0
    row["u_pump_sec"] = 1.0
    row["u_eff_bg"] = 1.0
    row["v_prim_l_min"] = 15.0

    cols = {k: [v] for k, v in row.items()}
    hist = preprocessing.preprocess_history_columns(cols)
    assert hist.t_sup_sec_c[0] == pytest.approx(30.0)

    hist_recomp = preprocessing.recompute_history_for_khx(
        hist,
        cols,
        k_hx_by_group={"wk": priors.DAY1_K_HX_BY_GROUP_W_K["wk"] + 10.0},
    )
    assert hist_recomp.t_sup_sec_c[0] == pytest.approx(30.0)


def test_bug5a_cop_fit_sensor_fallback_matches_domain_cop_model() -> None:
    """Bug 5a: COP sensor fallback must match `cop_model.predict_cop_cv`."""
    coord_stub = SimpleNamespace(
        fit_sensor_generation=1,
        engine=SimpleNamespace(
            cop_params=priors.DAY1_COP_PARAMS,
            cop_diagnostics={},
        ),
        last_raw_row={
            "t_out_dak_c": 1.0,
            "t_wp_out_c": 28.0,
            "rh_out_pct": 95.0,
        },
    )
    sensor = BetonBatterijCopFitSensor(
        coordinator=coord_stub,  # type: ignore[arg-type]
        entry_id="test_entry",
    )
    got = sensor._compute_live_expected_cop(
        priors.DAY1_COP_PARAMS.to_dict(), {}
    )
    expected = cop_model.predict_cop_cv(
        priors.DAY1_COP_PARAMS, 1.0, 28.0, 95.0
    )
    assert got == pytest.approx(expected, abs=1e-9)


def test_bug6_accumulator_preserves_raw_pv_and_does_not_override_domain_curtailment() -> (
    None
):
    """Bug 6: accumulator preserves raw `p_pv_w` and defers curtailment."""
    # (a) Small PV during negative EPEX with forecast=200 W (< 350 W domain
    # threshold): accumulator must NOT set FLAG_PV_FALLBACK.
    core_a = accumulator.WindowAccumulatorCore(t_mono=0.0)
    core_a.ingest_state(
        house_config.HOUSE_ENTITIES.pv_power_w,
        "50.0",
        {"unit_of_measurement": "W"},
        t_mono=0.0,
        is_initial_seed=True,
    )
    core_a.ingest_state(
        house_config.HOUSE_ENTITIES.pv_forecast_now_w,
        "200.0",
        {"unit_of_measurement": "W"},
        t_mono=0.0,
        is_initial_seed=True,
    )
    core_a.ingest_state(
        house_config.HOUSE_ENTITIES.epex_price_eur_kwh,
        "-0.05",
        {},
        t_mono=0.0,
        is_initial_seed=True,
    )
    res_a = core_a.close_window(
        ts_utc=1_763_670_300, t_mono=300.0, window_dt=300.0
    )
    assert (int(res_a.raw_row["flags"]) & const.FLAG_PV_FALLBACK) == 0
    assert float(res_a.raw_row["p_pv_w"]) == pytest.approx(50.0)

    # (b) Forecast present with NO p_pv_w reading: raw p_pv_w must remain NaN.
    core_b = accumulator.WindowAccumulatorCore(t_mono=0.0)
    core_b.ingest_state(
        house_config.HOUSE_ENTITIES.pv_forecast_now_w,
        "1200.0",
        {"unit_of_measurement": "W"},
        t_mono=0.0,
        is_initial_seed=True,
    )
    res_b = core_b.close_window(
        ts_utc=1_763_670_300, t_mono=300.0, window_dt=300.0
    )
    assert math.isnan(float(res_b.raw_row["p_pv_w"]))


def test_bug7_backfill_does_not_call_hass_states_get_from_executor_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug 7: `hass.states.get` must only be called on the event-loop thread."""

    async def _run() -> list[str]:
        loop_tid = threading.get_ident()
        off_loop_calls: list[str] = []

        class _StatesStub:
            def get(self, entity_id: str) -> Any:
                if threading.get_ident() != loop_tid:
                    off_loop_calls.append(entity_id)
                return SimpleNamespace(attributes={"unit_of_measurement": "W"})

        class _RecInstanceStub:
            async def async_add_executor_job(
                self, fn: Callable[..., Any], *args: Any
            ) -> Any:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, fn, *args)

        hass_stub = SimpleNamespace(
            config=SimpleNamespace(components={"recorder"}),
            states=_StatesStub(),
        )

        # Inject minimal stub `homeassistant.components.recorder` module tree.
        ha_mod = ModuleType("homeassistant")
        comp_mod = ModuleType("homeassistant.components")
        rec_mod = ModuleType("homeassistant.components.recorder")
        hist_mod = ModuleType("homeassistant.components.recorder.history")
        stat_mod = ModuleType("homeassistant.components.recorder.statistics")
        stat_mod.statistics_during_period = (  # type: ignore[attr-defined]
            lambda *args, **kwargs: {}
        )
        hist_mod.get_significant_states = (  # type: ignore[attr-defined]
            lambda *args, **kwargs: {}
        )
        rec_mod.get_instance = (  # type: ignore[attr-defined]
            lambda _hass: _RecInstanceStub()
        )
        rec_mod.history = hist_mod  # type: ignore[attr-defined]
        rec_mod.statistics = stat_mod  # type: ignore[attr-defined]
        comp_mod.recorder = rec_mod  # type: ignore[attr-defined]
        ha_mod.components = comp_mod  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "homeassistant", ha_mod)
        monkeypatch.setitem(sys.modules, "homeassistant.components", comp_mod)
        monkeypatch.setitem(
            sys.modules, "homeassistant.components.recorder", rec_mod
        )
        monkeypatch.setitem(
            sys.modules, "homeassistant.components.recorder.history", hist_mod
        )
        monkeypatch.setitem(
            sys.modules,
            "homeassistant.components.recorder.statistics",
            stat_mod,
        )

        rb = storage_ring.ColumnarRingBuffer(chunk_rows=16, num_chunks=2)
        eng = engine.DigitalTwinEngine()
        ring_store_stub = SimpleNamespace(
            async_save_dirty=lambda: asyncio.sleep(0)
        )
        await backfill.async_run_cold_start_backfill(
            hass=hass_stub,  # type: ignore[arg-type]
            engine=eng,
            ringbuffer=rb,
            ring_store=ring_store_stub,  # type: ignore[arg-type]
            now_utc=1_763_670_300.0,
        )
        return off_loop_calls

    assert asyncio.run(_run()) == []


def test_bug8a_ring_store_preserves_dirty_slot_written_during_async_save() -> (
    None
):
    """Bug 8a: concurrent write to slot `s` during `async_save` stays dirty."""

    async def _run() -> None:
        rb = storage_ring.ColumnarRingBuffer(chunk_rows=16, num_chunks=2)
        slot = rb.append_row(_minimal_row(1_763_670_300))
        assert slot in rb.dirty_slots

        save_entered = asyncio.Event()
        allow_save_finish = asyncio.Event()

        class _SuspendedStore:
            async def async_save(self, payload: Mapping[str, Any]) -> None:
                del payload
                save_entered.set()
                await allow_save_finish.wait()

        ring_store = object.__new__(storage_ring.HARingBufferStore)
        ring_store.ringbuffer = rb
        ring_store._stores = [  # type: ignore[assignment]
            _SuspendedStore() for _ in range(rb.num_chunks)
        ]

        save_task = asyncio.create_task(ring_store.async_save_dirty())
        await asyncio.wait_for(save_entered.wait(), timeout=2.0)

        # Write a second row into the same slot while async_save is suspended.
        slot2 = rb.append_row(_minimal_row(1_763_670_600))
        assert slot2 == slot

        allow_save_finish.set()
        await asyncio.wait_for(save_task, timeout=2.0)
        assert slot in rb.dirty_slots

    asyncio.run(_run())


def test_bug8b_coordinator_rollback_and_reset_release_processing_lock_during_executor() -> (
    None
):
    """Bug 8b: rollback and reset must not hold `_processing_lock` in executor."""

    class _GatedExecutorHassStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(latitude=51.4, longitude=5.5)
            self.job_started = threading.Event()
            self.allow_job_finish = threading.Event()

        async def async_add_executor_job(
            self, target: Callable[..., Any], *args: Any
        ) -> Any:
            def _wrapped() -> Any:
                self.job_started.set()
                assert self.allow_job_finish.wait(timeout=5.0)
                return target(*args)

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _wrapped)

    async def _run() -> None:
        hass_stub = _GatedExecutorHassStub()
        entry_stub = SimpleNamespace(entry_id="test_entry", options={})
        coord = coord_mod.BetonBatterijCoordinator(
            hass_stub,  # type: ignore[arg-type]
            entry_stub,  # type: ignore[arg-type]
        )
        coord.ringbuffer.append_row(_minimal_row(1_763_670_300))
        coord.ringbuffer.append_row(_minimal_row(1_763_670_600))
        replay_calls: list[int] = []
        orig_replay = coord.engine.replay_48h_from_columns

        def _tracked_replay(cols: Mapping[str, Sequence[float | int]]) -> None:
            replay_calls.append(len(cols.get("ts_utc", ())))
            orig_replay(cols)

        coord.engine.replay_48h_from_columns = _tracked_replay  # type: ignore[method-assign]

        # 0. Empty-stack rollback must return False without flushing or bumping gen.
        flush_calls = 0
        orig_flush = coord.async_flush_storage

        async def _tracked_flush() -> None:
            nonlocal flush_calls
            flush_calls += 1
            await orig_flush()

        coord.async_flush_storage = _tracked_flush  # type: ignore[method-assign]
        empty_rb = await coord.async_trigger_rollback()
        assert empty_rb is False
        assert flush_calls == 0
        assert coord.fit_sensor_generation == 0
        assert replay_calls == []

        coord.engine.snapshot_stack.append(
            coord.engine._capture_snapshot(reason="test_snap")
        )

        # 1. Rollback runs 48h replay in Phase 1 and releases lock during executor.
        rb_task = asyncio.create_task(coord.async_trigger_rollback())
        started = await asyncio.get_running_loop().run_in_executor(
            None, hass_stub.job_started.wait, 5.0
        )
        assert started is True
        locked_during_rollback = coord._processing_lock.locked()
        replays_before_rb_exec_finish = len(replay_calls)
        hass_stub.allow_job_finish.set()
        rolled_back = await asyncio.wait_for(rb_task, timeout=5.0)
        assert rolled_back is True
        assert locked_during_rollback is False
        assert replays_before_rb_exec_finish == 1
        assert len(replay_calls) == 1

        # 2. Reset runs 48h replay in Phase 1 and releases lock during executor.
        hass_stub.job_started.clear()
        hass_stub.allow_job_finish.clear()
        reset_task = asyncio.create_task(coord.async_trigger_reset())
        started2 = await asyncio.get_running_loop().run_in_executor(
            None, hass_stub.job_started.wait, 5.0
        )
        assert started2 is True
        locked_during_reset = coord._processing_lock.locked()
        replays_before_reset_exec_finish = len(replay_calls)
        hass_stub.allow_job_finish.set()
        await asyncio.wait_for(reset_task, timeout=5.0)
        assert locked_during_reset is False
        assert replays_before_reset_exec_finish == 2
        assert len(replay_calls) == 2

    asyncio.run(_run())


def test_coordinator_shutdown_cancels_active_backfill_holding_lock() -> None:
    """Shutdown cancels `_backfill_task` holding `_processing_lock` promptly."""

    async def _run() -> None:
        hass_stub = SimpleNamespace(
            config=SimpleNamespace(latitude=51.4, longitude=5.5)
        )
        entry_stub = SimpleNamespace(entry_id="test_entry", options={})
        coord = coord_mod.BetonBatterijCoordinator(
            hass_stub,  # type: ignore[arg-type]
            entry_stub,  # type: ignore[arg-type]
        )

        backfill_locked = asyncio.Event()
        backfill_cancelled = False

        async def _stuck_backfill() -> None:
            nonlocal backfill_cancelled
            async with coord._processing_lock:
                backfill_locked.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    backfill_cancelled = True
                    raise

        coord._backfill_task = asyncio.create_task(_stuck_backfill())
        await asyncio.wait_for(backfill_locked.wait(), timeout=2.0)
        assert coord._processing_lock.locked() is True

        await asyncio.wait_for(coord.async_shutdown(), timeout=2.0)
        assert backfill_cancelled is True
        assert coord._is_shut_down is True
        assert coord._processing_lock.locked() is False

    asyncio.run(_run())


def test_bug8c_coordinator_auto_calibrate_and_shutdown_acquire_processing_lock() -> (
    None
):
    """Bug 8c: `async_set_auto_calibrate` & `async_shutdown` wait for lock."""

    async def _run() -> None:
        hass_stub = SimpleNamespace(
            config=SimpleNamespace(latitude=51.4, longitude=5.5)
        )
        entry_stub = SimpleNamespace(entry_id="test_entry", options={})
        coord = coord_mod.BetonBatterijCoordinator(
            hass_stub,  # type: ignore[arg-type]
            entry_stub,  # type: ignore[arg-type]
        )

        async with coord._processing_lock:
            t_auto = asyncio.create_task(
                coord.async_set_auto_calibrate(False, persist=True)
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            auto_bypassed_lock = t_auto.done()
        await asyncio.wait_for(t_auto, timeout=2.0)
        assert auto_bypassed_lock is False
        assert coord.auto_calibrate_enabled is False

        async with coord._processing_lock:
            t_shut = asyncio.create_task(coord.async_shutdown())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            shut_bypassed_lock = t_shut.done()
        await asyncio.wait_for(t_shut, timeout=2.0)
        assert shut_bypassed_lock is False
        assert coord._is_shut_down is True

    asyncio.run(_run())


# ============================================================================
# Architectural Boundary & Purity AST Guard Tests
# ============================================================================
def test_domain_layer_ast_purity_and_acyclic_dag() -> None:
    """Verifies `domain/` has zero forbidden imports and a strict import DAG."""
    domain_dir = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "betonbatterij"
        / "domain"
    )
    py_files = sorted(domain_dir.glob("*.py"))
    assert len(py_files) >= 13

    forbidden_modules = {"homeassistant", "numpy", "scipy", "house_config"}
    module_names = {p.stem for p in py_files}
    domain_pkg = "custom_components.betonbatterij.domain"
    graph: dict[str, set[str]] = {name: set() for name in module_names}
    purity_violations: list[str] = []

    for py_file in py_files:
        mod_name = py_file.stem
        tree = ast.parse(
            py_file.read_text(encoding="utf-8"), filename=str(py_file)
        )

        # 1. Purity check across the entire AST (including functions & typing).
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = set(alias.name.split("."))
                    bad = parts & forbidden_modules
                    if bad:
                        purity_violations.append(
                            f"{py_file.name}:{node.lineno} imports "
                            f"{alias.name!r}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod_parts = (
                    set(node.module.split(".")) if node.module else set()
                )
                bad_mod = mod_parts & forbidden_modules
                if bad_mod:
                    purity_violations.append(
                        f"{py_file.name}:{node.lineno} imports from "
                        f"{node.module!r}"
                    )
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_modules:
                        purity_violations.append(
                            f"{py_file.name}:{node.lineno} imports "
                            f"{alias.name!r} from {node.module!r}"
                        )

        # 2. Top-level runtime intra-domain dependency extraction.
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                if stmt.level == 0 and stmt.module == domain_pkg:
                    for alias in stmt.names:
                        if alias.name in module_names:
                            graph[mod_name].add(alias.name)
                elif stmt.level == 0 and (
                    stmt.module is not None
                    and stmt.module.startswith(f"{domain_pkg}.")
                ):
                    target = stmt.module[len(domain_pkg) + 1 :].split(".")[0]
                    if target in module_names:
                        graph[mod_name].add(target)
                elif stmt.level == 1:
                    if stmt.module is None:
                        for alias in stmt.names:
                            if alias.name in module_names:
                                graph[mod_name].add(alias.name)
                    else:
                        target = stmt.module.split(".")[0]
                        if target in module_names:
                            graph[mod_name].add(target)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name.startswith(f"{domain_pkg}."):
                        target = alias.name[len(domain_pkg) + 1 :].split(".")[0]
                        if target in module_names:
                            graph[mod_name].add(target)

    assert purity_violations == [], (
        f"Forbidden imports in domain/: {purity_violations}"
    )

    # 3. Verify strict DAG via 3-color DFS (0=unvisited, 1=visiting, 2=done).
    color: dict[str, int] = {name: 0 for name in module_names}
    cycles: list[list[str]] = []

    def _dfs(node_name: str, stack: list[str]) -> None:
        color[node_name] = 1
        stack.append(node_name)
        for dep in sorted(graph[node_name]):
            if color[dep] == 1:
                idx = stack.index(dep)
                cycles.append(stack[idx:] + [dep])
            elif color[dep] == 0:
                _dfs(dep, stack)
        stack.pop()
        color[node_name] = 2

    for name in sorted(module_names):
        if color[name] == 0:
            _dfs(name, [])

    assert cycles == [], f"Circular imports detected in domain/: {cycles}"


def test_no_forbidden_cross_module_private_calls() -> None:
    """AST-guards against forbidden cross-module private calls & accesses."""
    pkg_dir = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "betonbatterij"
    )
    violations: list[str] = []

    def _parse(rel_path: str) -> ast.Module:
        path = pkg_dir / rel_path
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    # 1. ha_bridge/coordinator.py: no `_refresh_cached_snapshots`,
    # `_k_hx_by_group_w_k`, or `accumulator.core`.
    coord_tree = _parse("ha_bridge/coordinator.py")
    for node in ast.walk(coord_tree):
        if isinstance(node, ast.Attribute):
            if node.attr in ("_refresh_cached_snapshots", "_k_hx_by_group_w_k"):
                violations.append(
                    f"ha_bridge/coordinator.py:{node.lineno} accesses "
                    f"forbidden private attribute '{node.attr}'"
                )
            if node.attr == "core" and (
                (
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr == "accumulator"
                )
                or (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "accumulator"
                )
            ):
                violations.append(
                    f"ha_bridge/coordinator.py:{node.lineno} accesses "
                    f"forbidden 'accumulator.core'"
                )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == "_k_hx_by_group_w_k"
        ):
            violations.append(
                f"ha_bridge/coordinator.py:{node.lineno} references forbidden "
                f"private attribute string '_k_hx_by_group_w_k'"
            )

    # 2. sensor.py: no private attribute access on external objects (non-self).
    sensor_tree = _parse("sensor.py")
    for node in ast.walk(sensor_tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and not node.attr.startswith("__")
            and not (isinstance(node.value, ast.Name) and node.value.id == "self")
        ):
            violations.append(
                f"sensor.py:{node.lineno} accesses private attribute "
                f"'{node.attr}' on non-self object"
            )

    # 3. domain/evaluation.py: no private zoh_kalman helpers.
    forbidden_zoh_privates = {
        "_discretize_ab_cached",
        "_discretize_qd_basis_cached",
        "_tobit_equivalent_innovation",
        "_joseph_scalar_update",
    }
    eval_tree = _parse("domain/evaluation.py")
    for node in ast.walk(eval_tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in forbidden_zoh_privates
        ):
            violations.append(
                f"domain/evaluation.py:{node.lineno} accesses forbidden "
                f"private helper '{node.attr}'"
            )
        elif isinstance(node, ast.Name) and node.id in forbidden_zoh_privates:
            violations.append(
                f"domain/evaluation.py:{node.lineno} references forbidden "
                f"private helper '{node.id}'"
            )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in forbidden_zoh_privates:
                    violations.append(
                        f"domain/evaluation.py:{node.lineno} imports "
                        f"forbidden private helper '{alias.name}'"
                    )

    # 4. domain/engine.py: no `_last_meas_temp_c`.
    engine_tree = _parse("domain/engine.py")
    for node in ast.walk(engine_tree):
        if isinstance(node, ast.Attribute) and node.attr == "_last_meas_temp_c":
            violations.append(
                f"domain/engine.py:{node.lineno} accesses forbidden private "
                f"attribute '_last_meas_temp_c'"
            )
        elif isinstance(node, ast.Name) and node.id == "_last_meas_temp_c":
            violations.append(
                f"domain/engine.py:{node.lineno} references forbidden private "
                f"attribute '_last_meas_temp_c'"
            )

    # 5. house_config.py: zero imports from `domain/`.
    house_cfg_tree = _parse("house_config.py")
    for node in ast.walk(house_cfg_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "domain" in alias.name.split("."):
                    violations.append(
                        f"house_config.py:{node.lineno} imports forbidden "
                        f"domain module {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod_parts = set(node.module.split(".")) if node.module else set()
            if "domain" in mod_parts or any(
                alias.name.split(".")[0] == "domain" for alias in node.names
            ):
                violations.append(
                    f"house_config.py:{node.lineno} imports from forbidden "
                    f"domain module {node.module!r}"
                )

    assert violations == [], (
        "Forbidden cross-module private accesses detected:\n"
        + "\n".join(violations)
    )
