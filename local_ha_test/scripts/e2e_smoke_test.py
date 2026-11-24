"""End-to-end smoke & integration test suite against live Docker Home Assistant.

Exercises the real Home Assistant Core runtime (ConfigFlow, OptionsFlow,
18 registered entities, switch/button services, CSV/JSON export to `/share`,
redacted diagnostics, multi-scenario event ingestion, config-entry reload,
`.storage` persistence, and HA log inspection for errors or blocking calls).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
import subprocess
import sys
import time
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LOCAL_HA_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_LOCAL_HA_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_HA_DIR))

from custom_components.betonbatterij import const  # noqa: E402
from scripts import ha_client, mock_house  # noqa: E402

_LOGGER = logging.getLogger(__name__)

EXPECTED_SENSORS: tuple[str, ...] = (
    "sensor.woonkamer_geschatte_vloertemperatuur",
    "sensor.woonkamer_geschatte_pvc_oppervlaktetemperatuur",
    "sensor.woonkamer_ongeplande_warmtelast",
    "sensor.woonkamer_vloer_afgiftevermogen",
    "sensor.woonkamer_hindcast_6u",
    "sensor.kantoor_geschatte_vloertemperatuur",
    "sensor.kantoor_geschatte_pvc_oppervlaktetemperatuur",
    "sensor.kantoor_ongeplande_warmtelast",
    "sensor.kantoor_vloer_afgiftevermogen",
    "sensor.kantoor_hindcast_6u",
    "sensor.betonbatterij_fit_woonkamer",
    "sensor.betonbatterij_fit_kantoor",
    "sensor.betonbatterij_fit_warmtepomp_cop",
)

EXPECTED_SWITCHES: tuple[str, ...] = ("switch.betonbatterij_auto_kalibratie",)

EXPECTED_BUTTONS: tuple[str, ...] = (
    "button.betonbatterij_herbereken_parameters",
    "button.betonbatterij_rollback_parameters",
    "button.betonbatterij_reset_parameters",
    "button.betonbatterij_exporteer_csv",
)

ALL_EXPECTED_ENTITIES: tuple[str, ...] = (
    EXPECTED_SENSORS + EXPECTED_SWITCHES + EXPECTED_BUTTONS
)


class E2EAssertionError(AssertionError):
    """Raised when an end-to-end Home Assistant assertion fails."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise E2EAssertionError(message)


def _wait_for_entities(
    client: ha_client.LocalHomeAssistantClient,
    entity_ids: tuple[str, ...],
    *,
    timeout_s: float = 30.0,
) -> dict[str, dict[str, Any]]:
    """Wait until every entity in `entity_ids` is registered in HA."""
    deadline = time.monotonic() + timeout_s
    missing: list[str] = []
    while time.monotonic() < deadline:
        by_id = {
            str(item["entity_id"]): item for item in client.get_states()
        }
        missing = [eid for eid in entity_ids if eid not in by_id]
        if not missing:
            return {eid: by_id[eid] for eid in entity_ids}
        time.sleep(1.0)
    raise E2EAssertionError(
        f"Timed out waiting for BetonBatterij entities: {', '.join(missing)}"
    )


def _ensure_betonbatterij_entry(
    client: ha_client.LocalHomeAssistantClient,
) -> str:
    """Create the BetonBatterij config entry via ConfigFlow if not present."""
    existing = client.list_config_entries(domain=const.DOMAIN)
    if existing:
        entry_id = str(existing[0]["entry_id"])
        _LOGGER.info(
            "[STEP] Reusing existing BetonBatterij entry_id=%s", entry_id
        )
        return entry_id

    _LOGGER.info("[STEP] Running 1-click BetonBatterij ConfigFlow...")
    step1 = client.start_config_flow(const.DOMAIN)
    _assert(
        step1.get("type") == "form" and step1.get("step_id") == "user",
        f"Unexpected ConfigFlow step 1 response: {step1}",
    )
    flow_id = str(step1["flow_id"])
    step2 = client.submit_config_flow(flow_id, {})
    _assert(
        step2.get("type") == "create_entry",
        f"ConfigFlow failed to create entry: {step2}",
    )
    result = step2.get("result") or {}
    entry_id = str(result.get("entry_id") or "")
    _assert(bool(entry_id), f"Missing entry_id in ConfigFlow result: {step2}")
    _LOGGER.info("  PASS: Created config entry %s", entry_id)

    # Verify single_config_entry abort guard.
    dup = client.start_config_flow(const.DOMAIN)
    _assert(
        dup.get("type") == "abort",
        f"Expected duplicate ConfigFlow to abort, got: {dup}",
    )
    _LOGGER.info(
        "  PASS: Duplicate ConfigFlow aborted cleanly (%s)", dup.get("reason")
    )
    return entry_id


def _verify_entities(
    client: ha_client.LocalHomeAssistantClient,
) -> dict[str, dict[str, Any]]:
    _LOGGER.info(
        "[STEP] Verifying all %d BetonBatterij entities...",
        len(ALL_EXPECTED_ENTITIES),
    )
    states = _wait_for_entities(client, ALL_EXPECTED_ENTITIES, timeout_s=30.0)

    cop_state = states["sensor.betonbatterij_fit_warmtepomp_cop"]
    _assert(
        cop_state["attributes"].get("unit_of_measurement") == "COP",
        f"Unexpected COP sensor attributes: {cop_state}",
    )
    _assert(
        cop_state["state"] not in ("unavailable", "unknown"),
        f"COP sensor has invalid state: {cop_state}",
    )

    wk_floor = states["sensor.woonkamer_geschatte_vloertemperatuur"]
    _assert(
        wk_floor["attributes"].get("unit_of_measurement") == "°C",
        f"Unexpected WK floor sensor attributes: {wk_floor}",
    )
    _LOGGER.info(
        "  PASS: All %d entities online (live COP=%s, wk_floor=%s °C)",
        len(ALL_EXPECTED_ENTITIES),
        cop_state["state"],
        wk_floor["state"],
    )
    return states


def _verify_options_flow(
    client: ha_client.LocalHomeAssistantClient, entry_id: str
) -> None:
    _LOGGER.info("[STEP] Testing OptionsFlow (rotameter flow rates)...")
    init = client.start_options_flow(entry_id)
    _assert(
        init.get("type") == "form",
        f"Unexpected OptionsFlow init response: {init}",
    )
    flow_id = str(init["flow_id"])

    # Out-of-range value must be rejected by Voluptuous validation.
    invalid_payload = {
        opt_key: float(const.DEFAULT_FLOW_RATES_L_MIN[group])
        for group, opt_key in const.FLOW_OPTION_KEYS.items()
    }
    invalid_payload[const.OPT_FLOW_WK_L_MIN] = 99.0
    try:
        bad_resp = client.submit_options_flow(flow_id, invalid_payload)
        _assert(
            bad_resp.get("type") == "form" and bool(bad_resp.get("errors")),
            f"Expected validation error for flow=99.0, got: {bad_resp}",
        )
    except ha_client.HomeAssistantAPIError as err:
        # HA returns HTTP 400 with schema errors for out-of-range values.
        _assert(
            err.status_code == 400,
            f"Expected HTTP 400 on invalid OptionsFlow, got {err}",
        )
    _LOGGER.info("  PASS: Out-of-range flow rate (99.0 L/min) rejected.")

    # Submit valid update (wk: 5.0 -> 6.5 L/min).
    updated_payload = {
        opt_key: float(const.DEFAULT_FLOW_RATES_L_MIN[group])
        for group, opt_key in const.FLOW_OPTION_KEYS.items()
    }
    updated_payload[const.OPT_FLOW_WK_L_MIN] = 6.5
    ok_resp = client.submit_options_flow(flow_id, updated_payload)
    _assert(
        ok_resp.get("type") == "create_entry",
        f"Valid OptionsFlow submit failed: {ok_resp}",
    )

    time.sleep(1.0)
    diag = client.get_diagnostics(entry_id)
    data = diag.get("data", diag)
    flow_rates = data.get("flow_rates_l_min", {})
    _assert(
        abs(float(flow_rates.get("wk", 0.0)) - 6.5) < 1e-6,
        f"Updated wk flow rate not reflected in diagnostics: {flow_rates}",
    )

    # Restore default flow rates cleanly.
    restore_init = client.start_options_flow(entry_id)
    restore_payload = {
        opt_key: float(const.DEFAULT_FLOW_RATES_L_MIN[group])
        for group, opt_key in const.FLOW_OPTION_KEYS.items()
    }
    restore_resp = client.submit_options_flow(
        str(restore_init["flow_id"]), restore_payload
    )
    _assert(
        restore_resp.get("type") == "create_entry",
        f"Restoring default OptionsFlow failed: {restore_resp}",
    )
    _LOGGER.info("  PASS: OptionsFlow update & restore verified.")


def _verify_switch_and_buttons(
    client: ha_client.LocalHomeAssistantClient,
    share_dir: pathlib.Path,
) -> None:
    _LOGGER.info("[STEP] Testing auto-calibration switch & action buttons...")
    switch_id = "switch.betonbatterij_auto_kalibratie"
    client.call_service("switch", "turn_off", {"entity_id": switch_id})
    time.sleep(0.5)
    st_off = client.get_state(switch_id)
    _assert(
        st_off is not None and st_off.get("state") == "off",
        f"Switch failed to turn off: {st_off}",
    )

    client.call_service("switch", "turn_on", {"entity_id": switch_id})
    time.sleep(0.5)
    st_on = client.get_state(switch_id)
    _assert(
        st_on is not None and st_on.get("state") == "on",
        f"Switch failed to turn on: {st_on}",
    )
    _LOGGER.info("  PASS: Auto-kalibratie switch toggled off -> on.")

    # Export CSV/JSON to /share/betonbatterij_export (bind-mounted to ./share).
    export_dir = share_dir / "betonbatterij_export"
    csv_file = export_dir / "telemetry_5m.csv"
    json_file = export_dir / "fit_report.json"
    for old_file in (csv_file, json_file):
        if old_file.is_file():
            old_file.unlink()

    client.call_service(
        "button",
        "press",
        {"entity_id": "button.betonbatterij_exporteer_csv"},
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if csv_file.is_file() and json_file.is_file():
            break
        time.sleep(0.5)

    _assert(
        csv_file.is_file() and json_file.is_file(),
        f"Expected CSV/JSON exports in {export_dir}, files not created.",
    )
    with csv_file.open("r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh), [])
    _assert(
        "ts_utc" in header and "flags" in header,
        f"Unexpected CSV header: {header[:10]}",
    )
    report_data = json.loads(json_file.read_text(encoding="utf-8"))
    _assert(
        isinstance(report_data, dict) and bool(report_data),
        "fit_report.json is empty or invalid JSON.",
    )
    _LOGGER.info(
        "  PASS: Export button created valid telemetry_5m.csv (%d cols) & "
        "fit_report.json.",
        len(header),
    )

    # Safe rollback (no-op on empty history) & reset buttons.
    client.call_service(
        "button",
        "press",
        {"entity_id": "button.betonbatterij_rollback_parameters"},
    )
    client.call_service(
        "button",
        "press",
        {"entity_id": "button.betonbatterij_reset_parameters"},
    )
    time.sleep(1.0)
    _LOGGER.info("  PASS: Rollback and reset buttons executed cleanly.")


def _verify_diagnostics(
    client: ha_client.LocalHomeAssistantClient, entry_id: str
) -> None:
    _LOGGER.info("[STEP] Verifying redacted diagnostics payload...")
    diag = client.get_diagnostics(entry_id)
    data = diag.get("data", diag)
    for key in (
        "flow_rates_l_min",
        "clock_guard",
        "ringbuffer",
        "zones",
        "cop_params",
        "evaluation_14d",
    ):
        _assert(key in data, f"Missing '{key}' in diagnostics payload.")
    _assert(
        "wk" in data["zones"] and "kt" in data["zones"],
        f"Missing wk/kt in diagnostics zones: {data['zones'].keys()}",
    )
    _LOGGER.info(
        "  PASS: Diagnostics OK (ringbuffer rows=%s, local_config_loaded=%s).",
        data["ringbuffer"].get("row_count"),
        data.get("local_config_loaded"),
    )


def _exercise_scenarios(
    client: ha_client.LocalHomeAssistantClient, config_dir: pathlib.Path
) -> None:
    _LOGGER.info("[STEP] Cycling virtual house scenarios through event bus...")
    sequence = (
        "cv_heating",
        "dhw_tank",
        "defrost",
        "buffer_harvest",
        "ac_kantoor_heat",
        "solar_day",
        "summer_cooling",
        "sensor_stale",
        "cv_heating",
    )
    for name in sequence:
        mock_house.apply_scenario(client, name, config_dir=config_dir)
        time.sleep(0.4)
    _LOGGER.info("  PASS: All %d scenarios ingested without errors.", len(sequence))


def _verify_reload_and_storage(
    client: ha_client.LocalHomeAssistantClient,
    entry_id: str,
    config_dir: pathlib.Path,
) -> None:
    _LOGGER.info("[STEP] Reloading config entry & verifying .storage files...")
    client.reload_config_entry(entry_id)
    time.sleep(2.0)
    _wait_for_entities(client, ALL_EXPECTED_ENTITIES, timeout_s=30.0)

    storage_dir = config_dir / ".storage"
    state_file = storage_dir / const.STORAGE_KEY_STATE
    _assert(
        state_file.is_file(),
        f"Expected persisted state file {state_file} after reload/flush.",
    )
    _LOGGER.info("  PASS: Clean reload & persisted %s.", state_file.name)


def _setup_temp_local_json(
    client: ha_client.LocalHomeAssistantClient,
    entry_id: str,
    config_dir: pathlib.Path,
) -> bool:
    """Temporarily write a synthetic `betonbatterij_local.json` and verify load.

    Returns:
        `True` if a temporary file was created and should be cleaned up later,
        or `False` if a user-provided `betonbatterij_local.json` already existed.
    """
    local_json = config_dir / "betonbatterij_local.json"
    created_temp = False
    if not local_json.is_file():
        _LOGGER.info(
            "[STEP] Testing betonbatterij_local.json override loading..."
        )
        payload = {
            "entities": {
                "desk_kantoor_power_w": "sensor.bureau_kantoor_vermogen_test",
                "bg_manifold_supply_temp": "sensor.bg_verdeler_aanvoer_test",
                "bg_manifold_return_temp": "sensor.bg_verdeler_retour_test",
                "bg_manifold_return_offset_c": -0.15,
                "floor_surface_sensor_wk": "sensor.vloer_oppervlak_wk_test",
            }
        }
        local_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        created_temp = True
        client.reload_config_entry(entry_id)
        time.sleep(2.0)
        _wait_for_entities(client, ALL_EXPECTED_ENTITIES, timeout_s=30.0)
    else:
        _LOGGER.info(
            "[STEP] Existing betonbatterij_local.json detected; verifying it "
            "is loaded..."
        )

    diag = client.get_diagnostics(entry_id)
    data = diag.get("data", diag)
    _assert(
        bool(data.get("local_config_loaded")),
        f"Expected local_config_loaded=True in diagnostics: {data}",
    )
    _LOGGER.info("  PASS: local_config_loaded=True verified in diagnostics.")
    return created_temp


def _verify_real_5m_tick_and_csv(
    client: ha_client.LocalHomeAssistantClient,
    entry_id: str,
    config_dir: pathlib.Path,
    share_dir: pathlib.Path,
) -> tuple[int, int]:
    """Wait for 1 real 5-minute UTC tick and validate ringbuffer + CSV output.

    Returns:
        Tuple `(row_count, latest_ts_utc)` observed after the tick.
    """
    diag_before = client.get_diagnostics(entry_id).get("data", {})
    rb_before = diag_before.get("ringbuffer", {})
    before_ts = int(rb_before.get("latest_ts_utc") or 0)
    before_rows = int(rb_before.get("row_count") or 0)

    # Seed steady cv_heating (including optional sensors from local JSON).
    mock_house.apply_scenario(client, "cv_heating", config_dir=config_dir)
    remaining_s = 300 - (int(time.time()) % 300)
    timeout_s = remaining_s + 45.0
    _LOGGER.info(
        "[STEP] Waiting ~%ds for next real 5-minute UTC tick (before_rows=%d, "
        "before_ts=%d)...",
        remaining_s,
        before_rows,
        before_ts,
    )

    deadline = time.monotonic() + timeout_s
    post_rows = before_rows
    post_ts = before_ts
    while time.monotonic() < deadline:
        diag = client.get_diagnostics(entry_id).get("data", {})
        rb = diag.get("ringbuffer", {})
        post_rows = int(rb.get("row_count") or 0)
        post_ts = int(rb.get("latest_ts_utc") or 0)
        if post_ts > before_ts and post_rows >= 1:
            break
        time.sleep(3.0)

    _assert(
        post_ts > before_ts and post_rows >= 1,
        f"Timed out waiting for 5-minute tick (rows={post_rows}, ts={post_ts}).",
    )
    _LOGGER.info(
        "  PASS: 5-minute tick recorded (row_count=%d, latest_ts_utc=%d).",
        post_rows,
        post_ts,
    )

    # Export CSV and inspect the populated telemetry row.
    export_dir = share_dir / "betonbatterij_export"
    csv_file = export_dir / "telemetry_5m.csv"
    if csv_file.is_file():
        csv_file.unlink()

    client.call_service(
        "button",
        "press",
        {"entity_id": "button.betonbatterij_exporteer_csv"},
    )
    csv_deadline = time.monotonic() + 15.0
    while time.monotonic() < csv_deadline and not csv_file.is_file():
        time.sleep(0.5)
    _assert(csv_file.is_file(), f"Exported CSV missing at {csv_file}.")

    with csv_file.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    _assert(len(rows) >= 1, "Expected >= 1 data row in telemetry_5m.csv.")
    latest = rows[-1]

    ts_val = int(float(latest["ts_utc"]))
    flags_val = int(float(latest["flags"]))
    wind_ms = float(latest["v_wind_ms"])
    t_air_wk = float(latest["t_air_wk_c"])
    t_floor_wk = float(latest["t_floor_wk_c"])
    f_comp_hz = float(latest["f_comp_hz"])
    p_wp_w = float(latest["p_wp_raw_el_w"])

    _assert(ts_val == post_ts, f"CSV ts_utc {ts_val} != ringbuffer {post_ts}.")
    _assert(
        bool(flags_val & const.FLAG_VALID_WK),
        f"Expected FLAG_VALID_WK in flags={flags_val}.",
    )
    _assert(
        bool(flags_val & const.FLAG_PUMP_SEC_ON),
        f"Expected FLAG_PUMP_SEC_ON in flags={flags_val}.",
    )
    _assert(
        abs(wind_ms - 4.0) < 0.05,
        f"Expected wind_speed 14.4 km/h -> 4.0 m/s, got {wind_ms}.",
    )
    _assert(
        abs(t_air_wk - 20.4) < 0.15,
        f"Expected t_air_wk_c ~20.4 C, got {t_air_wk}.",
    )
    _assert(
        math.isfinite(t_floor_wk) and 15.0 < t_floor_wk < 35.0,
        f"Implausible t_floor_wk_c: {t_floor_wk}.",
    )
    _assert(
        abs(f_comp_hz - 24.0) < 1.0,
        f"Expected f_comp_hz ~24 Hz, got {f_comp_hz}.",
    )
    _assert(
        abs(p_wp_w - 800.0) < 50.0,
        f"Expected p_wp_raw_el_w ~800 W, got {p_wp_w}.",
    )
    _LOGGER.info(
        "  PASS: Populated CSV row verified (t_wk=%.2f C, t_floor=%.2f C, "
        "wind=%.2f m/s, comp=%.0f Hz, wp=%.0f W, flags=0x%x).",
        t_air_wk,
        t_floor_wk,
        wind_ms,
        f_comp_hz,
        p_wp_w,
        flags_val,
    )
    return post_rows, post_ts


def _verify_container_restart_persistence(
    client: ha_client.LocalHomeAssistantClient,
    entry_id: str,
    config_dir: pathlib.Path,
    *,
    expected_rows: int,
    expected_ts: int,
    compose_service: str,
) -> None:
    """Restart the Docker HA container and verify cold-boot state restoration."""
    compose_file = _LOCAL_HA_DIR / "docker-compose.yml"
    _LOGGER.info(
        "[STEP] Restarting Docker service '%s' to verify cold-boot storage "
        "restoration...",
        compose_service,
    )
    cmd = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
    ]
    if compose_service == "homeassistant-rpi4":
        cmd.extend(["--profile", "rpi4"])
    cmd.extend(["restart", compose_service])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    _assert(
        result.returncode == 0,
        f"Docker restart failed (rc={result.returncode}): {result.stderr}",
    )

    storage_dir = config_dir / ".storage"
    ring_chunks = list(storage_dir.glob(f"{const.STORAGE_KEY_RING_PREFIX}*"))
    _assert(
        bool(ring_chunks),
        f"Expected persisted weekly ringbuffer chunk in {storage_dir}.",
    )

    client.wait_until_ready(timeout_s=120.0)
    client.ensure_authenticated(auto_onboard=False)
    _wait_for_entities(client, ALL_EXPECTED_ENTITIES, timeout_s=45.0)

    diag_after = client.get_diagnostics(entry_id).get("data", {})
    rb_after = diag_after.get("ringbuffer", {})
    restored_rows = int(rb_after.get("row_count") or 0)
    restored_ts = int(rb_after.get("latest_ts_utc") or 0)
    _assert(
        restored_rows >= expected_rows and restored_ts >= expected_ts,
        f"Ringbuffer lost data across restart: before=({expected_rows}, "
        f"{expected_ts}), after=({restored_rows}, {restored_ts}).",
    )

    sw_state = client.get_state("switch.betonbatterij_auto_kalibratie")
    _assert(
        sw_state is not None and sw_state.get("state") == "on",
        f"Auto-kalibratie switch did not restore 'on': {sw_state}",
    )
    _LOGGER.info(
        "  PASS: Cold-boot restart restored %d ringbuffer rows (ts=%d, "
        "%d chunk file(s)) and switch state.",
        restored_rows,
        restored_ts,
        len(ring_chunks),
    )


def _check_ha_logs(
    client: ha_client.LocalHomeAssistantClient, config_dir: pathlib.Path
) -> None:
    _LOGGER.info("[STEP] Scanning Home Assistant logs for BetonBatterij issues...")
    log_text = client.get_error_log()
    log_file = config_dir / "home-assistant.log"
    if log_file.is_file():
        log_text = f"{log_text}\n{log_file.read_text(encoding='utf-8', errors='replace')}"

    suspicious: list[str] = []
    for line in log_text.splitlines():
        lower = line.lower()
        if "betonbatterij" not in lower:
            continue
        if any(
            marker in lower
            for marker in (
                " error ",
                "traceback",
                "detected blocking call",
                "exception",
                "deprecated",
            )
        ):
            suspicious.append(line.strip())

    _assert(
        not suspicious,
        "Found BetonBatterij errors/warnings in HA logs:\n"
        + "\n".join(suspicious[:20]),
    )
    _LOGGER.info("  PASS: Zero BetonBatterij errors, tracebacks, or blocking calls.")


def run_smoke_suite(
    base_url: str,
    *,
    config_dir: pathlib.Path,
    share_dir: pathlib.Path,
    full: bool = False,
    skip_restart: bool = False,
    compose_service: str = "homeassistant",
) -> None:
    """Run the end-to-end Docker Home Assistant test suite."""
    client = ha_client.LocalHomeAssistantClient(base_url)
    _LOGGER.info("[STEP] Waiting for Home Assistant at %s...", base_url)
    client.wait_until_ready(timeout_s=120.0)
    client.ensure_authenticated(auto_onboard=True)

    _LOGGER.info("[STEP] Seeding baseline virtual house (cv_heating)...")
    mock_house.apply_scenario(client, "cv_heating", config_dir=config_dir)

    entry_id = _ensure_betonbatterij_entry(client)
    _verify_entities(client)
    _verify_options_flow(client, entry_id)
    _verify_switch_and_buttons(client, share_dir)
    _verify_diagnostics(client, entry_id)
    _exercise_scenarios(client, config_dir)
    _verify_reload_and_storage(client, entry_id, config_dir)

    if full:
        created_temp_local = False
        try:
            created_temp_local = _setup_temp_local_json(
                client, entry_id, config_dir
            )
            post_rows, post_ts = _verify_real_5m_tick_and_csv(
                client, entry_id, config_dir, share_dir
            )
            if not skip_restart:
                _verify_container_restart_persistence(
                    client,
                    entry_id,
                    config_dir,
                    expected_rows=post_rows,
                    expected_ts=post_ts,
                    compose_service=compose_service,
                )
        finally:
            if created_temp_local:
                temp_file = config_dir / "betonbatterij_local.json"
                if temp_file.is_file():
                    temp_file.unlink()
                try:
                    client.wait_until_ready(timeout_s=30.0)
                    client.ensure_authenticated(auto_onboard=False)
                    client.reload_config_entry(entry_id)
                    time.sleep(1.5)
                except ha_client.HomeAssistantAPIError as cleanup_err:
                    _LOGGER.warning(
                        "Post-test reload after removing temp local JSON "
                        "failed: %s",
                        cleanup_err,
                    )

    _check_ha_logs(client, config_dir)

    mode = "FULL E2E" if full else "E2E SMOKE"
    _LOGGER.info("ALL %s CHECKS PASSED!", mode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BetonBatterij E2E smoke tests against local Docker HA."
    )
    parser.add_argument(
        "--url",
        default=ha_client.DEFAULT_BASE_URL,
        help="Local Home Assistant base URL (default: http://127.0.0.1:8123).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Also test betonbatterij_local.json loading, wait for 1 real "
            "5-minute UTC tick + populated CSV validation, and verify "
            "cold-boot ringbuffer persistence across a Docker container restart."
        ),
    )
    parser.add_argument(
        "--skip-restart",
        action="store_true",
        help="In --full mode, skip the Docker container restart check.",
    )
    parser.add_argument(
        "--compose-service",
        default="homeassistant",
        choices=("homeassistant", "homeassistant-rpi4"),
        help="Docker Compose service name to restart in --full mode.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args()
    try:
        run_smoke_suite(
            args.url,
            config_dir=_LOCAL_HA_DIR / "config",
            share_dir=_LOCAL_HA_DIR / "share",
            full=args.full,
            skip_restart=args.skip_restart,
            compose_service=args.compose_service,
        )
    except (E2EAssertionError, ha_client.HomeAssistantAPIError) as err:
        _LOGGER.error("E2E TEST FAILED: %s", err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
