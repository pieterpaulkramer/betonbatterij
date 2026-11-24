"""End-to-end pytest checks against a live Docker Home Assistant Core instance.

Excluded by default from `python -m pytest` (see `pyproject.toml`) so offline
unit tests always run in seconds without Docker. Run explicitly with:

  - Fast smoke suite (~15s):
      `python -m pytest -m e2e -v`
  - Full suite incl. 5-minute UTC tick, CSV telemetry & container restart:
      `python -m pytest -m "e2e or e2e_full" -v`
"""

from __future__ import annotations

import pathlib
import sys
import time

import pytest

_LOCAL_HA_DIR = pathlib.Path(__file__).resolve().parents[2] / "local_ha_test"
if str(_LOCAL_HA_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_HA_DIR))

from scripts import e2e_smoke_test, ha_client  # noqa: E402
from tests.e2e import conftest  # noqa: E402


@pytest.mark.e2e
def test_all_18_entities_registered_and_valid(
    ha_client_session: ha_client.LocalHomeAssistantClient,
    betonbatterij_entry_id: str,
) -> None:
    """All 13 sensors, 1 switch, and 4 buttons register with valid units."""
    assert betonbatterij_entry_id
    e2e_smoke_test._verify_entities(ha_client_session)


@pytest.mark.e2e
def test_options_flow_validation_and_update(
    ha_client_session: ha_client.LocalHomeAssistantClient,
    betonbatterij_entry_id: str,
) -> None:
    """OptionsFlow rejects out-of-range flow rates and persists valid edits."""
    e2e_smoke_test._verify_options_flow(
        ha_client_session, betonbatterij_entry_id
    )


@pytest.mark.e2e
def test_auto_calibrate_switch_and_action_buttons(
    ha_client_session: ha_client.LocalHomeAssistantClient,
    betonbatterij_entry_id: str,
    ha_env: conftest.DockerHAEnv,
) -> None:
    """Auto-calibration switch and CSV export/rollback/reset buttons work."""
    del betonbatterij_entry_id  # Ensures integration is loaded first.
    e2e_smoke_test._verify_switch_and_buttons(
        ha_client_session, ha_env.share_dir
    )


@pytest.mark.e2e
def test_diagnostics_payload_structure(
    ha_client_session: ha_client.LocalHomeAssistantClient,
    betonbatterij_entry_id: str,
) -> None:
    """Diagnostics endpoint returns complete, redacted integration state."""
    e2e_smoke_test._verify_diagnostics(
        ha_client_session, betonbatterij_entry_id
    )


@pytest.mark.e2e
def test_all_virtual_house_scenarios_ingested(
    ha_client_session: ha_client.LocalHomeAssistantClient,
    betonbatterij_entry_id: str,
    ha_env: conftest.DockerHAEnv,
) -> None:
    """All 9 physical house scenarios flow through HA's event bus cleanly."""
    del betonbatterij_entry_id
    e2e_smoke_test._exercise_scenarios(ha_client_session, ha_env.config_dir)


@pytest.mark.e2e
def test_config_entry_reload_and_storage_persistence(
    ha_client_session: ha_client.LocalHomeAssistantClient,
    betonbatterij_entry_id: str,
    ha_env: conftest.DockerHAEnv,
) -> None:
    """In-place reload flushes `.storage/betonbatterij.state` and recovers."""
    e2e_smoke_test._verify_reload_and_storage(
        ha_client_session, betonbatterij_entry_id, ha_env.config_dir
    )


@pytest.mark.e2e
def test_zero_errors_or_blocking_calls_in_ha_logs(
    ha_client_session: ha_client.LocalHomeAssistantClient,
    betonbatterij_entry_id: str,
    ha_env: conftest.DockerHAEnv,
) -> None:
    """HA logs contain zero BetonBatterij errors, tracebacks, or blocking calls."""
    del betonbatterij_entry_id
    e2e_smoke_test._check_ha_logs(ha_client_session, ha_env.config_dir)


@pytest.mark.e2e_full
def test_full_local_json_real_5m_tick_csv_and_restart_persistence(
    ha_client_session: ha_client.LocalHomeAssistantClient,
    betonbatterij_entry_id: str,
    ha_env: conftest.DockerHAEnv,
) -> None:
    """Verify local JSON override, 1 real 5m tick + CSV row, and cold restart."""
    created_temp_local = False
    try:
        created_temp_local = e2e_smoke_test._setup_temp_local_json(
            ha_client_session, betonbatterij_entry_id, ha_env.config_dir
        )
        post_rows, post_ts = e2e_smoke_test._verify_real_5m_tick_and_csv(
            ha_client_session,
            betonbatterij_entry_id,
            ha_env.config_dir,
            ha_env.share_dir,
        )
        if not ha_env.skip_restart:
            e2e_smoke_test._verify_container_restart_persistence(
                ha_client_session,
                betonbatterij_entry_id,
                ha_env.config_dir,
                expected_rows=post_rows,
                expected_ts=post_ts,
                compose_service=ha_env.compose_service,
            )
    finally:
        if created_temp_local:
            temp_file = ha_env.config_dir / "betonbatterij_local.json"
            if temp_file.is_file():
                temp_file.unlink()
            ha_client_session.wait_until_ready(timeout_s=30.0)
            ha_client_session.ensure_authenticated(auto_onboard=False)
            ha_client_session.reload_config_entry(betonbatterij_entry_id)
            time.sleep(1.5)

    e2e_smoke_test._check_ha_logs(ha_client_session, ha_env.config_dir)
