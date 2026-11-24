"""Shared pytest fixtures for Docker Home Assistant end-to-end tests.

Environment overrides (all optional):
  - `HA_BASE_URL`: Base URL of the local Docker HA instance
    (default: `http://127.0.0.1:8123`).
  - `HA_CONFIG_DIR`: Path to the bind-mounted HA `/config` directory
    (default: `local_ha_test/config`).
  - `HA_SHARE_DIR`: Path to the bind-mounted HA `/share` directory
    (default: `local_ha_test/share`).
  - `HA_COMPOSE_SERVICE`: Docker Compose service name for restart checks
    (default: `homeassistant`).
  - `HA_SKIP_RESTART`: Set to `1` to skip the Docker container restart check in
    `e2e_full` tests.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LOCAL_HA_DIR = _REPO_ROOT / "local_ha_test"
if str(_LOCAL_HA_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_HA_DIR))

from scripts import e2e_smoke_test, ha_client, mock_house  # noqa: E402


@dataclasses.dataclass(frozen=True)
class DockerHAEnv:
    """Resolved paths and connection settings for the local Docker HA instance."""

    base_url: str
    config_dir: pathlib.Path
    share_dir: pathlib.Path
    compose_service: str
    skip_restart: bool


@pytest.fixture(scope="session")
def ha_env() -> DockerHAEnv:
    """Return resolved paths and URL for the running Docker HA test target."""
    return DockerHAEnv(
        base_url=os.environ.get("HA_BASE_URL", ha_client.DEFAULT_BASE_URL),
        config_dir=pathlib.Path(
            os.environ.get("HA_CONFIG_DIR", str(_LOCAL_HA_DIR / "config"))
        ),
        share_dir=pathlib.Path(
            os.environ.get("HA_SHARE_DIR", str(_LOCAL_HA_DIR / "share"))
        ),
        compose_service=os.environ.get("HA_COMPOSE_SERVICE", "homeassistant"),
        skip_restart=os.environ.get("HA_SKIP_RESTART", "").strip().lower()
        in ("1", "true", "yes"),
    )


@pytest.fixture(scope="session")
def ha_client_session(ha_env: DockerHAEnv) -> ha_client.LocalHomeAssistantClient:
    """Connect to local Docker HA, auto-onboard if fresh, and seed baseline."""
    client = ha_client.LocalHomeAssistantClient(ha_env.base_url)
    client.wait_until_ready(timeout_s=120.0)
    client.ensure_authenticated(auto_onboard=True)
    mock_house.apply_scenario(
        client, "cv_heating", config_dir=ha_env.config_dir
    )
    return client


@pytest.fixture(scope="session")
def betonbatterij_entry_id(
    ha_client_session: ha_client.LocalHomeAssistantClient,
) -> str:
    """Ensure the BetonBatterij config entry exists (testing 1-click ConfigFlow)."""
    return e2e_smoke_test._ensure_betonbatterij_entry(ha_client_session)
