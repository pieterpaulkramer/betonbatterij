"""Zero-dependency Home Assistant REST API client & automatic onboarding helper.

Designed for local Docker Home Assistant integration testing of BetonBatterij.
Uses only the Python standard library (`urllib.request`) so it runs directly on
Windows without installing extra packages.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

_LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL: str = "http://127.0.0.1:8123"
DEFAULT_DEV_USER: str = "dev"
DEFAULT_DEV_PASSWORD: str = "betonbatterij-dev-123"
DEFAULT_DEV_NAME: str = "BetonBatterij Dev"


class HomeAssistantAPIError(RuntimeError):
    """Raised when a Home Assistant HTTP API request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str = "",
    ) -> None:
        """Initialize the API error with optional HTTP status and body."""
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class LocalHomeAssistantClient:
    """HTTP client for a local Docker Home Assistant test instance.

    Authentication resolution order:
      1. Explicit `token` argument or `HA_TOKEN` environment variable.
      2. Long-lived token stored in `<state_dir>/.ha_token`.
      3. Cached OAuth `refresh_token` in `<state_dir>/.ha_auth.json` (auto-
         refreshed whenever expired).
      4. Automatic first-boot onboarding via `/api/onboarding/*` when
         `ensure_authenticated(auto_onboard=True)` is called on a fresh container.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        token: str | None = None,
        state_dir: pathlib.Path | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Initialize the Home Assistant REST client."""
        self.base_url = base_url.rstrip("/")
        self.client_id = f"{self.base_url}/"
        self.timeout_s = float(timeout_s)
        self.state_dir = (
            state_dir
            if state_dir is not None
            else pathlib.Path(__file__).resolve().parent.parent
        )
        self._auth_cache_path = self.state_dir / ".ha_auth.json"
        self._llat_path = self.state_dir / ".ha_token"
        self._access_token: str | None = token or os.environ.get("HA_TOKEN")
        self._refresh_token: str | None = None
        self._token_expires_at: float = 0.0
        self._is_long_lived: bool = bool(self._access_token)

    def _raw_request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        form_body: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        use_auth: bool = True,
        expect_json: bool = True,
    ) -> Any:
        """Execute one HTTP request against Home Assistant."""
        url = f"{self.base_url}{path}"
        req_headers: dict[str, str] = {}
        if headers:
            req_headers.update(headers)
        if use_auth:
            token = self._get_valid_access_token()
            req_headers["Authorization"] = f"Bearer {token}"

        data: bytes | None = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        elif form_body is not None:
            data = urllib.parse.urlencode(form_body).encode("utf-8")
            req_headers["Content-Type"] = "application/x-www-form-urlencoded"

        request = urllib.request.Request(
            url, data=data, headers=req_headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_s
            ) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as err:
            body_text = err.read().decode("utf-8", errors="replace")
            raise HomeAssistantAPIError(
                f"{method} {path} failed with HTTP {err.code}: {body_text}",
                status_code=err.code,
                body=body_text,
            ) from err
        except urllib.error.URLError as err:
            raise HomeAssistantAPIError(
                f"{method} {path} failed to connect: {err.reason}"
            ) from err
        except (http.client.HTTPException, OSError) as err:
            raise HomeAssistantAPIError(
                f"{method} {path} connection interrupted: {err}"
            ) from err

        if not raw_bytes:
            return None
        text = raw_bytes.decode("utf-8", errors="replace")
        if not expect_json:
            return text
        return json.loads(text)

    def _load_cached_tokens(self) -> bool:
        """Load a saved LLAT or OAuth refresh token from disk."""
        if self._access_token and self._is_long_lived:
            return True

        if self._llat_path.is_file():
            saved_llat = self._llat_path.read_text(encoding="utf-8").strip()
            if saved_llat:
                self._access_token = saved_llat
                self._is_long_lived = True
                return True

        if self._auth_cache_path.is_file():
            try:
                cached = json.loads(
                    self._auth_cache_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as err:
                _LOGGER.warning(
                    "Ignoring corrupt auth cache %s: %s",
                    self._auth_cache_path,
                    err,
                )
                return False
            refresh_token = cached.get("refresh_token")
            client_id = cached.get("client_id")
            if isinstance(refresh_token, str) and refresh_token:
                self._refresh_token = refresh_token
                if isinstance(client_id, str) and client_id:
                    self.client_id = client_id
                self._access_token = cached.get("access_token")
                self._token_expires_at = float(cached.get("expires_at", 0.0))
                self._is_long_lived = False
                return True
        return False

    def _save_oauth_cache(self) -> None:
        """Persist OAuth refresh/access tokens locally for reuse across runs."""
        payload = {
            "base_url": self.base_url,
            "client_id": self.client_id,
            "refresh_token": self._refresh_token,
            "access_token": self._access_token,
            "expires_at": self._token_expires_at,
        }
        self._auth_cache_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _refresh_access_token(self) -> str:
        """Exchange the cached refresh token for a fresh OAuth access token."""
        if not self._refresh_token:
            raise HomeAssistantAPIError(
                "No refresh token available. Run automatic onboarding or put a "
                "Long-Lived Access Token in local_ha_test/.ha_token."
            )
        token_resp = self._raw_request(
            "POST",
            "/auth/token",
            form_body={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self.client_id,
            },
            use_auth=False,
        )
        self._access_token = str(token_resp["access_token"])
        expires_in = float(token_resp.get("expires_in", 1800))
        self._token_expires_at = time.time() + max(60.0, expires_in - 60.0)
        self._save_oauth_cache()
        return self._access_token

    def _get_valid_access_token(self) -> str:
        """Return a valid bearer token, refreshing via OAuth if needed."""
        if self._access_token and self._is_long_lived:
            return self._access_token
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token
        if self._refresh_token:
            return self._refresh_access_token()
        raise HomeAssistantAPIError(
            "Not authenticated. Call ensure_authenticated() first."
        )

    def wait_until_ready(
        self,
        *,
        timeout_s: float = 120.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        """Poll Home Assistant until its HTTP server responds."""
        deadline = time.monotonic() + timeout_s
        last_err: str = "no response yet"
        while time.monotonic() < deadline:
            try:
                self._raw_request(
                    "GET",
                    "/manifest.json",
                    use_auth=False,
                    expect_json=False,
                )
                return
            except HomeAssistantAPIError as err:
                last_err = str(err)
                time.sleep(poll_interval_s)
        raise HomeAssistantAPIError(
            f"Home Assistant at {self.base_url} did not become ready within "
            f"{timeout_s:.0f}s (last error: {last_err})"
        )

    def get_onboarding_steps(self) -> list[dict[str, Any]]:
        """Return HA onboarding steps, or an empty list if already completed."""
        try:
            steps = self._raw_request("GET", "/api/onboarding", use_auth=False)
        except HomeAssistantAPIError as err:
            if err.status_code == 404:
                return []
            raise
        if isinstance(steps, list):
            return steps
        return []

    def ensure_authenticated(
        self,
        *,
        auto_onboard: bool = True,
        username: str = DEFAULT_DEV_USER,
        password: str = DEFAULT_DEV_PASSWORD,
        name: str = DEFAULT_DEV_NAME,
    ) -> None:
        """Authenticate against local HA, auto-onboarding a fresh container."""
        self._load_cached_tokens()

        if self._access_token or self._refresh_token:
            try:
                self.api_get("/api/")
                return
            except HomeAssistantAPIError as err:
                if err.status_code != 401:
                    raise
                _LOGGER.info(
                    "Cached token rejected (HTTP 401); checking onboarding "
                    "status."
                )
                self._access_token = None
                self._refresh_token = None

        steps = self.get_onboarding_steps()
        step_done = {
            str(item.get("step")): bool(item.get("done")) for item in steps
        }

        if not steps or all(step_done.values()):
            raise HomeAssistantAPIError(
                "Home Assistant is already onboarded, but no valid token was "
                "found. Either set HA_TOKEN / write a Long-Lived Access Token "
                f"to {self._llat_path}, or run `run_test.ps1 reset` for a "
                "fresh auto-onboarded test container."
            )

        if not auto_onboard:
            raise HomeAssistantAPIError(
                "Home Assistant requires onboarding and auto_onboard=False."
            )

        if not step_done.get("user", False):
            _LOGGER.info(
                "Auto-onboarding fresh HA instance (user=%s)...", username
            )
            user_resp = self._raw_request(
                "POST",
                "/api/onboarding/users",
                json_body={
                    "client_id": self.client_id,
                    "name": name,
                    "username": username,
                    "password": password,
                    "language": "nl",
                },
                use_auth=False,
            )
            auth_code = str(user_resp["auth_code"])
            token_resp = self._raw_request(
                "POST",
                "/auth/token",
                form_body={
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "client_id": self.client_id,
                },
                use_auth=False,
            )
            self._access_token = str(token_resp["access_token"])
            self._refresh_token = str(token_resp["refresh_token"])
            expires_in = float(token_resp.get("expires_in", 1800))
            self._token_expires_at = time.time() + max(60.0, expires_in - 60.0)
            self._is_long_lived = False
            self._save_oauth_cache()

        if not step_done.get("core_config", False):
            self.api_post("/api/onboarding/core_config", {})
        if not step_done.get("analytics", False):
            self.api_post("/api/onboarding/analytics", {})
        if not step_done.get("integration", False):
            self.api_post(
                "/api/onboarding/integration",
                {
                    "client_id": self.client_id,
                    "redirect_uri": f"{self.base_url}/?auth_callback=1",
                },
            )

        self.api_get("/api/")
        _LOGGER.info(
            "Local HA onboarding & authentication complete (cached at %s).",
            self._auth_cache_path,
        )

    def api_get(self, path: str, *, expect_json: bool = True) -> Any:
        """Perform an authenticated GET request with 401 token refresh retry."""
        try:
            return self._raw_request(
                "GET", path, use_auth=True, expect_json=expect_json
            )
        except HomeAssistantAPIError as err:
            if err.status_code == 401 and self._refresh_token:
                self._refresh_access_token()
                return self._raw_request(
                    "GET", path, use_auth=True, expect_json=expect_json
                )
            raise

    def api_post(
        self,
        path: str,
        payload: Any | None = None,
        *,
        expect_json: bool = True,
    ) -> Any:
        """Perform an authenticated POST request with 401 token refresh retry."""
        try:
            return self._raw_request(
                "POST",
                path,
                json_body=payload,
                use_auth=True,
                expect_json=expect_json,
            )
        except HomeAssistantAPIError as err:
            if err.status_code == 401 and self._refresh_token:
                self._refresh_access_token()
                return self._raw_request(
                    "POST",
                    path,
                    json_body=payload,
                    use_auth=True,
                    expect_json=expect_json,
                )
            raise

    def api_delete(self, path: str) -> Any:
        """Perform an authenticated DELETE request."""
        try:
            return self._raw_request("DELETE", path, use_auth=True)
        except HomeAssistantAPIError as err:
            if err.status_code == 401 and self._refresh_token:
                self._refresh_access_token()
                return self._raw_request("DELETE", path, use_auth=True)
            raise

    def get_states(self) -> list[dict[str, Any]]:
        """Fetch all Home Assistant entity states."""
        return list(self.api_get("/api/states"))

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        """Fetch a single entity state, returning None on HTTP 404."""
        try:
            return dict(self.api_get(f"/api/states/{entity_id}"))
        except HomeAssistantAPIError as err:
            if err.status_code == 404:
                return None
            raise

    def set_state(
        self,
        entity_id: str,
        state: str | float,
        attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Set or create a state in Home Assistant's state machine."""
        body: dict[str, Any] = {"state": str(state)}
        if attributes is not None:
            body["attributes"] = dict(attributes)
        return dict(self.api_post(f"/api/states/{entity_id}", body))

    def call_service(
        self,
        domain: str,
        service: str,
        service_data: Mapping[str, Any] | None = None,
    ) -> Any:
        """Invoke a Home Assistant service."""
        return self.api_post(
            f"/api/services/{domain}/{service}",
            dict(service_data) if service_data is not None else {},
        )

    def list_config_entries(
        self, domain: str | None = None
    ) -> list[dict[str, Any]]:
        """List config entries, optionally filtered by integration domain."""
        entries = list(self.api_get("/api/config/config_entries/entry"))
        if domain is None:
            return entries
        return [e for e in entries if e.get("domain") == domain]

    def start_config_flow(
        self, handler: str, *, show_advanced_options: bool = False
    ) -> dict[str, Any]:
        """Start a ConfigFlow for an integration domain."""
        return dict(
            self.api_post(
                "/api/config/config_entries/flow",
                {
                    "handler": handler,
                    "show_advanced_options": show_advanced_options,
                },
            )
        )

    def submit_config_flow(
        self, flow_id: str, user_input: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Submit form input to an active ConfigFlow."""
        return dict(
            self.api_post(
                f"/api/config/config_entries/flow/{flow_id}",
                dict(user_input),
            )
        )

    def start_options_flow(self, entry_id: str) -> dict[str, Any]:
        """Start an OptionsFlow for an existing config entry."""
        return dict(
            self.api_post(
                "/api/config/config_entries/options/flow",
                {"handler": entry_id},
            )
        )

    def submit_options_flow(
        self, flow_id: str, user_input: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Submit form input to an active OptionsFlow."""
        return dict(
            self.api_post(
                f"/api/config/config_entries/options/flow/{flow_id}",
                dict(user_input),
            )
        )

    def reload_config_entry(self, entry_id: str) -> dict[str, Any]:
        """Reload a config entry in place."""
        return dict(
            self.api_post(
                f"/api/config/config_entries/entry/{entry_id}/reload", {}
            )
        )

    def delete_config_entry(self, entry_id: str) -> dict[str, Any]:
        """Remove a config entry."""
        return dict(
            self.api_delete(f"/api/config/config_entries/entry/{entry_id}")
        )

    def get_diagnostics(self, entry_id: str) -> dict[str, Any]:
        """Fetch redacted integration diagnostics JSON for a config entry."""
        return dict(self.api_get(f"/api/diagnostics/config_entry/{entry_id}"))

    def get_error_log(self) -> str:
        """Fetch the current Home Assistant log text via `/api/error_log`."""
        return str(self.api_get("/api/error_log", expect_json=False))
