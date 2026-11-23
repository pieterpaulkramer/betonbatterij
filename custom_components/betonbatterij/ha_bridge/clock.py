"""Wall-clock jump guard for RTC-less Raspberry Pi deployments.

Pure Python module with zero `homeassistant.*` imports. Protects the 5-minute
monotonic tick pipeline against pre-NTP boot epochs and backwards wall-clock
jumps while computing elapsed durations strictly from `time.monotonic()`.
"""

from __future__ import annotations

from ..const import DT_STEP_S

MIN_PLAUSIBLE_UTC_EPOCH: int = 1_704_067_200


class ClockJumpGuard:
    """Guard against pre-NTP epochs and NTP wall-clock jumps on RTC-less RPi 4.

    Attributes:
        last_accepted_utc: Highest accepted UTC epoch timestamp [s].
        min_valid_utc: Minimum plausible UTC epoch floor [s].
        last_mono: Most recent `time.monotonic()` reading [s], or `None`.
        ntp_synced: Whether at least one valid monotonic UTC tick was accepted.
        rejected_ticks: Count of skipped pre-NTP or non-monotonic ticks.
    """

    __slots__ = (
        "last_accepted_utc",
        "last_mono",
        "min_valid_utc",
        "ntp_synced",
        "rejected_ticks",
    )

    def __init__(
        self,
        saved_wall_utc: int = 0,
        t_mono_start: float | None = None,
        min_epoch_utc: int = MIN_PLAUSIBLE_UTC_EPOCH,
    ) -> None:
        """Initialize the wall-clock jump guard."""
        self.last_accepted_utc: int = max(0, int(saved_wall_utc))
        self.min_valid_utc: int = max(
            int(min_epoch_utc), self.last_accepted_utc
        )
        self.last_mono: float | None = (
            float(t_mono_start) if t_mono_start is not None else None
        )
        self.ntp_synced: bool = False
        self.rejected_ticks: int = 0

    def reset_monotonic(self, t_mono: float) -> None:
        """Anchor the monotonic clock reference."""
        self.last_mono = float(t_mono)

    def record_bridged_utc(self, bridged_utc: int, t_mono: float) -> None:
        """Advance accepted UTC floor after restart-gap bridging or backfill."""
        if (
            bridged_utc > self.last_accepted_utc
            and bridged_utc >= self.min_valid_utc
        ):
            self.last_accepted_utc = int(bridged_utc)
            self.min_valid_utc = max(
                self.min_valid_utc, self.last_accepted_utc
            )
            self.ntp_synced = True
        self.last_mono = float(t_mono)

    def check_and_advance(
        self, wall_utc: float | int, t_mono: float
    ) -> tuple[bool, int, float]:
        """Validate `wall_utc` and compute elapsed step duration from `t_mono`.

        Args:
            wall_utc: Candidate UTC epoch timestamp at tick boundary.
            t_mono: Current `time.monotonic()` reading.

        Returns:
            A tuple `(accepted, ts_utc, dt_mono_s)`.
        """
        if self.last_mono is None:
            dt_mono = float(DT_STEP_S)
        else:
            dt_mono = max(0.0, float(t_mono) - self.last_mono)
        self.last_mono = float(t_mono)

        ts_int = int(round(float(wall_utc)))
        if ts_int < self.min_valid_utc or ts_int <= self.last_accepted_utc:
            self.rejected_ticks += 1
            return False, ts_int, dt_mono

        self.ntp_synced = True
        self.last_accepted_utc = ts_int
        return True, ts_int, dt_mono
