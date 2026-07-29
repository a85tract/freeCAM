"""Calendar-aware model clock with exact integer-second progression."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigurationError


_COMMON_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_ALL_LEAP_MONTH_LENGTHS = (
    31,
    29,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
)
_CALENDAR_ALIASES = {
    "NOLEAP": "NO_LEAP",
    "365_DAY": "NO_LEAP",
    "STANDARD": "GREGORIAN",
    "366_DAY": "ALL_LEAP",
}


def normalize_calendar(value: str) -> str:
    name = str(value).strip().upper()
    return _CALENDAR_ALIASES.get(name, name)


def _is_leap_year(year: int, calendar: str) -> bool:
    if calendar == "ALL_LEAP":
        return True
    if calendar in {"NO_LEAP", "360_DAY"}:
        return False
    if calendar == "JULIAN":
        return year % 4 == 0
    if calendar in {"GREGORIAN", "PROLEPTIC_GREGORIAN"}:
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    raise ConfigurationError(f"unsupported calendar {calendar!r}")


def month_lengths(year: int, calendar: str) -> tuple[int, ...]:
    calendar = normalize_calendar(calendar)
    if calendar == "360_DAY":
        return (30,) * 12
    return (
        _ALL_LEAP_MONTH_LENGTHS
        if _is_leap_year(year, calendar)
        else _COMMON_MONTH_LENGTHS
    )


@dataclass(slots=True)
class ModelClock:
    year: int = 1
    month: int = 1
    day: int = 1
    seconds: int = 0
    nstep: int = 0
    dt_seconds: int = 1800
    calendar: str = "NO_LEAP"
    base_year: int | None = None
    base_month: int | None = None
    base_day: int | None = None
    base_seconds: int | None = None

    def __post_init__(self) -> None:
        self.calendar = normalize_calendar(self.calendar)
        if self.year < 1 or not 1 <= self.month <= 12:
            raise ConfigurationError("clock date is outside the supported range")
        if not 1 <= self.day <= month_lengths(self.year, self.calendar)[
            self.month - 1
        ]:
            raise ConfigurationError("clock day is invalid for its calendar")
        if not 0 <= self.seconds < 86400:
            raise ConfigurationError("clock seconds must be between 0 and 86399")
        if self.dt_seconds <= 0 or self.nstep < 0:
            raise ConfigurationError("clock timestep must be positive")
        if self.base_year is None:
            self.base_year = self.year
        if self.base_month is None:
            self.base_month = self.month
        if self.base_day is None:
            self.base_day = self.day
        if self.base_seconds is None:
            self.base_seconds = self.seconds

    def advance(self) -> None:
        self.seconds += self.dt_seconds
        self.nstep += 1
        while self.seconds >= 86400:
            self.seconds -= 86400
            self.day += 1
            lengths = month_lengths(self.year, self.calendar)
            if self.day > lengths[self.month - 1]:
                self.day = 1
                self.month += 1
                if self.month > 12:
                    self.month = 1
                    self.year += 1

    @property
    def yyyymmdd(self) -> int:
        return self.year * 10000 + self.month * 100 + self.day

    @property
    def base_yyyymmdd(self) -> int:
        assert self.base_year is not None
        assert self.base_month is not None
        assert self.base_day is not None
        return (
            self.base_year * 10000
            + self.base_month * 100
            + self.base_day
        )

    @property
    def netcdf_calendar(self) -> str:
        return {
            "NO_LEAP": "noleap",
            "GREGORIAN": "gregorian",
            "PROLEPTIC_GREGORIAN": "proleptic_gregorian",
            "JULIAN": "julian",
            "ALL_LEAP": "all_leap",
            "360_DAY": "360_day",
        }[self.calendar]

    @property
    def time_units(self) -> str:
        assert self.base_year is not None
        assert self.base_month is not None
        assert self.base_day is not None
        assert self.base_seconds is not None
        hour, remainder = divmod(self.base_seconds, 3600)
        minute, second = divmod(remainder, 60)
        return (
            f"days since {self.base_year:04d}-{self.base_month:02d}-"
            f"{self.base_day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        )

    @property
    def elapsed_seconds(self) -> int:
        return self.nstep * self.dt_seconds

    def fractional_calendar_day(self, offset_seconds: int = 0) -> float:
        """Return CAM's one-based day-of-year including fractional day."""

        year = self.year
        month = self.month
        day = self.day
        seconds = self.seconds + int(offset_seconds)
        while seconds >= 86400:
            seconds -= 86400
            day += 1
            lengths = month_lengths(year, self.calendar)
            if day > lengths[month - 1]:
                day = 1
                month += 1
                if month > 12:
                    month = 1
                    year += 1
        while seconds < 0:
            month -= 1
            if month < 1:
                month = 12
                year -= 1
            day = month_lengths(year, self.calendar)[month - 1]
            seconds += 86400
        day_of_year = sum(month_lengths(year, self.calendar)[: month - 1]) + day
        # ESMF_TimeGet(dayOfYear_r8=...) forms the fractional day from
        # elapsed seconds since the start of the year, then adds the one-based
        # day origin.  Keeping that operation order is scientifically
        # observable: after day one it can differ by one ULP from
        # ``day_of_year + seconds / 86400`` and therefore alter solar geometry.
        elapsed_since_year_start = (day_of_year - 1) * 86400 + seconds
        return 1.0 + float(elapsed_since_year_start) / 86400.0

    @property
    def iso_stamp(self) -> str:
        return (
            f"{self.year:04d}-{self.month:02d}-{self.day:02d}-"
            f"{self.seconds:05d}"
        )


# Backward-compatible import name used by existing checkpoints and public code.
NoLeapClock = ModelClock
