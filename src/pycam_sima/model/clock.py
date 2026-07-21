"""Minimal NO_LEAP clock with exact integer-second progression."""

from __future__ import annotations

from dataclasses import dataclass


_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(slots=True)
class NoLeapClock:
    year: int = 1
    month: int = 1
    day: int = 1
    seconds: int = 0
    nstep: int = 0
    dt_seconds: int = 1800

    def advance(self) -> None:
        self.seconds += self.dt_seconds
        self.nstep += 1
        while self.seconds >= 86400:
            self.seconds -= 86400
            self.day += 1
            if self.day > _MONTH_LENGTHS[self.month - 1]:
                self.day = 1
                self.month += 1
                if self.month > 12:
                    self.month = 1
                    self.year += 1

    @property
    def yyyymmdd(self) -> int:
        return self.year * 10000 + self.month * 100 + self.day

    @property
    def iso_stamp(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}-{self.seconds:05d}"
