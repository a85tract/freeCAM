from dataclasses import dataclass


@dataclass
class ModelClock:
    dt_seconds: int
    step: int = 0
    elapsed_seconds: int = 0

    def advance(self) -> None:
        self.step += 1
        self.elapsed_seconds += self.dt_seconds
