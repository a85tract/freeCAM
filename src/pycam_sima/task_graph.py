from __future__ import annotations

from collections.abc import Callable, Iterable

from taskflow import engines, task
from taskflow.patterns import linear_flow


class CallTask(task.Task):
    def __init__(self, name: str, function: Callable[[], None]) -> None:
        super().__init__(name=name)
        self.function = function

    def execute(self) -> None:
        self.function()


def run_linear(name: str, calls: Iterable[tuple[str, Callable[[], None]]]) -> None:
    flow = linear_flow.Flow(name)
    for task_name, function in calls:
        flow.add(CallTask(task_name, function))
    engines.run(flow, engine="serial")
