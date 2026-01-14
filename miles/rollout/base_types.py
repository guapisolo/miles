from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Protocol

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
    # TODO may refactor DataSource API
    data_source: DataSource


@dataclass(frozen=True)
class RolloutFnBaseInput:
    rollout_id: int

    @property
    def evaluation(self):
        raise NotImplementedError


# subclassing for different data in the future
@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    @property
    def evaluation(self):
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    @property
    def evaluation(self):
        return True


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


RolloutFnInput = RolloutFnTrainInput | RolloutFnEvalInput
RolloutFnOutput = RolloutFnTrainOutput | RolloutFnEvalOutput


# Duck typing, users do not need to extend this class
class RolloutFnProtocol(Protocol):
    def __init__(self, input: RolloutFnConstructorInput): ...

    @classmethod
    def add_arguments(cls, parser: ArgumentParser): ...

    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput: ...
