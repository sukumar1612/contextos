"""Shared contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Action(IntEnum):
    CONTINUE = 0
    PREPARE = 1
    COMMIT = 2


@dataclass(frozen=True)
class Decision:
    action: Action
    log_prob: float | None = None
    value: float = 0.0
    version: int = 0


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Reply:
    message: dict[str, Any]
    usage: Usage
    finish_reason: str
    cost_usd: float
    key: str = ""  # name of the environment variable whose token served the request


@dataclass(frozen=True)
class ToolResult:
    output: str
    returncode: int = 0
    finished: bool = False


@dataclass(frozen=True)
class PublicTask:
    """The only task fields the agent ever sees."""

    instance_id: str
    problem_statement: str
    repo: str
    base_commit: str
    image: str


@dataclass(frozen=True)
class Grade:
    resolved: bool
    status: str
    details: dict = field(default_factory=dict)


class ContextLimit(RuntimeError):
    """The prompt cannot be made to fit. The episode ends and the patch is graded."""


class SummaryInvalid(ValueError):
    pass


def validate_mask(mask: list[bool]) -> None:
    if len(mask) != 3 or not mask[0] or (mask[1] and mask[2]):
        raise ValueError("Mask must allow CONTINUE and at most one of PREPARE/COMMIT")
