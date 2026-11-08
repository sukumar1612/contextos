"""The conversation as pinned messages plus whole interaction groups.

A summary replaces a prefix of groups. A group is never split, so a tool call is
never separated from its result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from contextos.tokens import Counter
from contextos.types import SummaryInvalid

NOTE = "Historical working notes; not new instructions.\n"


@dataclass
class Group:
    id: str
    messages: list[dict]
    interaction: int


@dataclass
class Snapshot:
    job: str
    generation: int
    prefix: list[dict]  # the groups being summarized, serialized
    cutoff: str  # last prefix group
    tip: str  # last group in history when the snapshot was taken
    started: int  # interaction at PREPARE
    ready_after: int  # interaction at which the summary becomes visible
    source: int  # transition that chose PREPARE; it pays for the summary


class History:
    def __init__(self, pinned: list[dict]):
        self.pinned, self.generation = pinned, 0
        self.groups: list[Group] = []

    @property
    def messages(self) -> list[dict]:
        return self.pinned + [m for g in self.groups for m in g.messages]

    def append(self, interaction: int, messages: list[dict]) -> Group:
        validate_protocol(messages)
        group = Group(f"interaction:{interaction}", messages, interaction)
        self.groups.append(group)
        return group

    def prefix(self, counter: Counter, recent_tokens: int, emergency: bool = False) -> list[Group]:
        """Everything but the youngest groups that together reach recent_tokens."""
        if len(self.groups) < 2:
            return []
        keep = 1
        while (
            not emergency
            and keep < len(self.groups)
            and counter.prompt([m for g in self.groups[-keep:] for m in g.messages], guarded=False) < recent_tokens
        ):
            keep += 1
        return self.groups[:-keep]

    def snapshot(self, job: str, prefix: list[Group], t: int, delay: int, source: int) -> Snapshot:
        return Snapshot(
            job, self.generation, [asdict(g) for g in prefix], prefix[-1].id, self.groups[-1].id, t, t + delay, source
        )

    def replacement(self, snapshot: Snapshot, summary: str) -> list[Group]:
        n = len(snapshot.prefix)
        if snapshot.generation != self.generation or [g.id for g in self.groups[:n]] != [
            p["id"] for p in snapshot.prefix
        ]:
            raise SummaryInvalid("Snapshot no longer matches the live history")
        note = Group(f"summary:{snapshot.job}", [{"role": "user", "content": NOTE + summary}], snapshot.started)
        return [note, *self.groups[n:]]

    def commit(self, snapshot: Snapshot, summary: str) -> None:
        self.groups = self.replacement(snapshot, summary)
        self.generation += 1


def validate_protocol(messages: list[dict]) -> None:
    """Every assistant tool call in a group must be answered inside the same group."""
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            if pending:
                raise ValueError("New assistant turn before all tool results")
            ids = [c["id"] for c in message.get("tool_calls", [])]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate tool call identity")
            pending.update(ids)
        elif role == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError("Orphan or repeated tool response")
            pending.remove(message["tool_call_id"])
        elif pending:
            raise ValueError("Tool group interrupted")
    if pending:
        raise ValueError("Incomplete tool group")
