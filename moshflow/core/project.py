"""Project data: an ordered list of clips with one transition per cut.

Every type here is immutable. An edit returns a new Project and leaves the old
one untouched, so undo can keep earlier versions and a render can run on one
version while the user keeps editing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

ALIGNMENTS = ("before", "center", "after")
DEFAULT_EFFECT = "hard_cut"


@dataclass(frozen=True)
class Clip:
    path: str
    start: float = 0.0           # trim in, seconds
    end: float | None = None     # trim out, seconds; None = end of file


@dataclass(frozen=True)
class Transition:
    effect: str = DEFAULT_EFFECT
    duration: float = 3.0        # seconds
    alignment: str = "after"     # "before" / "center" / "after" the cut; after: A plays to its end
    params: Mapping[str, object] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def with_param(self, param_id: str, value: object) -> "Transition":
        return replace(self, params={**self.params, param_id: value})


@dataclass(frozen=True)
class OutputSettings:
    path: str = "output.mp4"
    width: int | None = None     # None = size of the first clip
    height: int | None = None
    fps: float | None = None     # None = frame rate of the first clip


@dataclass(frozen=True)
class Project:
    clips: tuple[Clip, ...] = ()
    transitions: tuple[Transition, ...] = ()   # transitions[i] sits between clips[i] and clips[i + 1]
    output: OutputSettings = OutputSettings()

    def __post_init__(self) -> None:
        object.__setattr__(self, "clips", tuple(self.clips))
        object.__setattr__(self, "transitions", tuple(self.transitions))

    # --- edits: each returns a new Project ---------------------------------
    def with_clip(self, clip: Clip, index: int | None = None) -> "Project":
        """Adds a clip at `index` (default: the end), with a new transition for the new cut."""
        transitions = self.transitions + (Transition(),) if self.clips else self.transitions
        added = replace(self, clips=self.clips + (clip,), transitions=transitions)
        last = len(added.clips) - 1
        return added if index is None or index == last else added.with_clip_moved(last, index)

    def without_clip(self, index: int) -> "Project":
        """Removes a clip and the transition of one of its cuts."""
        transitions = self.transitions
        if transitions:
            transitions = _without(transitions, min(index, len(transitions) - 1))
        return replace(self, clips=_without(self.clips, index), transitions=transitions)

    def with_clip_moved(self, src: int, dst: int) -> "Project":
        """Reorders clips; transitions stay attached to their cut position."""
        clips = list(self.clips)
        clips.insert(dst, clips.pop(src))
        return replace(self, clips=tuple(clips))

    def with_clip_replaced(self, index: int, clip: Clip) -> "Project":
        return replace(self, clips=_with_item(self.clips, index, clip))

    def with_transition(self, index: int, transition: Transition) -> "Project":
        return replace(self, transitions=_with_item(self.transitions, index, transition))

    def with_output(self, output: OutputSettings) -> "Project":
        return replace(self, output=output)

    def validate(self) -> None:
        if len(self.transitions) != max(0, len(self.clips) - 1):
            raise ValueError(f"{len(self.clips)} clips need {max(0, len(self.clips) - 1)} transitions, "
                             f"found {len(self.transitions)}")
        for i, t in enumerate(self.transitions, start=1):
            if t.alignment not in ALIGNMENTS:
                raise ValueError(f"transition {i}: alignment must be one of {ALIGNMENTS}")
            if t.duration < 0:
                raise ValueError(f"transition {i}: duration must not be negative")
        for i, c in enumerate(self.clips, start=1):
            if c.start < 0 or (c.end is not None and c.end <= c.start):
                raise ValueError(f"clip {i}: invalid trim {c.start}..{c.end}")


def _without(items: tuple, index: int) -> tuple:
    return items[:index] + items[index + 1:]


def _with_item(items: tuple, index: int, item: object) -> tuple:
    if not 0 <= index < len(items):
        raise IndexError(f"index {index} out of range for {len(items)} items")
    return items[:index] + (item,) + items[index + 1:]
