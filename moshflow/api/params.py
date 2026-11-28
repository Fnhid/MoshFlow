"""Parameter specifications.

An effect declares its tunable values as a list of these specs. The UI builds
one widget per spec, and the renderer validates project values against them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Union


@dataclass(frozen=True)
class IntParam:
    id: str
    label: str
    default: int
    min: int
    max: int


@dataclass(frozen=True)
class FloatParam:
    id: str
    label: str
    default: float
    min: float
    max: float


@dataclass(frozen=True)
class ChoiceParam:
    id: str
    label: str
    default: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class BoolParam:
    id: str
    label: str
    default: bool


ParamSpec = Union[IntParam, FloatParam, ChoiceParam, BoolParam]


def default_params(specs: Sequence[ParamSpec]) -> dict[str, object]:
    return {spec.id: spec.default for spec in specs}


def validate_params(specs: Sequence[ParamSpec], values: Mapping[str, object]) -> dict[str, object]:
    """Returns a complete, type-checked parameter dict.

    Missing values get their default. Unknown ids, wrong types and
    out-of-range values raise ValueError.
    """
    known = {spec.id: spec for spec in specs}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(sorted(unknown))}")

    result: dict[str, object] = {}
    for spec in specs:
        value = values.get(spec.id, spec.default)
        result[spec.id] = _check(spec, value)
    return result


def _check(spec: ParamSpec, value: object) -> object:
    if isinstance(spec, BoolParam):
        if not isinstance(value, bool):
            raise ValueError(f"{spec.id}: expected true/false, got {value!r}")
        return value

    if isinstance(spec, ChoiceParam):
        if value not in spec.options:
            raise ValueError(f"{spec.id}: expected one of {spec.options}, got {value!r}")
        return value

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{spec.id}: expected a number, got {value!r}")
    if isinstance(spec, IntParam):
        if value != int(value):
            raise ValueError(f"{spec.id}: expected an integer, got {value!r}")
        value = int(value)
    else:
        value = float(value)
    if not spec.min <= value <= spec.max:
        raise ValueError(f"{spec.id}: {value} is outside [{spec.min}, {spec.max}]")
    return value
