"""moshflow.api: the contract between the core and the effects (API v1).

Effects import only from this package and from moshflow.toolkit.
"""

from __future__ import annotations

from moshflow.api.effect import EffectInfo, TransitionContext, TransitionEffect
from moshflow.api.frames import Frame, FrameSink, FrameSource, ProgressReporter
from moshflow.api.params import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    IntParam,
    ParamSpec,
    default_params,
    validate_params,
)
from moshflow.api.registry import all_effects, get_effect, register

API_VERSION = "1.0"

__all__ = [
    "API_VERSION",
    "BoolParam",
    "ChoiceParam",
    "EffectInfo",
    "FloatParam",
    "Frame",
    "FrameSink",
    "FrameSource",
    "IntParam",
    "ParamSpec",
    "ProgressReporter",
    "TransitionContext",
    "TransitionEffect",
    "all_effects",
    "default_params",
    "get_effect",
    "register",
    "validate_params",
]
