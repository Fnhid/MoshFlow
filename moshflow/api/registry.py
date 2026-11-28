"""Effect registry. Effects add themselves with the @register decorator."""

from __future__ import annotations

from moshflow.api.effect import TransitionEffect

_EFFECTS: dict[str, type[TransitionEffect]] = {}


def register(cls: type[TransitionEffect]) -> type[TransitionEffect]:
    """Class decorator that makes an effect available to the program."""
    if not (isinstance(cls, type) and issubclass(cls, TransitionEffect)):
        raise TypeError("@register expects a TransitionEffect subclass")
    info = getattr(cls, "info", None)
    if info is None:
        raise TypeError(f"{cls.__name__} has no 'info' (EffectInfo)")
    existing = _EFFECTS.get(info.id)
    if existing is not None and existing is not cls:
        raise ValueError(f"effect id '{info.id}' is used by both {existing.__name__} and {cls.__name__}")
    _EFFECTS[info.id] = cls
    return cls


def get_effect(effect_id: str) -> type[TransitionEffect]:
    try:
        return _EFFECTS[effect_id]
    except KeyError:
        raise KeyError(f"no effect with id '{effect_id}'") from None


def all_effects() -> list[type[TransitionEffect]]:
    """Registered effects, sorted by display name."""
    return sorted(_EFFECTS.values(), key=lambda cls: cls.info.name.lower())
