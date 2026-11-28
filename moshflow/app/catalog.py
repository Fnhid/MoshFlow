"""EffectCatalog: the list of available effects, for the UI and the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from moshflow.api import ParamSpec, all_effects, get_effect
from moshflow.core.discovery import DiscoveryFailure, discover_effects


@dataclass(frozen=True)
class EffectEntry:
    id: str
    name: str
    author: str
    description: str
    specs: Sequence[ParamSpec]


class EffectCatalog:
    def __init__(self):
        self.failures: list[DiscoveryFailure] = discover_effects()

    def entries(self) -> list[EffectEntry]:
        return [self._entry(cls) for cls in all_effects()]

    def get(self, effect_id: str) -> EffectEntry:
        return self._entry(get_effect(effect_id))

    @staticmethod
    def _entry(cls) -> EffectEntry:
        info = cls.info
        return EffectEntry(info.id, info.name, info.author, info.description, tuple(cls.param_specs()))
