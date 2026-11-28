"""A small least-recently-used cache for decoded frames."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Generic, Hashable, TypeVar

V = TypeVar("V")


class LRUCache(Generic[V]):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: OrderedDict[Hashable, V] = OrderedDict()

    def get_or_create(self, key: Hashable, create: Callable[[], V]) -> V:
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = create()
        self._items[key] = value
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value
