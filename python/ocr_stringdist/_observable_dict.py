from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class _ObservableDict(dict[K, V]):
    """A dictionary that triggers a callback on mutation."""

    _MISSING = object()

    def __init__(
        self,
        mapping: dict[K, V],
        on_change: Callable[[], None],
        validator: Optional[Callable[[K, V], None]] = None,
    ) -> None:
        self._on_change = on_change
        self._validator = validator
        if validator:
            for k, v in mapping.items():
                validator(k, v)
        super().__init__(mapping)

    def __setitem__(self, key: K, value: V) -> None:
        if self._validator:
            self._validator(key, value)
        super().__setitem__(key, value)
        self._on_change()

    def __delitem__(self, key: K) -> None:
        super().__delitem__(key)
        self._on_change()

    def clear(self) -> None:
        super().clear()
        self._on_change()

    def pop(self, key: K, default: Any = _MISSING) -> V:
        res = super().pop(key) if default is self._MISSING else super().pop(key, default)
        self._on_change()
        return res

    def popitem(self) -> tuple[K, V]:
        res = super().popitem()
        self._on_change()
        return res

    def update(self, other: Any = (), /, **kwargs: V) -> None:
        items = self._items_from_update_args(other, kwargs)
        if self._validator:
            for key, value in items:
                self._validator(key, value)
        super().update(items)
        self._on_change()

    def setdefault(self, key: K, default: Any = None) -> V:
        if key in self:
            return self[key]
        if self._validator:
            self._validator(key, default)
        super().__setitem__(key, default)
        self._on_change()
        return default

    def __or__(self, other: object, /) -> Any:
        if not isinstance(other, dict):
            return NotImplemented
        return dict(self) | other

    def __ior__(self, other: object, /) -> Any:
        self.update(other)
        return self

    def _items_from_update_args(self, other: Any, kwargs: dict[str, V]) -> list[tuple[Any, Any]]:
        items = list(other.items()) if isinstance(other, Mapping) else list(other)
        items.extend(kwargs.items())
        return items
