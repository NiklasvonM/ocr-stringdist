from typing import Any, Callable, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class _ObservableDict(dict[K, V]):
    """A dictionary that triggers a callback on mutation."""

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

    def pop(self, key: K, default: Any = None) -> V:
        res = super().pop(key, default)
        self._on_change()
        return res

    def popitem(self) -> tuple[K, V]:
        res = super().popitem()
        self._on_change()
        return res
