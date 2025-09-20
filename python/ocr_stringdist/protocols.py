from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .edit_operation import EditOperation


@runtime_checkable
class Aligner(Protocol):
    def explain(self, s1: str, s2: str, filter_matches: bool) -> list["EditOperation"]: ...
