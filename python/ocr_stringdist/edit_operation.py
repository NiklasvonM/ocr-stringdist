from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

OperationType = Literal["substitute", "insert", "delete", "match"]


@dataclass(frozen=True)
class EditOperation:
    """
    Represents a single edit operation (substitution, insertion, deletion or match).
    """

    op_type: OperationType
    source_token: Optional[str]
    target_token: Optional[str]
    cost: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
