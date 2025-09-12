from dataclasses import dataclass
from typing import Literal, Optional

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
