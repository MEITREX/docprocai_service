from uuid import UUID
from dataclasses import dataclass
from enum import Enum, auto


class CrudOperation(Enum):
    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()

@dataclass
class ContentChangeEvent:
    contentIds: list[UUID]
    crudOperation: CrudOperation