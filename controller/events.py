import uuid
from dataclasses import dataclass
from enum import Enum, auto


class CrudOperation(Enum):
    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()

@dataclass
class ContentChangeEvent:
    contentIds: list[uuid]
    crudOperation: CrudOperation