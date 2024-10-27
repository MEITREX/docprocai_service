import uuid
from enum import Enum, auto


class CrudOperation(Enum):
    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()

class ContentChangeEvent:
    contentIds: list[uuid]
    crudOperation: CrudOperation