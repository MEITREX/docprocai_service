from typing import Generic, TypeVar, Callable

T = TypeVar("T")

class SortedPriorityQueue(Generic[T]):
    """
    A priority queue which is backed by a sorted list which is re-sorted each time an element is inserted, compared
    to the "regular" python priority queue which is backed by a heap.
    This queue is slower, but has the advantage that a definitive position in the queue can be assigned to each item
    (although that position in the queue might increase if another item with a higher priority is enqueued).
    Priority is determined through the items' sort order, where an item being LESS THAN another equals a HIGHER priority
    to the SMALLER item.
    """

    def __init__(self):
        self.__items: list[T] = []

    def put(self, item: T) -> None:
        """
        Enqueue an item into the queue.
        :param item: The item to enqueue.
        :return: Nothing
        """
        self.__items.append(item)
        self.__items.sort()

    def __len__(self) -> int:
        """
        :return: Number of items in the queue.
        """
        return len(self.__items)

    def first_index_satisfying_predicate(self, predicate: Callable[[T], bool]):
        """
        Returns the index of the first (i.e. highest-priority)
        :param predicate: Function to which the item is passed. If the function evaluates to true for an item, this
        item's index is returned.
        :return: Returns the index of the first (i.e. highest-priority) item in the queue matching the predicate.
        :raise: ValueError If no item matching predicate could be found in the queue.
        """
        try:
            return self.__items.index(next(x for x in self.__items if predicate(x)))
        except StopIteration:
            raise ValueError("No item matching predicate could be found in the queue.")

    def get(self) -> T:
        """
        Dequeues (i.e. removes) the highest-priority item from the queue and returns item.
        :return: The highest-priority item in the queue.
        """
        return self.__items.pop(0)