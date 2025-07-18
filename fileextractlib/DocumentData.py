from PIL.Image import Image
from typing import Optional
from numpy.typing import NDArray
import numpy as np

class PageData:
    def __init__(self, page_number: int, text: str, thumbnail: Image, embedding: Optional[NDArray[np.float_]]):
        self.page_number = page_number
        self.text = text
        self.thumbnail = thumbnail
        self.embedding = embedding


class DocumentData:
    def __init__(self, pages: list[PageData], summary: list[str]):
        self.pages = pages
        self.summary = summary
