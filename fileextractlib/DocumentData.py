from PIL.Image import Image
from torch import Tensor
from typing import Optional


class PageData:
    def __init__(self, page_number: int, text: str, thumbnail: Image, embedding: Optional[Tensor]):
        self.page_number = page_number
        self.text = text
        self.thumbnail = thumbnail
        self.embedding = embedding


class DocumentData:
    def __init__(self, pages: list[PageData], summary: list[str]):
        self.pages = pages
        self.summary = summary
