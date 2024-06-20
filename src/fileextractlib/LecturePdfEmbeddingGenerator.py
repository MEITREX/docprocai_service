from typing import Any, Generator
from torch._tensor import Tensor
import PdfProcessor
import SentenceEmbeddingRunner

class LecturePdfEmbeddingGenerator:
    def generate_embedding(file_url: str)-> list[tuple[Tensor, str, int]]:
        pdf_processsor = PdfProcessor.PdfProcessor()
        pages = pdf_processsor.process_from_url(file_url)

        # remove null and empty strings
        filtered_pages = [x for x in pages if x["text"] is not None and x["text"].strip()]

        embeddings = SentenceEmbeddingRunner.generate_embeddings([x["text"] for x in filtered_pages])

        return list(zip(embeddings, [x["text"] for x in filtered_pages], [x["page_number"] for x in filtered_pages]))