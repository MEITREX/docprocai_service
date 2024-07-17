from typing import Any, Generator
from torch._tensor import Tensor
import fileextractlib.PdfProcessor as PdfProcessor
import fileextractlib.SentenceEmbeddingRunner as SentenceEmbeddingRunner


class LecturePdfEmbeddingGenerator:
    class GenerateEmbeddingResult:
        def __init__(self, embedding: Tensor, text: str, page_number: int):
            self.embedding: Tensor = embedding
            self.text: str = text
            self.page_number: int = page_number

    def generate_embedding(self, file_url: str) -> list[GenerateEmbeddingResult]:
        pdf_processor = PdfProcessor.PdfProcessor()
        pages = pdf_processor.process_from_url(file_url)

        # remove null and empty strings
        filtered_pages = [x for x in pages if x["text"] is not None and x["text"].strip()]

        sentence_embedding_runner = SentenceEmbeddingRunner.SentenceEmbeddingRunner()
        embeddings = sentence_embedding_runner.generate_embeddings([x["text"] for x in filtered_pages])

        results: list[LecturePdfEmbeddingGenerator.GenerateEmbeddingResult] = []

        for i, embedding in enumerate(embeddings):
            results.append(LecturePdfEmbeddingGenerator.GenerateEmbeddingResult(embedding,
                                                                                filtered_pages[i]["text"],
                                                                                filtered_pages[i]["page_number"]))
        return results
