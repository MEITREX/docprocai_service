import fileextractlib.SentenceEmbeddingRunner as SentenceEmbeddingRunner
from fileextractlib.DocumentData import PageData


class LectureDocumentEmbeddingGenerator:
    def generate_embeddings(self, page_data: list[PageData]):
        """
        Generates embeddings for the passed page data.
        """

        sentence_embedding_runner = SentenceEmbeddingRunner.SentenceEmbeddingRunner()
        embeddings = sentence_embedding_runner.generate_embeddings([page.text for page in page_data])

        for i, embedding in enumerate(embeddings):
            page_data[i].embedding = embedding
