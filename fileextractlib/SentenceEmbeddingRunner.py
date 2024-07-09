from sentence_transformers import SentenceTransformer


class SentenceEmbeddingRunner:
    def __init__(self):
        self._model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

    def generate_embeddings(self, inputs: list[str]):
        embeddings = self._model.encode(inputs)
        return embeddings


