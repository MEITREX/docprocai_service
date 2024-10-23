from sentence_transformers import SentenceTransformer
import config

class SentenceEmbeddingRunner:
    def __init__(self):
        self._model = SentenceTransformer(config.current["text_embedding"]["model_path"], trust_remote_code=True)

    def generate_embeddings(self, inputs: list[str]):
        embeddings = self._model.encode(inputs)
        return embeddings


