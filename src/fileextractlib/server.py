import fastapi
import SentenceEmbeddingRunner
from pydantic import BaseModel

app = fastapi.FastAPI()

@app.get("/generate-embedding/")
def generate_embedding(input_text: str):
    embeddings = SentenceEmbeddingRunner.generate_embeddings([input_text])
    return {
        "embedding": embeddings[0].tolist()
    }

class GenerateEmbeddingsRequest(BaseModel):
    inputs: list[str]

@app.post("/generate-embeddings/")
def generate_embeddings(request: GenerateEmbeddingsRequest):
    embeddings = SentenceEmbeddingRunner.generate_embeddings(request.inputs)
    return {
        "embeddings": [embedding.tolist() for embedding in embeddings]
    }