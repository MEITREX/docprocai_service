from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

def generate_embeddings(inputs: list[str]):
    embeddings = model.encode(inputs)
    return embeddings


