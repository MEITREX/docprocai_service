from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

inputs = ["The weather is nice today.", "It is sunny outside.", "The weather is bad today."]

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
embeddings = model.encode(inputs)

print(str(cos_sim(embeddings[0], embeddings[1])))
print(str(cos_sim(embeddings[0], embeddings[2])))
