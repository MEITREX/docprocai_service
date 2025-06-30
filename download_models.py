from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

cache_dir = "./llm_data/models/gte-large-en-v1.5"

model_id = "Alibaba-NLP/gte-large-en-v1.5"

snapshot_download(repo_id=model_id, local_dir=cache_dir)