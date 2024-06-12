import fastapi
import SentenceEmbeddingRunner
from pydantic import BaseModel
import LlamaRunner
import LectureVideoProcessor

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

class GenerateTagsFromVideoRequest(BaseModel):
    video_url: str

@app.post(path="/generate-tags-from-video/")
def generate_tags_from_video(request: GenerateTagsFromVideoRequest):
    class TranscriptAnswerSchema(BaseModel):
        tag1: str
        tag2: str
        tag3: str
        tag4: str
        tag5: str

    lecture_video_processor = LectureVideoProcessor.LectureVideoProcessor()
    transcript_text = lecture_video_processor.process(request.video_url)

    input_text = "# Video Transcript:\n" + transcript_text + "\n\n# Json Schema:\n" + TranscriptAnswerSchema.schema_json() + "\n\n# Json Result:\n"

    LlamaRunner.generate_text(input_text, TranscriptAnswerSchema)