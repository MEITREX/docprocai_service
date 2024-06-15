import fastapi
import SentenceEmbeddingRunner
from pydantic import BaseModel
import LlamaRunner
import LectureVideoProcessor
import PdfProcessor as PdfProcessor
import uvicorn
from pgvector.psycopg import register_vector
import psycopg

app = fastapi.FastAPI()

db_conn = psycopg.connect("user=root password=root host=localhost port=5431 dbname=search-service")

db_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(db_conn)

db_conn.execute("DROP TABLE IF EXISTS documents")
db_conn.execute("CREATE TABLE IF NOT EXISTS documents (id SERIAL PRIMARY KEY, text text, embedding vector(1024))")

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

class IngestFileIntoDbRequest(BaseModel):
    file_url: str

@app.post("/db-ingest-pages/", status_code=202)
def ingest_file_into_db(request: IngestFileIntoDbRequest, background_tasks: fastapi.BackgroundTasks):
    def ingest_file_task(file_url: str):
        pdf_processsor = PdfProcessor.PdfProcessor()
        pages: list[str] = pdf_processsor.process(file_url)

        # remove null and empty strings
        filtered_pages_text = [x["text"] for x in pages if x["text"] is not None and x["text"].strip()]

        embeddings = SentenceEmbeddingRunner.generate_embeddings(filtered_pages_text)

        for embedding in embeddings:
            db_conn.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s)", (filtered_pages_text.pop(0), embedding.tolist()))


    background_tasks.add_task(ingest_file_task, request.file_url)

    return {"message": "File has been added to ingest queue."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)