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

db_conn = psycopg.connect("user=root password=root host=localhost port=5431 dbname=search-service", autocommit=True, row_factory=psycopg.rows.dict_row)

db_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(db_conn)

#db_conn.execute("DROP TABLE IF EXISTS documents")
db_conn.execute("CREATE TABLE IF NOT EXISTS documents (PRIMARY KEY(origin_file, page), text text, origin_file text, page int, embedding vector(1024))")

llamaRunner: LlamaRunner.LlamaRunner | None = None

def ingest_document_task(file_url: str):
    pdf_processsor = PdfProcessor.PdfProcessor()
    pages: list[str] = pdf_processsor.process(file_url)

    # remove null and empty strings
    filtered_pages_text = [x["text"] for x in pages if x["text"] is not None and x["text"].strip()]

    embeddings = SentenceEmbeddingRunner.generate_embeddings(filtered_pages_text)

    for (page_no, embedding) in enumerate(embeddings, 1):
        db_conn.execute("INSERT INTO documents (text, origin_file, page, embedding) VALUES (%s, %s, %s, %s)", (filtered_pages_text.pop(0), file_url, page_no, embedding))

    print(f"File {file_url} has been ingested into the database.")

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
    
    if llamaRunner == None:
        llamaRunner = LlamaRunner.LlamaRunner()
    llamaRunner.generate_text(input_text, TranscriptAnswerSchema)

class IngestDocumentIntoDbRequest(BaseModel):
    file_url: str

@app.post("/db-ingest-document/", status_code=202)
def ingest_document_into_db(request: IngestDocumentIntoDbRequest, background_tasks: fastapi.BackgroundTasks):
    background_tasks.add_task(ingest_document_task, request.file_url)

    return {"message": "File has been added to ingest queue."}

class IngestDocumentsIntoDbRequest(BaseModel):
    file_urls: list[str]

@app.post("/db-ingest-documents/", status_code=202)
def ingest_documents_into_db(request: IngestDocumentsIntoDbRequest, background_tasks: fastapi.BackgroundTasks):
    for file_url in request.file_urls:
        background_tasks.add_task(ingest_document_task, file_url)

    return {"message": "Files have been added to ingest queue."}

@app.get("/search/")
def db_find_neighbor(query: str, count: int = 5):
    if count < 1 or count > 100:
        raise fastapi.HTTPException(status_code=400, detail="Count must be between 1 and 100.")

    query_embedding = SentenceEmbeddingRunner.generate_embeddings([query])[0]

    result = db_conn.execute("SELECT * FROM documents ORDER BY embedding <=> %s LIMIT %s", (query_embedding, count))
    return {
        "results": [{
                        "file": row["origin_file"],
                        "page": row["page"],
                        "text": row["text"],
                    } for row in result]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)