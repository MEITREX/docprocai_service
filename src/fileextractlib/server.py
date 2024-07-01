import fastapi
import SentenceEmbeddingRunner
from pydantic import BaseModel
import LlamaRunner
import TranscriptGenerator as TranscriptGenerator
import PdfProcessor as PdfProcessor
import uvicorn
from pgvector.psycopg import register_vector
import psycopg
from LecturePdfEmbeddingGenerator import LecturePdfEmbeddingGenerator
from LectureVideoEmbeddingGenerator import LectureVideoEmbeddingGenerator

app = fastapi.FastAPI()

db_conn = psycopg.connect("user=root password=root host=localhost port=5431 dbname=search-service", autocommit=True, row_factory=psycopg.rows.dict_row)

db_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(db_conn)

#db_conn.execute("DROP TABLE IF EXISTS documents")
#db_conn.execute("DROP TABLE IF EXISTS videos")

db_conn.execute("CREATE TABLE IF NOT EXISTS documents (PRIMARY KEY(origin_file, page), text text, origin_file text, page int, embedding vector(1024))")
db_conn.execute("CREATE TABLE IF NOT EXISTS videos (PRIMARY KEY(origin_file, start_time), screen_text text, transcript text, origin_file text, start_time int, embedding vector(1024))")

llamaRunner: LlamaRunner.LlamaRunner | None = None

def ingest_document_task(file_url: str):
    embeddings = LecturePdfEmbeddingGenerator.generate_embedding(file_url)

    for embedding, text, page_no in embeddings:
        db_conn.execute(query="INSERT INTO documents (text, origin_file, page, embedding) VALUES (%s, %s, %s, %s)",
                        params=(text, file_url, page_no, embedding))

    print(f"File {file_url} has been ingested into the database.")

def ingest_video_task(video_url: str):
    lecture_video_embedding_generator = LectureVideoEmbeddingGenerator()
    embeddings: list[LectureVideoEmbeddingGenerator.Section] = lecture_video_embedding_generator.generate_embeddings(video_url)

    for embedding in embeddings:
        db_conn.execute(query="INSERT INTO videos (screen_text, transcript, origin_file, start_time, embedding) VALUES (%s, %s, %s, %s, %s)",
                        params=(embedding.screen_text, embedding.transcript, video_url, embedding.start_time, embedding.embedding))
        
    print(f"Video {video_url} has been ingested into the database.")
    

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

    lecture_video_processor = TranscriptGenerator.LectureVideoProcessor()
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

class IngestVideoIntoDbRequest(BaseModel):
    video_url: str

@app.post("/db-ingest-video/", status_code=202)
def ingest_video_into_db(request: IngestVideoIntoDbRequest, background_tasks: fastapi.BackgroundTasks):
    background_tasks.add_task(ingest_video_task, request.video_url)

    return {"message": "Video has been added to ingest queue."}

@app.get("/search/")
def db_find_neighbor(query: str, count: int = 5):
    if count < 1 or count > 100:
        raise fastapi.HTTPException(status_code=400, detail="Count must be between 1 and 100.")

    query_embedding = SentenceEmbeddingRunner.generate_embeddings([query])[0]

    query = """
    WITH document_results AS (
        SELECT
            origin_file,
            'document' AS source,
            page,
            NULL::integer AS start_time,
            text,
            NULL::text AS screen_text,
            NULL::text AS transcript,
            embedding <=> %s AS distance
        FROM documents
    ),
    video_results AS (
        SELECT origin_file,
            'video' AS source,
            NULL::integer AS page,
            start_time,
            NULL::text AS text,
            screen_text,
            transcript,
            embedding <=> %s AS distance
        FROM videos
    ),
    results AS (
        SELECT * FROM document_results
        UNION ALL
        SELECT * FROM video_results
    )
    SELECT * FROM results ORDER BY distance LIMIT %s
    """

    query_result = db_conn.execute(query=query, params=(query_embedding, query_embedding, count)).fetchall()

    for result in query_result:
        if result["source"] == "document":
            del result["start_time"]
            del result["screen_text"]
            del result["transcript"]
        elif result["source"] == "video":
            del result["page"]
            del result["text"]

    return query_result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)