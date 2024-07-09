from pgvector.psycopg import register_vector
import psycopg
import fileextractlib.LlamaRunner as LlamaRunner
import threading
import queue
from typing import Callable, Self
import uuid
import client.MediaServiceClient as MediaServiceClient
from fileextractlib.LecturePdfEmbeddingGenerator import LecturePdfEmbeddingGenerator
from fileextractlib.LectureVideoEmbeddingGenerator import LectureVideoEmbeddingGenerator
from fileextractlib.SentenceEmbeddingRunner import SentenceEmbeddingRunner


class DocProcAiService:
    def __init__(self):
        # TODO: Make db connection configurable
        self._db_conn: psycopg.Connection = psycopg.connect(
            "user=root password=root host=database-docprocai port=5432 dbname=search-service",
            autocommit=True,
            row_factory=psycopg.rows.dict_row
        )

        # ensure pgvector extension is installed, we need it to store text embeddings
        self._db_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._db_conn)

        # db_conn.execute("DROP TABLE IF EXISTS documents")
        # db_conn.execute("DROP TABLE IF EXISTS videos")

        # ensure database tables exist
        self._db_conn.execute("CREATE TABLE IF NOT EXISTS documents "
                              + "(PRIMARY KEY(mediaRecord, page), "
                              + "text text, "
                              + "mediaRecord uuid, "
                              + "page int, "
                              + "embedding vector(1024))")
        self._db_conn.execute("CREATE TABLE IF NOT EXISTS videos "
                              + "(PRIMARY KEY(mediaRecord, startTime), "
                              + "screenText text, "
                              + "transcript text, "
                              + "mediaRecord uuid, "
                              + "startTime int, "
                              + "embedding vector(1024))")

        # graphql client for interacting with the media service
        self._media_service_client: MediaServiceClient.MediaServiceClient = MediaServiceClient.MediaServiceClient()

        # only load the llamaRunner the first time we actually need it, not now
        self._llama_runner: LlamaRunner.LlamaRunner | None = None

        self._sentence_embedding_runner: SentenceEmbeddingRunner = SentenceEmbeddingRunner()

        self._lecture_pdf_embedding_generator: LecturePdfEmbeddingGenerator = LecturePdfEmbeddingGenerator()
        self._lecture_video_embedding_generator: LectureVideoEmbeddingGenerator = LectureVideoEmbeddingGenerator()

        self._background_task_queue: queue.PriorityQueue[DocProcAiService.BackgroundTaskItem] = queue.PriorityQueue()

        self._keep_background_task_thread_alive: threading.Event = threading.Event()
        self._keep_background_task_thread_alive.set()

        self._background_task_thread: threading.Thread = threading.Thread(
            target=_background_task_runner,
            args=[self._background_task_queue, self._keep_background_task_thread_alive])
        self._background_task_thread.start()

    def __del__(self):
        self._keep_background_task_thread_alive = False

    def enqueue_ingest_media_record_task(self, media_record_id: uuid.UUID):
        media_record = self._media_service_client.get_media_record_type_and_download_url(media_record_id)
        download_url = media_record["downloadUrl"]

        def ingest_document_task():
            embeddings = self._lecture_pdf_embedding_generator.generate_embedding(download_url)
            for embedding, text, page_no in embeddings:
                self._db_conn.execute(
                    query="INSERT INTO documents (text, mediaRecord, page, embedding) VALUES (%s, %s, %s, %s)",
                    params=(text, media_record_id, page_no, embedding))

        def ingest_video_task():
            embeddings = self._lecture_video_embedding_generator.generate_embeddings(download_url)
            for embedding, screen_text, transcript, start_time in embeddings:
                self._db_conn.execute(
                    query="INSERT INTO videos (screenText, transcript, mediaRecord, startTime, embedding) "
                          + "VALUES (%s, %s, %s, %s, %s)",
                    params=(screen_text, transcript, media_record_id, start_time, embedding))

        if media_record["type"] == "VIDEO":
            self._background_task_queue.put(DocProcAiService.BackgroundTaskItem(ingest_video_task, 0))
        elif media_record["type"] == "PRESENTATION" or media_record["type"] == "DOCUMENT":
            self._background_task_queue.put(DocProcAiService.BackgroundTaskItem(ingest_document_task, 0))
        else:
            raise ValueError("Asked to ingest unsupported media record type of type " + media_record["type"])

    def semantic_search(self, query_text: str, count: int) -> dict[str, any]:
        query_embedding = self._sentence_embedding_runner.generate_embeddings([query_text])[0]

        # sql query to get the closest embeddings to the query embedding, both from the video and document tables
        query = """
            WITH document_results AS (
                SELECT
                    mediaRecord,
                    'document' AS source,
                    page,
                    NULL::integer AS startTime,
                    text,
                    NULL::text AS screenText,
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

        query_result = self._db_conn.execute(query=query, params=(query_embedding, query_embedding, count)).fetchall()

        for result in query_result:
            if result["source"] == "document":
                del result["startTime"]
                del result["screenText"]
                del result["transcript"]
            elif result["source"] == "video":
                del result["page"]
                del result["text"]
            # TODO: Delete this or not?
            del result["source"]

        return query_result

    class BackgroundTaskItem:
        def __init__(self, task: Callable[[], None], priority: int):
            self.task: Callable[[], None] = task
            self.priority: int = priority

        def __lt__(self, other: Self):
            return self.priority < other.priority


def _background_task_runner(task_queue: queue.PriorityQueue[DocProcAiService.BackgroundTaskItem],
                            keep_alive: threading.Event):
    while keep_alive.is_set():
        try:
            background_task_item = task_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        background_task_item.task()
        task_queue.task_done()
