import asyncio

from pgvector.psycopg import register_vector
import psycopg
import fileextractlib.LlamaRunner as LlamaRunner
import threading
import queue
from typing import Callable, Self, Awaitable
import uuid
import client.MediaServiceClient as MediaServiceClient
from fileextractlib.LecturePdfEmbeddingGenerator import LecturePdfEmbeddingGenerator
from fileextractlib.LectureVideoEmbeddingGenerator import LectureVideoEmbeddingGenerator
from fileextractlib.SentenceEmbeddingRunner import SentenceEmbeddingRunner
import logging
import Levenshtein

_logger = logging.getLogger(__name__)


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
                              + "(PRIMARY KEY(media_record, page), "
                              + "text text, "
                              + "media_record uuid, "
                              + "page int, "
                              + "embedding vector(1024))")
        self._db_conn.execute("CREATE TABLE IF NOT EXISTS videos "
                              + "(PRIMARY KEY(media_record, start_time), "
                              + "screen_text text, "
                              + "transcript text, "
                              + "media_record uuid, "
                              + "start_time int, "
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
        self._db_conn.close()

    def enqueue_ingest_media_record_task(self, media_record_id: uuid.UUID):
        async def ingest_media_record_task():
            media_record = await self._media_service_client.get_media_record_type_and_download_url(media_record_id)
            download_url = media_record["internalDownloadUrl"]

            record_type: str = media_record["type"]

            _logger.info("Ingesting media record with download URL: " + download_url)

            if record_type == "PRESENTATION" or record_type == "DOCUMENT":
                embedding_results = self._lecture_pdf_embedding_generator.generate_embedding(download_url)
                for embedding_result in embedding_results:
                    self._db_conn.execute(
                        query="INSERT INTO documents (text, media_record, page, embedding) VALUES (%s, %s, %s, %s)",
                        params=(embedding_result.text, media_record_id, embedding_result.page_number,
                                embedding_result.embedding))
            elif record_type == "VIDEO":
                embedding_results = self._lecture_video_embedding_generator.generate_embeddings(download_url)
                for embedding_result in embedding_results:
                    self._db_conn.execute(
                        query="INSERT INTO videos (screen_text, transcript, media_record, start_time, embedding) "
                              + "VALUES (%s, %s, %s, %s, %s)",
                        params=(embedding_result.screen_text, embedding_result.transcript, media_record_id,
                                embedding_result.start_time, embedding_result.embedding))
            else:
                raise ValueError("Asked to ingest unsupported media record type of type " + media_record["type"])

            _logger.info("Finished ingesting media record with download URL: " + download_url)

        priority = 0
        self._background_task_queue.put(DocProcAiService.BackgroundTaskItem(ingest_media_record_task, priority))

    def enqueue_generate_content_media_record_links(self, content_id: uuid.UUID, media_record_ids: list[uuid.UUID]):
        async def generate_content_media_record_links_task():
            query = """
            WITH document_results AS (
                SELECT
                    media_record AS "mediaRecordId",
                    'document' AS source,
                    page,
                    NULL::integer AS "startTime",
                    text AS "text"
                FROM documents
                WHERE media_record = ANY(%(mediaRecordIds)s)
            ),
            video_results AS (
                SELECT 
                    media_record AS "mediaRecordId",
                    'video' AS source,
                    NULL::integer AS page,
                    start_time AS "startTime",
                    screen_text AS "text"
                FROM videos
                WHERE media_record = ANY(%(mediaRecordIds)s)
            ),
            results AS (
                SELECT * FROM document_results
                UNION ALL
                SELECT * FROM video_results
            )
            SELECT * FROM results
            """

            # we could first run a query to check if any records even match, but considering that linking usually only
            # happens after ingestion, we can assume that the records exist, so that would be an unnecessary query
            query_result = self._db_conn.execute(query=query, params={"mediaRecordIds": media_record_ids}).fetchall()

            # create separate lists of records for each media record
            media_records_segments = {}
            # group the results by media record id
            for result in query_result:
                media_record_id = result["mediaRecordId"]
                if media_record_id not in media_records_segments:
                    media_records_segments[media_record_id] = []
                media_records_segments[media_record_id].append(result)

            # now we can check for links

            # go through each media record's segments
            for media_record_id, segments in media_records_segments.items():
                for segment in segments:
                    # get the text of the segment
                    segment_text = segment["text"]
                    # go through each other media record's segments
                    for other_media_record_id, other_segments in media_records_segments.items():
                        # skip if the other media record is the same as the current one
                        if other_media_record_id == media_record_id:
                            continue
                        for other_segment in other_segments:
                            other_segment_text = other_segment["text"]
                            # calculate the levenshtein distance between the two texts
                            levenshtein_distance = Levenshtein.distance(segment_text, other_segment_text)
                            # if the distance is less than 10, we can assume that the two segments are similar
                            if levenshtein_distance < 10:
                                # insert a link between the two segments
                                # TODO
                                pass



        # priority of media record link generation needs to be higher than that of media record ingestion (higher
        # priority items are processed last), so that the media records which are being linked have been processed
        # before being linked
        priority = 1
        self._background_task_queue.put(
            DocProcAiService.BackgroundTaskItem(generate_content_media_record_links_task, priority))

    def delete_entries_of_media_record(self, media_record_id: uuid.UUID):
        self._db_conn.execute("DELETE FROM documents WHERE media_record = %s", (media_record_id,))
        self._db_conn.execute("DELETE FROM videos WHERE media_record = %s", (media_record_id,))

    def semantic_search(self,
                        query_text: str,
                        count: int,
                        mediaRecordBlacklist: list[uuid.UUID],
                        mediaRecordWhitelist: list[uuid.UUID]) -> dict[str, any]:
        query_embedding = self._sentence_embedding_runner.generate_embeddings([query_text])[0]

        # sql query to get the closest embeddings to the query embedding, both from the video and document tables
        query = """
            WITH document_results AS (
                SELECT
                    media_record AS "mediaRecordId",
                    'document' AS source,
                    page,
                    NULL::integer AS "startTime",
                    text,
                    NULL::text AS "screenText",
                    NULL::text AS transcript,
                    embedding <=> %(query_embedding)s AS score
                FROM documents
                WHERE media_record = ANY(%(mediaRecordWhitelist)s) AND NOT media_record = ANY(%(mediaRecordBlacklist)s)
            ),
            video_results AS (
                SELECT 
                    media_record AS "mediaRecordId",
                    'video' AS source,
                    NULL::integer AS page,
                    start_time AS "startTime",
                    NULL::text AS text,
                    screen_text AS "screenText",
                    transcript,
                    embedding <=> %(query_embedding)s AS score
                FROM videos
                WHERE media_record = ANY(%(mediaRecordWhitelist)s) AND NOT media_record = ANY(%(mediaRecordBlacklist)s)
            ),
            results AS (
                SELECT * FROM document_results
                UNION ALL
                SELECT * FROM video_results
            )
            SELECT * FROM results ORDER BY score LIMIT %(count)s
        """

        query_result = self._db_conn.execute(query=query, params={
            "query_embedding": query_embedding,
            "count": count,
            "mediaRecordBlacklist": mediaRecordBlacklist,
            "mediaRecordWhitelist": mediaRecordWhitelist
        }).fetchall()

        for result in query_result:
            _logger.error(result)
            if result["source"] == "document":
                del result["startTime"]
                del result["screenText"]
                del result["transcript"]
            elif result["source"] == "video":
                del result["page"]
                del result["text"]

        return query_result

    class BackgroundTaskItem:
        def __init__(self, task: Callable[[], Awaitable[None]], priority: int):
            self.task: Callable[[], Awaitable[None]] = task
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
        asyncio.run(background_task_item.task())
        task_queue.task_done()
