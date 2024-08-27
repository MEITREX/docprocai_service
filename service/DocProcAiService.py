import asyncio

import PIL.Image
from pgvector.psycopg import register_vector
import psycopg
import fileextractlib.LlamaRunner as LlamaRunner
import threading
import queue
from typing import Callable, Self, Awaitable
import uuid
import client.MediaServiceClient as MediaServiceClient
from fileextractlib.DocumentProcessor import DocumentProcessor
from fileextractlib.LectureDocumentEmbeddingGenerator import LectureDocumentEmbeddingGenerator
from fileextractlib.LectureVideoEmbeddingGenerator import LectureVideoEmbeddingGenerator
from fileextractlib.SentenceEmbeddingRunner import SentenceEmbeddingRunner
import logging
from fileextractlib.VideoProcessor import VideoProcessor
from fileextractlib.ImageTemplateMatcher import ImageTemplateMatcher
import io

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

        # db_conn.execute("DROP TABLE IF EXISTS document_sections")
        # db_conn.execute("DROP TABLE IF EXISTS video_sections")

        # ensure database tables exist
        # table which contains the sections of all documents including their text, page number, and text embedding
        self._db_conn.execute("""
                              CREATE TABLE IF NOT EXISTS document_sections (
                                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                                text text,
                                media_record uuid,
                                page int,
                                thumbnail bytea,
                                embedding vector(1024)
                              );
                              """)
        # table which contains the sections of all videos including their screen text, transcript, start time, and text
        self._db_conn.execute("""
                              CREATE TABLE IF NOT EXISTS video_sections (
                                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                                screen_text text,
                                transcript text,
                                media_record uuid,
                                start_time int,
                                thumbnail bytea,
                                embedding vector(1024)
                              );
                              """)
        # table which contains the caption of full videos in WebVTT format. Primary key is the uuid of the media record
        # the row stores captions for, and the vtt column stores the WebVTT formatted captions
        self._db_conn.execute("""
                              CREATE TABLE IF NOT EXISTS video_captions (
                                media_record_id uuid PRIMARY KEY,
                                vtt text
                              );
                              """)
        # table which contains links between segments of different media records
        self._db_conn.execute("""
                              CREATE TABLE IF NOT EXISTS media_record_links (
                                content_id uuid,
                                segment1_id uuid,
                                segment2_id uuid
                              );
                              """)

        # graphql client for interacting with the media service
        self._media_service_client: MediaServiceClient.MediaServiceClient = MediaServiceClient.MediaServiceClient()

        # only load the llamaRunner the first time we actually need it, not now
        self._llama_runner: LlamaRunner.LlamaRunner | None = None

        self._sentence_embedding_runner: SentenceEmbeddingRunner = SentenceEmbeddingRunner()

        self._lecture_pdf_embedding_generator: LectureDocumentEmbeddingGenerator = LectureDocumentEmbeddingGenerator()
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

        if self._db_conn is not None:
            self._db_conn.close()

    def enqueue_ingest_media_record_task(self, media_record_id: uuid.UUID):
        async def ingest_media_record_task():
            media_record = await self._media_service_client.get_media_record_type_and_download_url(media_record_id)
            download_url = media_record["internalDownloadUrl"]

            record_type: str = media_record["type"]

            _logger.info("Ingesting media record with download URL: " + download_url)

            if record_type == "PRESENTATION" or record_type == "DOCUMENT":
                document_processor = DocumentProcessor()
                document_data = document_processor.process(download_url)
                self._lecture_pdf_embedding_generator.generate_embeddings(document_data.pages)
                for section in document_data.pages:
                    thumbnail_bytes = io.BytesIO()
                    section.thumbnail.save(thumbnail_bytes, format="JPEG", quality=93)
                    self._db_conn.execute(
                        query="""
                              INSERT INTO document_sections (text, media_record, page, thumbnail, embedding) 
                              VALUES (%s, %s, %s, %s, %s)
                              """,
                        params=(section.text, media_record_id, section.page_number,
                                thumbnail_bytes.getvalue(), section.embedding))
            elif record_type == "VIDEO":
                # TODO: make this configurable
                video_processor = VideoProcessor(section_image_similarity_threshold=0.9, minimum_section_length=15)
                video_data = video_processor.process(download_url)
                del video_processor

                # store the captions of the video
                self._db_conn.execute(
                    query="""
                          INSERT INTO video_captions (media_record_id, vtt)
                          VALUES (%s, %s)
                          """,
                    params=(media_record_id, video_data.vtt.content))

                # generate and store text embeddings for the sections of the video
                self._lecture_video_embedding_generator.generate_embeddings(video_data.sections)
                for section in video_data.sections:
                    thumbnail_bytes = io.BytesIO()
                    section.thumbnail.save(thumbnail_bytes, format="JPEG", quality=93)
                    self._db_conn.execute(
                        query="""
                              INSERT INTO video_sections (
                                screen_text,
                                transcript,
                                media_record,
                                start_time,
                                thumbnail,
                                embedding
                              )
                              VALUES (%s, %s, %s, %s, %s, %s)
                              """,
                        params=(section.screen_text, section.transcript, media_record_id,
                                section.start_time, thumbnail_bytes.getvalue(), section.embedding))
            else:
                raise ValueError("Asked to ingest unsupported media record type of type " + media_record["type"])

            _logger.info("Finished ingesting media record with download URL: " + download_url)

        priority = 0
        self._background_task_queue.put(DocProcAiService.BackgroundTaskItem(ingest_media_record_task, priority))

    def enqueue_generate_content_media_record_links(self, content_id: uuid.UUID, media_record_ids: list[uuid.UUID]):
        def generate_content_media_record_links_task():
            _logger.debug("Generating content media record links for content " + str(content_id) + "...")
            query = """
            WITH document_results AS (
                SELECT
                    id,
                    media_record AS "mediaRecordId",
                    'document' AS source,
                    page,
                    NULL::integer AS "startTime",
                    text AS "text",
                    thumbnail
                FROM document_sections
                WHERE media_record = ANY(%(mediaRecordIds)s)
            ),
            video_results AS (
                SELECT 
                    id,
                    media_record AS "mediaRecordId",
                    'video' AS source,
                    NULL::integer AS page,
                    start_time AS "startTime",
                    screen_text AS "text",
                    thumbnail
                FROM video_sections
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
            media_records_segments: dict[uuid.UUID, list] = {}
            # group the results by media record id
            for result in query_result:
                media_record_id: uuid.UUID = result["mediaRecordId"]
                if media_record_id not in media_records_segments:
                    media_records_segments[media_record_id] = []
                media_records_segments[media_record_id].append(result)

            # now we can check for links
            # go through each media record's segments
            for media_record_id, segments in reversed(media_records_segments.items()):
                for segment in segments:
                    # get the text of the segment
                    segment_thumbnail = PIL.Image.open(io.BytesIO(segment["thumbnail"]))

                    # TODO: Only crop if video
                    cropped_segment_thumbnail = segment_thumbnail.crop((
                        segment_thumbnail.width * 1/6, segment_thumbnail.height * 1/10,
                        segment_thumbnail.width * 5/6, segment_thumbnail.height * 9/10))

                    image_template_matcher = ImageTemplateMatcher(
                        template=cropped_segment_thumbnail,
                        scaling_factor=0.4,
                        enable_multi_scale_matching=True,
                        multi_scale_matching_steps=40
                    )

                    # go through each other media record's segments
                    for other_media_record_id, other_segments in media_records_segments.items():
                        # skip if the other media record is the same as the current one
                        if other_media_record_id == media_record_id:
                            continue

                        # find the segment with the highest similarity
                        max_similarity = 0
                        max_similarity_segment_id = None
                        for other_segment in other_segments:
                            # resize the other segment's thumbnail, so it's the same size as the template thumbnail
                            other_segment_thumbnail = PIL.Image.open(io.BytesIO(other_segment["thumbnail"]))
                            size_ratio = segment_thumbnail.height / other_segment_thumbnail.height
                            other_segment_thumbnail = other_segment_thumbnail.resize(
                                (int(other_segment_thumbnail.width * size_ratio), segment_thumbnail.height))

                            # use the image template matcher to try to match the thumbnails
                            # TODO: Only use center portion of image for matching
                            similarity = image_template_matcher.match(other_segment_thumbnail)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                max_similarity_segment_id = other_segment["id"]

                        # TODO: Make this configurable
                        if max_similarity > 0.75:
                            # create a link between the two segments
                            # skip if a link between these segments already exists
                            if not self.does_link_between_media_record_segments_exist(segment["id"],
                                                                                      max_similarity_segment_id):
                                self.create_link_between_media_record_segments(content_id,
                                                                               segment["id"],
                                                                               max_similarity_segment_id)

            _logger.debug("Generated content media record links for content " + str(content_id) + ".")

        # priority of media record link generation needs to be higher than that of media record ingestion (higher
        # priority items are processed last), so that the media records which are being linked have been processed
        # before being linked
        priority = 1
        generate_content_media_record_links_task()
        # TODO: this should be executed as a task, probably
        #self._background_task_queue.put(
        #    DocProcAiService.BackgroundTaskItem(generate_content_media_record_links_task, priority))

    def create_link_between_media_record_segments(self,
                                                  content_id: uuid.UUID,
                                                  segment_id: uuid.UUID,
                                                  other_segment_id: uuid.UUID):
        self._db_conn.execute("INSERT INTO media_record_links (content_id, segment1_id, segment2_id) "
                              + "VALUES (%s, %s, %s)",
                              (content_id, segment_id, other_segment_id))

    def does_link_between_media_record_segments_exist(self, segment_id: uuid.UUID, other_segment_id: uuid.UUID) -> bool:
        result = self._db_conn.execute("""
        SELECT EXISTS(
            SELECT 1 FROM media_record_links 
            WHERE (segment1_id = %(id1)s AND segment2_id = %(id2)s) 
                OR (segment1_id = %(id2)s AND segment2_id = %(id1)s)
        )
        """, {"id1": segment_id, "id2": other_segment_id}).fetchone()

        return result["exists"]

    def delete_entries_of_media_record(self, media_record_id: uuid.UUID):
        # delete media record segment links
        self._db_conn.execute("""
            DELETE FROM media_record_links 
            WHERE segment1_id = %(media_record_id)s OR segment2_id = %(media_record_id)s""",
                              {
                                    "media_record_id": media_record_id
                              })

        # delete media record segments
        self._db_conn.execute("DELETE FROM document_sections WHERE media_record = %s",
                              (media_record_id,))
        self._db_conn.execute("DELETE FROM video_sections WHERE media_record = %s",
                              (media_record_id,))

    def get_media_record_links_for_content(self, content_id: uuid.UUID):
        result = self._db_conn.execute("""
            SELECT
                segment1_id AS "segment1Id",
                segment2_id AS "segment2Id"
            FROM media_record_links WHERE content_id = %s
            """, (content_id,)).fetchall()

        all_segment_ids = []
        for segment_link in result:
            all_segment_ids.append(segment_link["segment1Id"])
            all_segment_ids.append(segment_link["segment2Id"])

        result_segments = self._db_conn.execute("""
            WITH document_results AS (
                SELECT
                    id,
                    media_record AS "mediaRecordId",
                    'document' AS source,
                    page,
                    NULL::integer AS "startTime",
                    text,
                    NULL::text AS "screenText",
                    NULL::text AS transcript
                FROM document_sections
                WHERE id = ANY(%(segmentIds)s)
            ),
            video_results AS (
                SELECT 
                    id,
                    media_record AS "mediaRecordId",
                    'video' AS source,
                    NULL::integer AS page,
                    start_time AS "startTime",
                    NULL::text AS text,
                    screen_text AS "screenText",
                    transcript
                FROM video_sections
                WHERE id = ANY(%(segmentIds)s)
            ),
            results AS (
                SELECT * FROM document_results
                UNION ALL
                SELECT * FROM video_results
            )
            SELECT * FROM results;
        """, {"segmentIds": all_segment_ids}).fetchall()

        for x in result_segments:
            if x["source"] == "document":
                del x["startTime"]
                del x["screenText"]
                del x["transcript"]
            elif x["source"] == "video":
                del x["page"]
                del x["text"]

        return [{
                    "segment1": next(x for x in result_segments if x["id"] == segment_link["segment1Id"]),
                    "segment2": next(x for x in result_segments if x["id"] == segment_link["segment2Id"])
                } for segment_link in result]

    def get_media_record_segments(self, media_record_id: uuid.UUID):
        query = """
            WITH document_results AS (
                SELECT
                    id,
                    media_record AS "mediaRecordId",
                    'document' AS source,
                    page,
                    NULL::integer AS "startTime",
                    text,
                    NULL::text AS "screenText",
                    NULL::text AS transcript
                FROM document_sections
                WHERE media_record = %(id)s
            ),
            video_results AS (
                SELECT 
                    id,
                    media_record AS "mediaRecordId",
                    'video' AS source,
                    NULL::integer AS page,
                    start_time AS "startTime",
                    NULL::text AS text,
                    screen_text AS "screenText",
                    transcript
                FROM video_sections
                WHERE media_record = %(id)s
            ),
            results AS (
                SELECT * FROM document_results
                UNION ALL
                SELECT * FROM video_results
            )
            SELECT * FROM results
            """

        results = self._db_conn.execute(query, {"id": media_record_id}).fetchall()

        for result in results:
            if result["source"] == "document":
                del result["startTime"]
                del result["screenText"]
                del result["transcript"]
            elif result["source"] == "video":
                del result["page"]
                del result["text"]

        return results

    def get_media_record_captions(self, media_record_id: uuid.UUID) -> str:
        result = self._db_conn.execute("SELECT vtt FROM video_captions WHERE media_record_id = %s",
                                       (media_record_id,)).fetchone()
        return result["vtt"]

    def semantic_search(self,
                        query_text: str,
                        count: int,
                        media_record_blacklist: list[uuid.UUID],
                        media_record_whitelist: list[uuid.UUID]) -> list[dict[str, any]]:
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
                FROM document_sections
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
                FROM video_sections
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
            "mediaRecordBlacklist": media_record_blacklist,
            "mediaRecordWhitelist": media_record_whitelist
        }).fetchall()

        for result in query_result:
            if result["source"] == "document":
                del result["startTime"]
                del result["screenText"]
                del result["transcript"]
            elif result["source"] == "video":
                del result["page"]
                del result["text"]

        return [{
            "score": result["score"],
            "mediaRecordSegment": result,
        } for result in query_result]

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
