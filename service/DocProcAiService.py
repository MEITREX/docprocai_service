import asyncio
import itertools

import PIL.Image
import fileextractlib.LlamaRunner as LlamaRunner
import threading
import queue
from typing import Callable, Self, Awaitable
import uuid
import client.MediaServiceClient as MediaServiceClient
from dto import MediaRecordSegmentLinkDto, DocumentRecordSegmentDto, VideoRecordSegmentDto, SemanticSearchResultDto
from fileextractlib.DocumentProcessor import DocumentProcessor
from fileextractlib.LectureDocumentEmbeddingGenerator import LectureDocumentEmbeddingGenerator
from fileextractlib.LectureLlmGenerator import LectureLlmGenerator
from fileextractlib.LectureVideoEmbeddingGenerator import LectureVideoEmbeddingGenerator
from fileextractlib.SentenceEmbeddingRunner import SentenceEmbeddingRunner
import logging
from fileextractlib.VideoProcessor import VideoProcessor
from fileextractlib.ImageTemplateMatcher import ImageTemplateMatcher
import io
import dto.mapper as mapper

from persistence.DbConnector import DbConnector
from persistence.entities import *

_logger = logging.getLogger(__name__)


class DocProcAiService:
    def __init__(self):
        # TODO: Make db connection configurable
        self.database = DbConnector("user=root password=root host=database-docprocai port=5432 dbname=search-service")

        # graphql client for interacting with the media service
        self.__media_service_client: MediaServiceClient.MediaServiceClient = MediaServiceClient.MediaServiceClient()

        # only load the llamaRunner the first time we actually need it, not now
        self.__llama_runner: LlamaRunner.LlamaRunner | None = None

        self.__sentence_embedding_runner: SentenceEmbeddingRunner = SentenceEmbeddingRunner()

        self.__lecture_pdf_embedding_generator: LectureDocumentEmbeddingGenerator = LectureDocumentEmbeddingGenerator()
        self.__lecture_video_embedding_generator: LectureVideoEmbeddingGenerator = LectureVideoEmbeddingGenerator()

        self.__lecture_llm_generator: LectureLlmGenerator = LectureLlmGenerator()

        self.__background_task_queue: queue.PriorityQueue[DocProcAiService.BackgroundTaskItem] = queue.PriorityQueue()

        self.__keep_background_task_thread_alive: threading.Event = threading.Event()
        self.__keep_background_task_thread_alive.set()

        self.__background_task_thread: threading.Thread = threading.Thread(
            target=_background_task_runner,
            args=[self.__background_task_queue, self.__keep_background_task_thread_alive])
        self.__background_task_thread.start()

    def __del__(self):
        self.__keep_background_task_thread_alive = False

    def enqueue_ingest_media_record_task(self, media_record_id: uuid.UUID):
        """
        Enqueues a task to ingest a media record with the given ID, which will be executed in the background.
        """
        async def ingest_media_record_task():
            media_record = await self.__media_service_client.get_media_record_type_and_download_url(media_record_id)
            download_url = media_record["internalDownloadUrl"]

            record_type: str = media_record["type"]

            _logger.info("Ingesting media record with download URL: " + download_url)

            if record_type == "PRESENTATION" or record_type == "DOCUMENT":
                document_processor = DocumentProcessor()
                document_data = document_processor.process(download_url)
                self.__lecture_pdf_embedding_generator.generate_embeddings(document_data.pages)
                for segment in document_data.pages:
                    thumbnail_bytes = io.BytesIO()
                    segment.thumbnail.save(thumbnail_bytes, format="JPEG", quality=93)
                    # TODO: Fill placeholder title
                    self.database.insert_document_segment(segment.text, media_record_id, segment.page_number,
                                                          thumbnail_bytes.getvalue(), "Placeholder Title", segment.embedding)

                # generate and store a summary of this media record
                self.__lecture_llm_generator.generate_summary_for_document(document_data)
                self.database.insert_media_record(media_record_id, document_data.summary)
            elif record_type == "VIDEO":
                # TODO: make this configurable
                video_processor = VideoProcessor(segment_image_similarity_threshold=0.9, minimum_segment_length=15)
                video_data = video_processor.process(download_url)
                del video_processor

                # store the captions of the video
                self.database.insert_video_captions(media_record_id, video_data.vtt.content)

                # generate text embeddings for the segments of the video
                self.__lecture_video_embedding_generator.generate_embeddings(video_data.segments)

                # generate titles for the video's segments
                self.__lecture_llm_generator.generate_titles_for_video(video_data)

                for segment in video_data.segments:
                    thumbnail_bytes = io.BytesIO()
                    segment.thumbnail.save(thumbnail_bytes, format="JPEG", quality=93)
                    self.database.insert_video_segment(segment.screen_text, segment.transcript, media_record_id,
                                                       segment.start_time, thumbnail_bytes.getvalue(),
                                                       segment.title, segment.embedding)

                # generate and store a summary of this media record
                self.__lecture_llm_generator.generate_summary_for_video(video_data)
                self.database.insert_media_record(media_record_id, video_data.summary)
            else:
                raise ValueError("Asked to ingest unsupported media record type of type " + media_record["type"])

            _logger.info("Finished ingesting media record with download URL: " + download_url)

        priority = 0
        self.__background_task_queue.put(DocProcAiService.BackgroundTaskItem(ingest_media_record_task, priority))

    def enqueue_generate_content_media_record_links(self, content_id: uuid.UUID, media_record_ids: list[uuid.UUID]):
        """
        Enqueues a task to generate media record links for a content with the given ID, which will be executed in the
        background.
        """
        def generate_content_media_record_links_task():
            _logger.debug("Generating content media record links for content " + str(content_id) + "...")

            # we could first run a query to check if any records even match, but considering that linking usually only
            # happens after ingestion, we can assume that the records exist, so that would be an unnecessary query
            # From the returned list, create a dict sorted by which segment is associated with which media record
            media_records_segments: dict[uuid.UUID, list[DocumentSegmentEntity | VideoSegmentEntity]] = {}
            for result in self.database.get_record_segments_by_media_record_ids(media_record_ids):
                if result.media_record_id not in media_records_segments:
                    media_records_segments[result.media_record_id] = []
                media_records_segments[result.media_record_id].append(result)

            # now we can check for links
            # go through each media record's segments
            for media_record_id, segments in reversed(media_records_segments.items()):
                for segment in segments:
                    # get the text of the segment
                    segment_thumbnail = PIL.Image.open(io.BytesIO(segment.thumbnail))

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
                            other_segment_thumbnail = PIL.Image.open(io.BytesIO(other_segment.thumbnail))
                            size_ratio = segment_thumbnail.height / other_segment_thumbnail.height
                            other_segment_thumbnail = other_segment_thumbnail.resize(
                                (int(other_segment_thumbnail.width * size_ratio), segment_thumbnail.height))

                            # use the image template matcher to try to match the thumbnails
                            # TODO: Only use center portion of image for matching
                            similarity = image_template_matcher.match(other_segment_thumbnail)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                max_similarity_segment_id = other_segment.id

                        # TODO: Make this configurable
                        if max_similarity > 0.75:
                            # create a link between the two segments
                            # skip if a link between these segments already exists
                            if not self.does_link_between_media_record_segments_exist(segment.id,
                                                                                      max_similarity_segment_id):
                                self.database.insert_media_record_segment_link(content_id,
                                                                               segment.id,
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
        """
        Creates a link between two media record segments which are associated (e.g. part of a video and the respective
        slide in the PDF).
        """
        self.database.insert_media_record_segment_link(content_id, segment_id, other_segment_id)

    def does_link_between_media_record_segments_exist(self, segment_id: uuid.UUID, other_segment_id: uuid.UUID) -> bool:
        """
        Checks if a link between two media record segments exists by their IDs.
        """
        return self.database.does_segment_link_exist(segment_id, other_segment_id)

    def delete_entries_of_media_record(self, media_record_id: uuid.UUID):
        """
        Deletes all entries this service's db keeps which are associated with the specified media record.
        This includes the segment links and the segments of the media record themselves.
        """
        # delete segments associated with this media record
        segment_ids = list(itertools.chain(
            [x.id for x in self.database.delete_video_segments_by_media_record_id([media_record_id])],
            [x.id for x in self.database.delete_document_segments_by_media_record_id([media_record_id])]
        ))

        # delete media record segment links which contain any of these segments
        self.database.delete_media_record_segment_links_by_segment_ids(segment_ids)

    def get_media_record_links_for_content(self, content_id: uuid.UUID) -> list[MediaRecordSegmentLinkDto]:
        """
        Gets all links between media record segments which are part of the specified content.
        """
        result_links = self.database.get_segment_links_by_content_id(content_id)

        all_segment_ids = []
        for segment_link in result_links:
            all_segment_ids.append(segment_link.segment1_id)
            all_segment_ids.append(segment_link.segment2_id)

        result_segments = self.database.get_record_segments_by_ids(all_segment_ids)

        # go over all links and resolve the referenced segments from entity to dto
        return [{
                    "segment1": mapper.media_record_segment_entity_to_dto(
                        next(x for x in result_segments if x.id == segment_link.segment1_id)),
                    "segment2": mapper.media_record_segment_entity_to_dto(
                        next(x for x in result_segments if x.id == segment_link.segment2_id))
                } for segment_link in result_links]

    def get_media_record_segments(self, media_record_id: uuid.UUID) \
            -> list[DocumentRecordSegmentDto | VideoRecordSegmentDto]:
        """
        Gets the segments of the specified media record.
        """
        results = self.database.get_record_segments_by_media_record_ids([media_record_id])

        return [mapper.media_record_segment_entity_to_dto(result) for result in results]

    def get_media_record_captions(self, media_record_id: uuid.UUID) -> str | None:
        """
        Returns the captions of the specified media record as a string in WebVTT format if the specified media record
        is a video and captions are available. Otherwise, returns None.
        """
        return self.database.get_video_captions_by_media_record_id(media_record_id)

    def get_media_record_summary(self, media_record_id: uuid.UUID) -> list[str]:
        """
        Returns a summary of the media record's contents as a list of strings which are bullet points
        :param media_record_id: The id of the media record to get a summary for
        :return: List of strings, where each string is a bullet point of the summary
        """
        return self.database.get_media_record_summary_by_media_record_id(media_record_id)

    def semantic_search(self,
                        query_text: str,
                        count: int,
                        media_record_blacklist: list[uuid.UUID],
                        media_record_whitelist: list[uuid.UUID]) -> list[SemanticSearchResultDto]:
        """
        Performs a semantic search on the specified query text. Returns the specified number of media record segments.
        Adheres to the passed black- & whitelist. Segments of media records whose ID is present in the blacklist OR
        whose ID is NOT present in the whitelist will be excluded from the results.
        """
        query_embedding = self.__sentence_embedding_runner.generate_embeddings([query_text])[0]

        query_result = self.database.get_top_record_segments_by_embedding_distance(query_embedding,
                                                                                   count,
                                                                                   media_record_blacklist,
                                                                                   media_record_whitelist)

        return [mapper.semantic_search_result_entity_to_dto(result) for result in query_result]

    """
    Helper class used by the internal background task queue of the service.
    """
    class BackgroundTaskItem:
        def __init__(self, task: Callable[[], Awaitable[None]], priority: int):
            self.task: Callable[[], Awaitable[None]] = task
            self.priority: int = priority

        def __lt__(self, other: Self):
            return self.priority < other.priority

def _background_task_runner(task_queue: queue.PriorityQueue[DocProcAiService.BackgroundTaskItem],
                            keep_alive: threading.Event):
    """
    Runner function which executes tasks from the task queue in the background.
    """
    while keep_alive.is_set():
        try:
            background_task_item = task_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        asyncio.run(background_task_item.task())
        task_queue.task_done()
