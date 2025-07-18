import asyncio
import io
import itertools
import logging
import threading
import time
import uuid
from time import sleep
from typing import Callable, Self, Awaitable, Optional

import PIL.Image
import psycopg

import client.MediaServiceClient as MediaServiceClient
import config
import dto
import dto.mapper as mapper
import fileextractlib.LLMService as LlamaRunner
from fileextractlib.TopicModel import TopicModel
from dto import MediaRecordSegmentLinkDto, SemanticSearchResultDto, \
    AiEntityProcessingProgressDto, MediaRecordSegmentDto, TaskInformationDto
from fileextractlib.DocumentProcessor import DocumentProcessor
from fileextractlib.ImageTemplateMatcher import ImageTemplateMatcher
from fileextractlib.LectureDocumentEmbeddingGenerator import LectureDocumentEmbeddingGenerator
from fileextractlib.LectureVideoEmbeddingGenerator import LectureVideoEmbeddingGenerator
from fileextractlib.SentenceEmbeddingRunner import SentenceEmbeddingRunner
from fileextractlib.VideoProcessor import VideoProcessor
from persistence.IngestionStateDbConnector import IngestionStateDbConnector
from persistence.MediaRecordInfoDbConnector import MediaRecordInfoDbConnector
from persistence.SegmentDbConnector import SegmentDbConnector
from persistence.AssesmentInfoDbConnector import AssessmentInfoDbConnector
from persistence.entities import *
from utils.SortedPriorityQueue import SortedPriorityQueue
from controller.events import ContentChangeEvent, CrudOperation
from numpy.typing import NDArray
import numpy as np

_logger = logging.getLogger(__name__)


class DocProcAiService:
    def __init__(self):
        self.database_connection = psycopg.connect(
            config.current["database"]["connection_string"],
            autocommit=True,
            row_factory=psycopg.rows.dict_row
        )

        self.segment_database = SegmentDbConnector(self.database_connection)
        self.media_record_info_database = MediaRecordInfoDbConnector(self.database_connection)
        self.ingestion_state_database = IngestionStateDbConnector(self.database_connection)
        self.assesment_database = AssessmentInfoDbConnector(self.database_connection)

        # graphql client for interacting with the media service
        self.__media_service_client: MediaServiceClient.MediaServiceClient = MediaServiceClient.MediaServiceClient()

        self.__sentence_embedding_runner: SentenceEmbeddingRunner = SentenceEmbeddingRunner()

        self.__lecture_pdf_embedding_generator: LectureDocumentEmbeddingGenerator = LectureDocumentEmbeddingGenerator()
        self.__lecture_video_embedding_generator: LectureVideoEmbeddingGenerator = LectureVideoEmbeddingGenerator()

        # only import & create a llm generator object if llm generation is enabled in the config
        if (config.current["lecture_llm_generator"]["segment_title_generator"]["enabled"] or
                config.current["lecture_llm_generator"]["document_summary_generator"]["enabled"]):
            from fileextractlib.LectureLlmGenerator import LectureLlmGenerator
            self.__lecture_llm_generator: LectureLlmGenerator = LectureLlmGenerator()

        # the "queue" we use to keep track of which items
        self._background_task_queue: SortedPriorityQueue[DocProcAiService.BackgroundTaskItem] = SortedPriorityQueue()

        self._keep_background_task_thread_alive: threading.Event = threading.Event()
        self._keep_background_task_thread_alive.set()

        self.__background_task_thread: threading.Thread = threading.Thread(
            target=self._background_task_runner,
            args=[])
        self.__background_task_thread.start()

    def __del__(self):
        self._keep_background_task_thread_alive.clear()

    def enqueue_ingest_media_record_task(self, media_record_id: uuid.UUID) -> None:
        """
        Enqueues a task to ingest a media record with the given ID, which will be executed in the background.
        """

        async def ingest_media_record_task():
            try:
                media_record = await self.__media_service_client.get_media_record_type_and_download_url(media_record_id)
                download_url = media_record["internalDownloadUrl"]

                record_type: str = media_record["type"]

                _logger.info("Ingesting media record with download URL: " + download_url)
                self.ingestion_state_database.upsert_entity_ingestion_info(media_record_id,
                                                                           IngestionEntityTypeDbType.MEDIA_RECORD,
                                                                           IngestionStateDbType.PROCESSING)

                self.segment_database.delete_document_segments_by_media_record_id([media_record_id])
                self.segment_database.delete_video_segments_by_media_record_id([media_record_id])

                _logger.info("Processing file of Type: " + record_type)
                if record_type == "PRESENTATION" or record_type == "DOCUMENT":
                    _logger.info("Starting document processor for " + str(media_record_id))
                    document_processor = DocumentProcessor()
                    document_data = document_processor.process(download_url)
                    _logger.info("Generating embeddings for " + str(media_record_id))
                    self.__lecture_pdf_embedding_generator.generate_embeddings(document_data.pages)
                    for segment in document_data.pages:
                        thumbnail_bytes = io.BytesIO()
                        segment.thumbnail.save(thumbnail_bytes, format="JPEG", quality=93)
                        self.segment_database.insert_document_segment(segment.text,
                                                                      media_record_id,
                                                                      segment.page_number,
                                                                      thumbnail_bytes.getvalue(),
                                                                      None,
                                                                      segment.embedding)

                    if config.current["lecture_llm_generator"]["document_summary_generator"]["enabled"]:
                        # generate and store a summary of this media record
                        _logger.info("Generating summary for " + str(media_record_id))
                        self.__lecture_llm_generator.generate_summary_for_document(document_data)

                    self.media_record_info_database.upsert_media_record_info(media_record_id, document_data.summary, None)
                elif record_type == "VIDEO":
                    _logger.info("Starting video processor for " + str(media_record_id))
                    video_processor = VideoProcessor(
                        segment_image_similarity_threshold=
                        config.current["video_segmentation"]["segment_image_similarity_threshold"],
                        minimum_segment_length=config.current["video_segmentation"]["minimum_segment_length"])
                    video_data = video_processor.process(download_url)
                    del video_processor

                    # generate text embeddings for the segments of the video
                    _logger.info("Generating embeddings for " + str(media_record_id))
                    self.__lecture_video_embedding_generator.generate_embeddings(video_data.segments)

                    # generate titles for the video's segments if llm features enabled
                    if config.current["lecture_llm_generator"]["segment_title_generator"]["enabled"]:
                        _logger.info("Generating title for " + str(media_record_id))
                        self.__lecture_llm_generator.generate_titles_for_video(video_data)
                    else:
                        # otherwise set empty data/placeholders
                        _logger.info("LLM generator disabled. Setting placeholders.")
                        video_data.summary = []
                        for i, segment in enumerate(video_data.segments, start=1):
                            segment.title = "Section " + str(i)

                    for segment in video_data.segments:
                        thumbnail_bytes = io.BytesIO()
                        segment.thumbnail.save(thumbnail_bytes, format="JPEG", quality=93)
                        self.segment_database.insert_video_segment(segment.screen_text, segment.transcript, media_record_id,
                                                                   segment.start_time, thumbnail_bytes.getvalue(),
                                                                   segment.title, segment.embedding)

                    # store media record-level data: summary & closed captions vtt string
                    self.media_record_info_database.upsert_media_record_info(media_record_id,
                                                                             video_data.summary,
                                                                             video_data.vtt.content)
                else:
                    raise ValueError("Asked to ingest unsupported media record type of type " + media_record["type"])

                try:
                    self.__generate_tags()
                except ValueError as e:
                    _logger.exception(e)

                self.ingestion_state_database.upsert_entity_ingestion_info(media_record_id,
                                                                           IngestionEntityTypeDbType.MEDIA_RECORD,
                                                                           IngestionStateDbType.DONE)

                _logger.info("Finished ingesting media record with download URL: " + download_url)
            except Exception as e:
                _logger.exception("Error while processing ingest_media_record queue task", e)

        priority = 0
        self.ingestion_state_database.upsert_entity_ingestion_info(media_record_id,
                                                                   IngestionEntityTypeDbType.MEDIA_RECORD,
                                                                   IngestionStateDbType.ENQUEUED)
        self._background_task_queue.put(DocProcAiService.BackgroundTaskItem(media_record_id,
                                                                            ingest_media_record_task,
                                                                            priority))

    def __generate_tags(self):
        """
        Generates the suggested tags for all media records and assessments. This will recreate all suggested tags
        """
        segments = self.segment_database.get_all_entity_segments()

        topic_model = TopicModel(segments)

        _logger.info("Running topic model")
        topic_model.create_topic_model()
        _logger.info("Finished running topic model")
        self.__generate_tags_for_media_records(segments, topic_model)
        self.__generate_tags_for_assessments(segments, topic_model)

    def __generate_tags_for_media_records(self, segments, topic_model):
        """
        Generates the suggested tags for all media records. This will recreate all suggested tags.
        This step will be skipped if no media records are found.
        """
        _logger.info("Generating tags for media records.")
        media_records = self.media_record_info_database.get_all_media_records()
        if not media_records: # check if media_records is empty
            _logger.info("No media records found, skipping step")
            return

        media_records_with_tags = topic_model.add_tags_to_media_records(segments)
        if media_records_with_tags is not None:
            for media_record_id, tags in media_records_with_tags.items():
                self.media_record_info_database.update_media_record_tags(media_record_id, list(tags))
            _logger.info("Generated tags for media records.")

    def __generate_tags_for_assessments(self, segments, topic_model):
        """
        Generates the suggested tags for all assessments. This will recreate all suggested tags.
        This step will be skipped if no assessments are found.
        """
        _logger.info("Generating tags for assesments.")
        assesments = self.assesment_database.get_all_assessments()
        if not assesments: # check if assessments is empty
            _logger.info("No assessments found, skipping step")
            return

        assesments_with_tags = topic_model.add_tags_to_assessments(segments)

        if assesments_with_tags is not None:
            for assesment_id, tags in assesments_with_tags.items():
                self.assesment_database.update_assessment_tags(assesment_id, list(tags))
            _logger.info("Generated tags for assessments.")


    def enqueue_generate_assessment_segments(self,
                                             assessment_id: uuid.UUID,
                                             task_information_list: list[TaskInformationDto]) -> None:
        async def generate_assessment_segments_task() -> None:
            try:
                _logger.info("Ingesting assessment with id " + str(assessment_id))
                self.ingestion_state_database.upsert_entity_ingestion_info(assessment_id,
                                                                           IngestionEntityTypeDbType.ASSESSMENT,
                                                                           IngestionStateDbType.PROCESSING)

                self.segment_database.delete_assessment_segments_by_assessment_id(assessment_id)

                sentence_embedding_runner: SentenceEmbeddingRunner = SentenceEmbeddingRunner()

                for task_information in task_information_list:
                    embedding: NDArray[np.float_] = sentence_embedding_runner.generate_embeddings(
                        [task_information["textualRepresentation"]])[0]

                    self.segment_database.upsert_assessment_segment(task_information["taskId"],
                                                                    assessment_id,
                                                                    task_information["textualRepresentation"],
                                                                    embedding)

                    self.assesment_database.upsert_assessment_info(assessment_id)

                try:
                    self.__generate_tags()
                except ValueError as e:
                    _logger.exception(e)

                self.ingestion_state_database.upsert_entity_ingestion_info(assessment_id,
                                                                           IngestionEntityTypeDbType.ASSESSMENT,
                                                                           IngestionStateDbType.DONE)
                _logger.info("Finished ingesting assessment with id " + str(assessment_id))
            except Exception as e:
                _logger.exception("Error while processing generate_assessment_segments queue task", e)
        priority = 2
        self.ingestion_state_database.upsert_entity_ingestion_info(assessment_id,
                                                                   IngestionEntityTypeDbType.ASSESSMENT,
                                                                   IngestionStateDbType.ENQUEUED)
        self._background_task_queue.put(DocProcAiService.BackgroundTaskItem(assessment_id,
                                                                            generate_assessment_segments_task,
                                                                            priority))

    def enqueue_generate_content_media_record_links(self, content_id: uuid.UUID) -> None:
        """
        Enqueues a task to generate media record links for a content with the given ID, which will be executed in the
        background.
        """

        async def generate_content_media_record_links_task() -> None:
            """
            Function which performs the media record linking. Executed as a task asynchronously.
            """
            try:
                _logger.info("Generating content media record links for content " + str(content_id) + "...")
                self.ingestion_state_database.upsert_entity_ingestion_info(content_id,
                                                                           IngestionEntityTypeDbType.MEDIA_CONTENT,
                                                                           IngestionStateDbType.PROCESSING)

                start_time = time.time()

                self.segment_database.delete_media_record_segment_links_by_content_ids([content_id])

                # get the ids of the media records which are part of this content
                media_record_ids = await self.__media_service_client.get_media_record_ids_of_contents([content_id])

                # we could first run a query to check if any records even match, but considering that linking usually
                # only happens after ingestion, we can assume that the records exist, so that would be an unnecessary
                # query
                # From the returned list, create a dict sorted by which segment is associated with which media record
                media_records_segments: dict[uuid.UUID, list[DocumentSegmentEntity | VideoSegmentEntity]] = {}
                for result in self.segment_database.get_media_record_segments_by_media_record_ids(media_record_ids):
                    if result.media_record_id not in media_records_segments:
                        media_records_segments[result.media_record_id] = []
                    media_records_segments[result.media_record_id].append(result)

                # now we can check for links
                # go through each media record's segments
                linking_results: dict[UUID, UUID] = {}
                threads: list[threading.Thread] = []
                for media_record_id, segments in reversed(media_records_segments.items()):
                    for segment in segments:
                        # only process video segments
                        if not isinstance(segment, VideoSegmentEntity):
                            continue

                        threads.append(
                            threading.Thread(target=DocProcAiService.__match_segment_against_other_media_records,
                                             args=(linking_results, media_record_id, segment, media_records_segments)))

                max_threads: int = config.current["content_linking"]["linking_processing_max_threads"]
                while len(threads) > 0:
                    # take the first max_threads threads and start them (or less if there are less than max_threads)
                    current_threads = threads[:max_threads]
                    threads = threads[max_threads:]

                    for thread in current_threads:
                        thread.start()

                    for thread in current_threads:
                        thread.join()

                for (segment1_id, segment2_id) in linking_results.items():
                    if not self.does_link_between_media_record_segments_exist(segment1_id,
                                                                              segment2_id,
                                                                              content_id):
                        self.segment_database.insert_media_record_segment_link(content_id,
                                                                               segment1_id,
                                                                               segment2_id)

                self.ingestion_state_database.upsert_entity_ingestion_info(content_id,
                                                                           IngestionEntityTypeDbType.MEDIA_CONTENT,
                                                                           IngestionStateDbType.DONE)
                _logger.info("Generated content media record links for content "
                             + str(content_id) + " in " + str(time.time() - start_time) + " seconds.")
            except Exception as e:
                _logger.exception("Error while processing generate_content_media_record_links queue task", e)

        # priority of media record link generation needs to be higher than that of media record ingestion (higher
        # priority items are processed last), so that the media records which are being linked have been processed
        # before being linked
        self.ingestion_state_database.upsert_entity_ingestion_info(content_id,
                                                                   IngestionEntityTypeDbType.MEDIA_CONTENT,
                                                                   IngestionStateDbType.ENQUEUED)
        self._background_task_queue.put(
            DocProcAiService.BackgroundTaskItem(content_id, generate_content_media_record_links_task, priority=1))

    def create_link_between_media_record_segments(self,
                                                  content_id: uuid.UUID,
                                                  segment_id: uuid.UUID,
                                                  other_segment_id: uuid.UUID):
        """
        Creates a link between two media record segments which are associated (e.g. part of a video and the respective
        slide in the PDF).
        """
        self.segment_database.insert_media_record_segment_link(content_id, segment_id, other_segment_id)

    def does_link_between_media_record_segments_exist(self,
                                                      segment_id: uuid.UUID,
                                                      other_segment_id: uuid.UUID,
                                                      content_id: uuid.UUID) -> bool:
        """
        Checks if a link between two media record segments exists by their IDs.
        """
        return self.segment_database.does_segment_link_exist(segment_id, other_segment_id, content_id)

    def delete_entries_of_media_record(self, media_record_id: uuid.UUID):
        """
        Deletes all entries this service's db keeps which are associated with the specified media record.
        This includes the segment links and the segments of the media record themselves.
        """
        # delete segments associated with this media record
        segment_ids = list(itertools.chain(
            [x.id for x in self.segment_database.delete_video_segments_by_media_record_id([media_record_id])],
            [x.id for x in self.segment_database.delete_document_segments_by_media_record_id([media_record_id])]
        ))

        # delete media record segment links which contain any of these segments
        self.segment_database.delete_media_record_segment_links_by_segment_ids(segment_ids)
        self.ingestion_state_database.delete_ingestion_state(media_record_id)
        self.media_record_info_database.delete_media_record_by_id(media_record_id)


    def delete_entries_of_assessments(self, content_change_event: ContentChangeEvent):
        """
        Deletes all entries this services db keeps which are associated with the specified assessment.
        """
        content_change_event = content_change_event

        for assessment_id in content_change_event.contentIds:
            self.segment_database.delete_assessment_segments_by_assessment_id(assessment_id)
            self.ingestion_state_database.delete_ingestion_state(assessment_id)
            self.assesment_database.delete_assessment_by_id(assessment_id)




    def get_media_record_links_for_content(self, content_id: uuid.UUID) -> list[MediaRecordSegmentLinkDto]:
        """
        Gets all links between media record segments which are part of the specified content.
        """
        result_links = self.segment_database.get_segment_links_by_content_id(content_id)

        all_segment_ids = []
        for segment_link in result_links:
            all_segment_ids.append(segment_link.segment1_id)
            all_segment_ids.append(segment_link.segment2_id)

        result_segments = self.segment_database.get_entity_segments_by_ids(all_segment_ids)

        # go over all links and resolve the referenced segments from entity to dto
        return [{
            "segment1": mapper.media_record_segment_entity_to_dto(
                next(x for x in result_segments if x.id == segment_link.segment1_id)),
            "segment2": mapper.media_record_segment_entity_to_dto(
                next(x for x in result_segments if x.id == segment_link.segment2_id))
        } for segment_link in result_links]

    def get_media_record_segments(self, media_record_id: uuid.UUID) \
            -> list[MediaRecordSegmentDto]:
        """
        Gets the segments of the specified media record.
        """
        results = self.segment_database.get_media_record_segments_by_media_record_ids([media_record_id])

        return [mapper.media_record_segment_entity_to_dto(result) for result in results]

    def get_media_record_segment_by_id(self, media_record_segment_id: uuid.UUID):
        """
        Gets the media record segment with the specified id or throws an error if it does not exist.
        :param media_record_segment_id: ID of the media record segment to return.
        :return: The media record segment with the specified ID.
        """
        query_results = self.segment_database.get_entity_segments_by_ids([media_record_segment_id])

        if len(query_results) != 1:
            raise ValueError("Media record segment with specified ID does not exist.")

        return mapper.media_record_segment_entity_to_dto(query_results[0])

    def get_media_record_captions(self, media_record_id: uuid.UUID) -> str | None:
        """
        Returns the captions of the specified media record as a string in WebVTT format if the specified media record
        is a video and captions are available. Otherwise, returns None.
        """
        return self.media_record_info_database.get_video_captions_by_media_record_id(media_record_id)

    def get_media_record_summary(self, media_record_id: uuid.UUID) -> list[str]:
        """
        Returns a summary of the media record's contents as a list of strings which are bullet points
        :param media_record_id: The id of the media record to get a summary for
        :return: List of strings, where each string is a bullet point of the summary
        """
        return self.media_record_info_database.get_media_record_summary_by_media_record_id(media_record_id)

    def get_media_record_tags(self, media_record_id: uuid.UUID) -> list[str]:
        """
        Returns the auto generated tags of the specified media record as a list.
        :param media_record_id: The ID of the media record
        :return: List containing the tags
        """

        return self.media_record_info_database.get_media_record_tags_by_media_record_id(media_record_id)

    def get_assessment_tags(self, assessment_id: uuid.UUID) -> list[str]:
        """
        Returns the auto generated tags of the specified media record as a list.
        :param assessment_id: The ID of the media record
        :return: List containing the tags
        """

        return self.assesment_database.get_assessment_tags_by_id(assessment_id)

    def get_entities_ai_processing_state(self, entity_ids: list[uuid.UUID]) -> list[AiEntityProcessingProgressDto]:
        """
        For the entities with the specified IDs, gets their AI processing progress and returns a list containing their
        progress in the order of the passed entity ids.
        :param entity_ids: The IDs of the entities to get processing progress for.
        :return: List containing processing progress for the specified entities in the same order.
        """
        query_results: list[EntityIngestionInfoEntity] = (
            self.ingestion_state_database.get_entities_ingestion_info(entity_ids))

        results: list[AiEntityProcessingProgressDto] = []

        for entity_id in entity_ids:
            query_result: Optional[EntityIngestionInfoEntity] \
                = next((x for x in query_results if x.entity_id == entity_id), None)

            if query_result is None:
                results.append({
                    "entityId": entity_id,
                    "state": dto.AiEntityProcessingStateDto.UNKNOWN,
                    "queuePosition": None
                })
                continue

            match query_result.ingestion_state:
                case IngestionStateDbType.PROCESSING:
                    state = dto.AiEntityProcessingStateDto.PROCESSING
                case IngestionStateDbType.ENQUEUED:
                    state = dto.AiEntityProcessingStateDto.ENQUEUED
                case IngestionStateDbType.DONE:
                    state = dto.AiEntityProcessingStateDto.DONE
                case _:
                    state = dto.AiEntityProcessingStateDto.UNKNOWN

            try:
                queue_position = self._background_task_queue.first_index_satisfying_predicate(
                    lambda x: x.entity_id == entity_id)
            except ValueError:
                # raised when element not in queue
                queue_position = None

            results.append({
                "entityId": entity_id,
                "state": state,
                "queuePosition": queue_position
            })
        return results

    async def semantic_search(self,
                        query_text: str,
                        count: int,
                        content_whitelist: list[UUID]) -> list[SemanticSearchResultDto]:
        """
        Performs a semantic search on the specified query text. Returns the specified number of media record segments.
        Adheres to the passed black- & whitelist. Segments of media records whose ID is present in the blacklist OR
        whose ID is NOT present in the whitelist will be excluded from the results.

        :param query_text: String search query using which the semantic search is performed
        :param count: Number of returned search results
        :param content_whitelist: Whitelist of IDs of contents whose segments should be included from the search results
        :return: List of search results
        """
        start_time = time.time()
        query_embedding = self.__sentence_embedding_runner.generate_embeddings([query_text])[0]
        stop_time = time.time()
        _logger.info("Generating Embeddings: " + str(stop_time - start_time) + " seconds.")

        start_time = time.time()
        parent_whitelist = await self.__generate_segment_parent_whitelist(content_whitelist)
        stop_time = time.time()
        _logger.info("Generating Segment parent whitelist: " + str(stop_time - start_time) + " seconds.")

        start_time = time.time()
        query_result = self.segment_database.get_top_segments_by_embedding_distance(query_embedding,
                                                                                    count,
                                                                                    parent_whitelist)
        stop_time = time.time()
        _logger.info("Get top segments from DB: " + str(stop_time - start_time) + " seconds.")

        return [mapper.semantic_search_result_entity_to_dto(result) for result in query_result]

    async def get_semantically_similar_entities(self, entity_id: UUID, count: int, content_whitelist: list[UUID],
                                                excludeEntitiesWithSameParent: bool) \
            -> list[SemanticSearchResultDto]:
        """
        Performs a semantic similarity search where the entities are returned which are the most semantically similar
        to the provided entity.

        :param entity_id: ID of the entity for which to search similar segments.
        :param count: Number of returned search results
        :param content_whitelist: Whitelist of content ids whose media records & segments should appear in the search
        results
        :return: List of search results
        """

        # firstly, we need to get the entity segment we want to find similar ones for
        segment_query_res = self.segment_database.get_entity_segments_by_ids([entity_id])

        if len(segment_query_res) < 1:
            raise ValueError("Could not find segment with ID " + str(entity_id))

        entity_segment = segment_query_res[0]

        parent_whitelist = await self.__generate_segment_parent_whitelist(content_whitelist)

        if excludeEntitiesWithSameParent:
            if isinstance(entity_segment, AssessmentSegmentEntity):
                parent_id = entity_segment.assessment_id
            elif isinstance(entity_segment, DocumentSegmentEntity):
                parent_id = entity_segment.media_record_id
            elif isinstance(entity_segment, VideoSegmentEntity):
                parent_id = entity_segment.media_record_id
            else:
                raise NotImplementedError("Tried to fetch parent id of unknown entity segment type.")

            parent_whitelist = [x for x in parent_whitelist if x != parent_id]

        # Fetch one more result than "count", because results will include the segment we're comparing to itself!
        query_result = self.segment_database.get_top_segments_by_embedding_distance(entity_segment.embedding,
                                                                                    count + 1,
                                                                                    parent_whitelist)

        results = [mapper.semantic_search_result_entity_to_dto(result) for result in query_result]

        # The result which contains the segment we were using as a base itself will have distance 0 to itself (duh)
        # remove it
        return [result for result in results if result["score"] > 0]

    def _ensure_processing_queue_in_consistent_state(self) -> None:
        """
        Helper method to ensure that the processing queue is in a consistent state.

        WARNING: Should only be executed when no item is currently being processed (items being in the queue is fine).
        """
        entities_in_queue = self.ingestion_state_database.get_enqueued_or_processing_ingestion_entities()

        def enqueue_entity(entity_id: UUID, entity_type: IngestionEntityTypeDbType):
            if entity_type == IngestionEntityTypeDbType.MEDIA_CONTENT:
                self.enqueue_generate_content_media_record_links(entity_id)
            elif entity_type == IngestionEntityTypeDbType.MEDIA_RECORD:
                self.enqueue_ingest_media_record_task(entity_id)

        for (entity_id, entity_type, entity_state) in entities_in_queue:
            if entity_state == IngestionStateDbType.PROCESSING:
                # No item should currently be in processing, so this is a wrong state
                # Enqueue it again so it gets processed
                self.ingestion_state_database.upsert_entity_ingestion_info(entity_id,
                                                                           entity_type,
                                                                           IngestionStateDbType.ENQUEUED)
                enqueue_entity(entity_id, entity_type)
            elif entity_state == IngestionStateDbType.ENQUEUED:
                # Ensure that entities which are listed as enqueued are actually in the processing queue
                # if not, add them
                try:
                    self._background_task_queue.first_index_satisfying_predicate(lambda x: x.entity_id == entity_id)
                except ValueError:
                    # A value error is raised when the item could not be found in the queue, re-queue it in that case
                    enqueue_entity(entity_id, entity_type)

    async def __generate_segment_parent_whitelist(self, content_whitelist: list[UUID]) -> list[UUID]:
        """
        Helper function which creates a whitelist of parent IDs for segments from a content whitelist. For media record
        segments, the parents are the media records, for assessment segments, the parent IDs are the content ids
        (assessment IDs and content IDs are identical). This means we firstly initialize the parent_whitelist with
        the UUIDs of the whitelisted contents, and then also fetch all IDs of media records which are children of
        the contents.
        """
        parent_whitelist: list[UUID] = list(content_whitelist)
        parent_whitelist.extend(await self.__media_service_client.get_media_record_ids_of_contents(content_whitelist))
        return parent_whitelist

    class BackgroundTaskItem:
        """
        Helper class used by the internal background task queue of the service.
        """
        def __init__(self, entity_id: UUID, task: Callable[[], Awaitable[None]], priority: int):
            self.entity_id = entity_id
            self.task: Callable[[], Awaitable[None]] = task
            self.priority: int = priority

        def __lt__(self, other: Self):
            return self.priority < other.priority

    def _background_task_runner(self):
        """
        Runner function which executes tasks from the task queue in the background.
        """
        while self._keep_background_task_thread_alive.is_set():
            self._ensure_processing_queue_in_consistent_state()

            if len(self._background_task_queue) == 0:
                sleep(1)
                continue

            background_task_item = self._background_task_queue.get()

            asyncio.run(background_task_item.task())


    @staticmethod
    def __match_segment_against_other_media_records(linking_results: dict[UUID, UUID],
                                                    media_record_id,
                                                    segment,
                                                    media_records_segments) -> None:
        """
        Helper function which matches a segment against the segments of other media records to find the most similar
        segment. If a segment is found which is similar enough, a link is added to the linking_results dictionary.
        """
        # get the text of the segment
        segment_thumbnail = PIL.Image.open(io.BytesIO(segment.thumbnail))

        cropped_segment_thumbnail = segment_thumbnail.crop((
            segment_thumbnail.width * 1 / 4, segment_thumbnail.height * 1 / 10,
            segment_thumbnail.width * 3 / 4, segment_thumbnail.height * 9 / 10))

        image_template_matcher = ImageTemplateMatcher(
            template=cropped_segment_thumbnail,
            scaling_factor=config.current["content_linking"]["linking_image_scaling_factor"],
            enable_multi_scale_matching=True,
            multi_scale_matching_steps=config.current["content_linking"]["linking_image_similarity_steps"]
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
                similarity = image_template_matcher.match(other_segment_thumbnail)
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similarity_segment_id = other_segment.id

            if max_similarity > config.current["content_linking"]["linking_image_similarity_threshold"]:
                # add the link we discovered to the results dictionary
                linking_results[segment.id] = max_similarity_segment_id
