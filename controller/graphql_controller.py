from uuid import UUID

import ariadne
import ariadne.asgi
from fastapi import FastAPI

from dto import AiEntityProcessingProgressDto
from service.DocProcAiService import DocProcAiService
import dto

from util import does_dict_match_typed_dict


class GraphQLController:
    def __init__(self, app: FastAPI, ai_service: DocProcAiService):
        self.ai_service = ai_service

        schema = ariadne.load_schema_from_path("schema/")

        # List containing all ariadne graphql bindables and a helper function to add newly created objects to the list
        # objects representing graphql types need to be added to the bindables list, which is later passed to ariadne,
        # otherwise ariadne would be unable to discover them when it creates the executable graphql schema
        bindables = []
        def bindable(cls):
            bindables.append(cls)
            return cls

        bindable(ariadne.EnumType("AiEntityProcessingState", dto.AiEntityProcessingStateDto))

        mutation = bindable(ariadne.MutationType())
        @mutation.field("_internal_noauth_ingestMediaRecord")
        def _internal_noauth_ingest_media_record(parent, info, input) -> UUID:
            media_record_id: UUID = input["id"]
            ai_service.enqueue_ingest_media_record_task(media_record_id)
            return media_record_id

        @mutation.field("_internal_noauth_enqueueGenerateMediaRecordLinksForContent")
        def _internal_noauth_enqueue_generate_media_record_links_for_content(parent, info, input) -> UUID:
            content_id: UUID = input["contentId"]

            ai_service.enqueue_generate_content_media_record_links(content_id)
            return content_id

        query = bindable(ariadne.QueryType())
        @query.field("_internal_noauth_semanticSearch")
        def semantic_search(parent, info, queryText: str, count: int,
                            mediaRecordBlacklist: list[UUID], mediaRecordWhitelist: list[UUID]) \
                -> list[dto.SemanticSearchResultDto]:
            return ai_service.semantic_search(queryText, count, mediaRecordBlacklist, mediaRecordWhitelist)

        @query.field("_internal_noauth_getSemanticallySimilarMediaRecordSegments")
        def get_semantically_similar_media_record_segments(parent, info, mediaRecordSegmentId: UUID, count: int,
                            mediaRecordBlacklist: list[UUID], mediaRecordWhitelist: list[UUID]) \
                -> list[dto.SemanticSearchResultDto]:
            return ai_service.get_semantically_similar_media_record_segments(mediaRecordSegmentId, count,
                                                                             mediaRecordBlacklist, mediaRecordWhitelist)

        @query.field("_internal_noauth_getMediaRecordLinksForContent")
        def get_media_record_links_for_content(parent, info, contentId: UUID) \
                -> list[dto.MediaRecordSegmentLinkDto]:
            return ai_service.get_media_record_links_for_content(contentId)

        @query.field("_internal_noauth_getMediaRecordSegments")
        def get_media_record_segments(parent, info, mediaRecordId: UUID) \
                -> list[dto.VideoRecordSegmentDto | dto.DocumentRecordSegmentDto]:
            return ai_service.get_media_record_segments(mediaRecordId)

        @query.field("_internal_noauth_getMediaRecordSegmentById")
        def get_media_record_segment_by_id(parent, info, mediaRecordSegmentId: UUID) \
                -> dto.VideoRecordSegmentDto | dto.DocumentRecordSegmentDto:
            return ai_service.get_media_record_segment_by_id(mediaRecordSegmentId)

        @query.field("_internal_noauth_getMediaRecordCaptions")
        def get_media_record_captions(parent, info, mediaRecordId: UUID) -> str | None:
            return ai_service.get_media_record_captions(mediaRecordId)

        @query.field("_internal_noauth_getMediaRecordSummary")
        def get_media_record_summary(parent, info, mediaRecordId: UUID) -> list[str]:
            return ai_service.get_media_record_summary(mediaRecordId)

        @query.field("_internal_noauth_getMediaRecordSuggestedTags")
        def get_media_record_suggested_tags(parent, info, mediaRecordId: UUID) -> list[str]:
            return ai_service.get_media_record_tags(mediaRecordId)

        @query.field("_internal_noauth_getMediaRecordsAiProcessingProgress")
        def get_media_records_ai_processing_state(parent, info, mediaRecordIds: list[UUID])\
                -> list[AiEntityProcessingProgressDto]:
            return ai_service.get_entities_ai_processing_state(mediaRecordIds)

        @query.field("_internal_noauth_getContentsAiProcessingProgress")
        def get_contents_ai_processing_state(parent, info, contentIds: list[UUID]) \
                -> list[AiEntityProcessingProgressDto]:
            return ai_service.get_entities_ai_processing_state(contentIds)

        media_record_segment_interface = bindable(ariadne.InterfaceType("MediaRecordSegment"))
        @media_record_segment_interface.type_resolver
        def resolve_semantic_search_result_type(obj, *_):
            if does_dict_match_typed_dict(obj, dto.DocumentRecordSegmentDto):
                return "DocumentRecordSegment"
            elif does_dict_match_typed_dict(obj, dto.VideoRecordSegmentDto):
                return "VideoRecordSegment"
            else:
                raise ValueError("Could not resolve source type of MediaRecordSegment interface. Object does not match"
                                 + " any known definitions", obj)

        uuid_scalar = bindable(ariadne.ScalarType("UUID"))
        @uuid_scalar.serializer
        def serialize_uuid(value):
            return str(value)

        @uuid_scalar.value_parser
        def parse_uuid_value(value):
            return UUID(value)

        schema = ariadne.make_executable_schema(schema,
                                                bindables)
        controller = ariadne.asgi.GraphQL(schema, debug=True)
        app.mount("/graphql", controller)

