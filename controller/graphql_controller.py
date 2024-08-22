import uuid

import ariadne
import ariadne.asgi
from fastapi import FastAPI

import service.DocProcAiService as DocProcAiService


class GraphQLController:
    def __init__(self, app: FastAPI, ai_service: DocProcAiService):
        self.ai_service = ai_service

        schema = ariadne.load_schema_from_path("schema/")

        query = ariadne.QueryType()
        mutation = ariadne.MutationType()
        media_record_segment_interface = ariadne.InterfaceType("MediaRecordSegment")
        uuid_scalar = ariadne.ScalarType("UUID")

        @mutation.field("_internal_noauth_ingestMediaRecord")
        def _internal_noauth_ingest_media_record(parent, info, input):
            media_record_id: uuid.UUID = input["id"]
            ai_service.enqueue_ingest_media_record_task(media_record_id)
            return media_record_id

        @mutation.field("_internal_noauth_enqueueGenerateMediaRecordLinksForContent")
        def _internal_noauth_enqueue_generate_media_record_links_for_content(parent, info, input):
            content_id: uuid.UUID = input["contentId"]
            media_record_ids: list[uuid.UUID] = [x for x in input["mediaRecordIds"]]

            ai_service.enqueue_generate_content_media_record_links(content_id, media_record_ids)
            return content_id

        @query.field("_internal_noauth_semanticSearch")
        def semantic_search(parent, info, queryText: str, count: int,
                            mediaRecordBlacklist: list[uuid.UUID], mediaRecordWhitelist: list[uuid.UUID]):
            # TODO: Implement filtering
            return ai_service.semantic_search(queryText, count, mediaRecordBlacklist, mediaRecordWhitelist)

        @query.field("_internal_noauth_getMediaRecordLinksForContent")
        def get_media_record_links_for_content(parent, info, contentId: uuid.UUID):
            return ai_service.get_media_record_links_for_content(contentId)

        @query.field("_internal_noauth_getMediaRecordSegments")
        def get_media_record_segments(parent, info, mediaRecordId: uuid.UUID):
            return ai_service.get_media_record_segments(mediaRecordId)

        @query.field("_internal_noauth_getMediaRecordCaptions")
        def get_media_record_captions(parent, info, mediaRecordId: uuid.UUID):
            return ai_service.get_media_record_captions(mediaRecordId)

        @media_record_segment_interface.type_resolver
        def resolve_semantic_search_result_type(obj, *_):
            if obj["source"] == "document":
                return "DocumentRecordSegment"
            elif obj["source"] == "video":
                return "VideoRecordSegment"
            else:
                raise ValueError("Unknown source type: " + obj["source"])

        @uuid_scalar.serializer
        def serialize_uuid(value):
            return str(value)

        @uuid_scalar.value_parser
        def parse_uuid_value(value):
            return uuid.UUID(value)

        schema = ariadne.make_executable_schema(schema,
                                                query,
                                                mutation,
                                                media_record_segment_interface,
                                                uuid_scalar)
        controller = ariadne.asgi.GraphQL(schema, debug=True)
        app.mount("/graphql", controller)
