import uuid

import ariadne
import ariadne.asgi
import uvicorn
import service.DocProcAiService as DocProcAiService


def create_app():
    service = DocProcAiService.DocProcAiService()

    schema = ariadne.load_schema_from_path("schema/")

    query = ariadne.QueryType()
    mutation = ariadne.MutationType()
    semantic_search_result_interface = ariadne.InterfaceType("SemanticSearchResult")
    uuid_scalar = ariadne.ScalarType("UUID")

    @mutation.field("_internal_noauth_ingestMediaRecord")
    def _internal_noauth_ingest_media_record(parent, info, input):
        document_id: uuid.UUID = input["id"]
        service.enqueue_ingest_media_record_task(document_id)
        return document_id

    @query.field("_internal_noauth_semanticSearch")
    def semantic_search(parent, info, queryText: str, count: int,
                        mediaRecordBlacklist: list[uuid.UUID], mediaRecordWhitelist: list[uuid.UUID]):
        # TODO: Implement filtering
        return service.semantic_search(queryText, count, mediaRecordBlacklist, mediaRecordWhitelist)

    @semantic_search_result_interface.type_resolver
    def resolve_semantic_search_result_type(obj, *_):
        if obj["source"] == "document":
            return "SemanticSearchDocumentResult"
        elif obj["source"] == "video":
            return "SemanticSearchVideoResult"
        else:
            raise ValueError("Unknown source type: " + obj["source"])

    @uuid_scalar.serializer
    def serialize_uuid(value):
        return str(value)

    @uuid_scalar.value_parser
    def parse_uuid_value(value):
        return uuid.UUID(value)

    schema = ariadne.make_executable_schema(schema, query, mutation, semantic_search_result_interface, uuid_scalar)
    app = ariadne.asgi.GraphQL(schema, debug=True)

    return app
