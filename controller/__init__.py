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

    @mutation.field("_internal_noauth_ingestMediaRecord")
    def _internal_noauth_ingest_media_record(parent, info, input):
        document_id: uuid.UUID = input["id"]
        service.enqueue_ingest_media_record_task(document_id)
        return document_id

    @query.field("semanticSearch")
    def semantic_search(parent, info, queryText: str, count: int, filteredDocuments: list[uuid.UUID], filterType: str):
        # TODO: Implement filtering
        return service.semantic_search(queryText, count)

    schema = ariadne.make_executable_schema(schema, query, mutation)
    app = ariadne.asgi.GraphQL(schema, debug=True)

    return app
