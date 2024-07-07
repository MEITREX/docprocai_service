import ariadne
import ariadne.asgi
import uvicorn


def create_app():
    schema = ariadne.load_schema_from_path("schema/")

    query = ariadne.QueryType()
    mutation = ariadne.MutationType()

    @mutation.field("_internal_noauth_ingestDocument")
    def _internal_noauth_ingestDocument(parent, info, input):
        pass

    @mutation.field("_internal_noauth_ingestVideo")
    def _internal_noauth_ingestVideo(parent, info, input):
        pass

    schema = ariadne.make_executable_schema(schema, query, mutation)
    app = ariadne.asgi.GraphQL(schema, debug=True)

    return app
