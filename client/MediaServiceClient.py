import uuid

import gql
import gql.transport.aiohttp


class MediaServiceClient:
    def __init__(self):
        # TODO: Don't hardcode the URL, instead make it configurable
        transport = gql.transport.aiohttp.AIOHTTPTransport(url="http:///app-media:3001/graphql")
        self._client = gql.Client(transport=transport, fetch_schema_from_transport=True)

    def get_media_record_type_and_download_url(self, document_id: uuid.UUID) -> dict:
        query = gql.gql(
            """
            query GetDocumentDownloadUrl($recordId: UUID!) {
                _internal_noauth_mediaRecordsByIds(ids: [$recordId]) {
                    downloadUrl
                    type
                }
            }
            """
        )
        variables = {"recordId": document_id}
        result = self._client.execute(query, variable_values=variables)
        return result["_internal_noauth_mediaRecordsByIds"][0]
