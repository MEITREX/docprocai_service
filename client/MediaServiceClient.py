import uuid
import config
import gql
import gql.transport.aiohttp


class MediaServiceClient:
    def __init__(self):
        transport = gql.transport.aiohttp.AIOHTTPTransport(url=config.current["media_service_url"])
        self._client = gql.Client(transport=transport, fetch_schema_from_transport=True)

    async def get_media_record_type_and_download_url(self, document_id: uuid.UUID) -> dict:
        query = gql.gql(
            """
            query GetDocumentDownloadUrl($recordId: UUID!) {
                _internal_noauth_mediaRecordsByIds(ids: [$recordId]) {
                    internalDownloadUrl
                    type
                }
            }
            """
        )
        variables = {"recordId": str(document_id)}
        result = await self._client.execute_async(query, variable_values=variables)
        return result["_internal_noauth_mediaRecordsByIds"][0]
