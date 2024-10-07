import uuid
import config
import gql
import gql.transport.aiohttp


class MediaServiceClient:
    def __init__(self):
        transport = gql.transport.aiohttp.AIOHTTPTransport(url=config.current["media_service_url"])
        self.__client = gql.Client(transport=transport, fetch_schema_from_transport=True)

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
        result = await self.__client.execute_async(query, variable_values=variables)
        return result["_internal_noauth_mediaRecordsByIds"][0]

    async def get_media_record_ids_of_content(self, content_id: uuid.UUID) -> list[uuid.UUID]:
        query = gql.gql(
            """
            query GetMediaRecordIdsOfContent($contentId: UUID!) {
                _internal_noauth_mediaRecordsByContentIds(contentIds: [$contentId]) {
                    id
                }
            }
            """)
        variables = {"contentId": str(content_id)}
        result = await self.__client.execute_async(query, variable_values=variables)
        return [x["id"] for x in result["_internal_noauth_mediaRecordsByContentIds"][0]]
