import uuid
import config
import gql
import gql.transport.aiohttp
import os


class MediaServiceClient:

    async def get_media_record_type_and_download_url(self, document_id: uuid.UUID) -> dict:
        self.__init_client_if_not_already()

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

    async def get_media_record_ids_of_contents(self, content_ids: list[uuid.UUID]) -> list[uuid.UUID]:
        self.__init_client_if_not_already()

        query = gql.gql(
            """
            query GetMediaRecordIdsOfContent($contentIds: [UUID!]!) {
                _internal_noauth_mediaRecordsByContentIds(contentIds: $contentIds) {
                    id
                }
            }
            """)
        # we need to convert the UUIDs to strings first, otherwise the graphql client we use struggles with them
        variables = {"contentIds": [str(x) for x in content_ids]}
        query_result = await self.__client.execute_async(query, variable_values=variables)

        # query result is a list of lists, flat-map it
        media_records: list[uuid.UUID] = []
        for media_record_list in query_result["_internal_noauth_mediaRecordsByContentIds"]:
            for media_record in media_record_list:
                media_records.append(media_record["id"])

        return media_records

    def __init_client_if_not_already(self):
        transport = gql.transport.aiohttp.AIOHTTPTransport(url=os.environ.get("media_service_url"))
        self.__client = gql.Client(transport=transport, fetch_schema_from_transport=True)