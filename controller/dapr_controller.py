import dapr
from dapr.ext.fastapi.app import DaprApp
from fastapi import FastAPI
import uuid
import service.DocProcAiService as DocProcAiService


class DaprController:
    def __init__(self, app: FastAPI, ai_service: DocProcAiService):
        dapr_app = DaprApp(app)

        @dapr_app.subscribe(pubsub="meitrex", topic="media-record-file-created")
        def media_record_file_created_handler(data: dict):
            media_record_id = uuid.UUID(data["data"]["mediaRecordId"])

            ai_service.enqueue_ingest_media_record_task(media_record_id)

        @dapr_app.subscribe(pubsub="meitrex", topic="media-record-deleted")
        def media_record_deleted_handler(data: dict):
            media_record_id = uuid.UUID(data["data"]["mediaRecordId"])

            ai_service.delete_entries_of_media_record(media_record_id)

        @dapr_app.subscribe(pubsub="meitrex", topic="content-media-record-links-set")
        def content_media_record_links_set_handler(data: dict):
            content_id = uuid.UUID(data["data"]["contentId"])
            media_record_ids: list[uuid.UUID] = [uuid.UUID(id) for id in data["data"]["mediaRecordIds"]]

            ai_service.enqueue_generate_content_media_record_links(content_id, media_record_ids)
