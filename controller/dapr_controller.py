from enum import Enum, auto

import dapr
from dapr.ext.fastapi.app import DaprApp
from fastapi import FastAPI
import uuid

from dto import TaskInformationDto
from controller.events import ContentChangeEvent
from service.DocProcAiService import DocProcAiService


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

            ai_service.enqueue_generate_content_media_record_links(content_id)

        @dapr_app.subscribe(pubsub="meitrex", topic="assessment-content-mutated")
        def assessment_content_mutated_handler(data: dict):
            assessment_id = uuid.UUID(data["data"]["assessmentId"])
            task_information: list[TaskInformationDto] = data["data"]["taskInformationList"]

            ai_service.enqueue_generate_assessment_segments(assessment_id, task_information)

        @dapr_app.subscribe(pubsub="meitrex", topic="content-changed")
        def assessment_content_deleted_handler(data: dict):

            content_change_event = ContentChangeEvent(data["data"]["contentIds"], data["data"]["crudOperation"])

            ai_service.delete_entries_of_assessments(content_change_event)

