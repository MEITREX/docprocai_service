import requests
import json
import os

from events import MediaRecordInfoEvent

PUBSUB_NAME = "meitrex"

class DaprPublisher:
    def __init__(self):
        dapr_port = os.environ.get("DAPR_HTTP_PORT", "3500")
        self.base_url = f"http://localhost:{dapr_port}/v1.0"

    def publish_media_record_info_event(self, event: MediaRecordInfoEvent):
        topic = "media-record-info"
        url = f"{self.base_url}/publish/{PUBSUB_NAME}/{topic}"
        resp = requests.post(url, json=event)
        resp.raise_for_status()
        return resp.json() if resp.content else None