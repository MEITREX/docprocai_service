from typing import Optional, TypedDict

class MediaRecordInfoEvent(TypedDict):
    mediaRecordId: str
    durationSeconds: Optional[float]
    pageCount: Optional[int]
