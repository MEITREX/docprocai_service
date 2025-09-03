from typing import Optional, TypedDict

class MediaRecordInfoEvent(TypedDict):
    mediaRecordId: str
    mediaType: str
    durationSeconds: Optional[float]
    pageCount: Optional[int]
