from typing import NotRequired, TypedDict, Optional
from uuid import UUID

from persistence.entities import *


class DocumentRecordSegmentDto(TypedDict):
    id: UUID
    mediaRecordId: UUID
    page: int
    text: str
    thumbnail: str
    title: NotRequired[str | None]


class VideoRecordSegmentDto(TypedDict):
    id: UUID
    mediaRecordId: UUID
    startTime: int
    screenText: str
    transcript: str
    thumbnail: str
    title: NotRequired[str | None]


class SemanticSearchResultDto(TypedDict):
    score: float
    mediaRecordSegment: VideoRecordSegmentDto | DocumentRecordSegmentDto


class MediaRecordSegmentLinkDto(TypedDict):
    segment1: VideoRecordSegmentDto | DocumentRecordSegmentDto
    segment2: VideoRecordSegmentDto | DocumentRecordSegmentDto

class AiEntityProcessingStateDto(Enum):
    UNKNOWN = auto()
    ENQUEUED = auto()
    PROCESSING = auto()
    DONE = auto()

class AiEntityProcessingProgressDto(TypedDict):
    entityId: UUID
    state: AiEntityProcessingStateDto
    queuePosition: Optional[int]