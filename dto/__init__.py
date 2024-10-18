"""
Contains the DTOs which are the in-code representation of the types defined in the service's GraphQL schema.

Implementer's Notes: As the GraphQL library we use (Ariadne GraphQL) expects the resolver functions to return
dictionaries, we use let our DTOs inherit from Python's TypedDict type. This means that they aren't actual custom
objects at runtime but are just regular dicts, however we still get the full type-checking capabilities at edit-time
in our IDE.
Basically these types are only used to annotate variables in the code with type hints, the actual "instantiation" of
these objects is with the regular python dict syntax: { "id": "123" }
"""

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

type MediaRecordSegmentDto = DocumentRecordSegmentDto | VideoRecordSegmentDto

class MediaRecordSegmentSemanticSearchResultDto(TypedDict):
    score: float
    mediaRecordSegment: VideoRecordSegmentDto | DocumentRecordSegmentDto

class AssessmentSemanticSearchResultDto(TypedDict):
    score: float
    assessmentId: UUID

type SemanticSearchResultDto = MediaRecordSegmentSemanticSearchResultDto |AssessmentSemanticSearchResultDto

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

class TaskInformationDto(TypedDict):
    taskId: UUID
    textualRepresentation: str