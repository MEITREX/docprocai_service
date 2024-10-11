from enum import Enum, auto
from uuid import UUID
from torch import Tensor

class MediaRecordEntity:
    def __init__(self, id: UUID, summary: list[str], tags: set):
        self.summary = summary
        self.tags = tags


class DocumentSegmentEntity:
    def __init__(self, id: UUID, media_record_id: UUID, page_index: int, text: str, thumbnail: bytes, title: str,
                 embedding: Tensor):
        self.id = id
        self.media_record_id = media_record_id
        self.page_index = page_index
        self.text = text
        self.thumbnail = thumbnail
        self.title = title
        self.embedding = embedding


class VideoSegmentEntity:
    def __init__(self, id: UUID, media_record_id: UUID, start_time: int, transcript: str,
                 screen_text: str, thumbnail: bytes, title: str, embedding: Tensor):
        self.id = id
        self.media_record_id = media_record_id
        self.start_time = start_time
        self.transcript = transcript
        self.screen_text = screen_text
        self.thumbnail = thumbnail
        self.title = title
        self.embedding = embedding


class AssessmentSegmentEntity:
    def __init__(self, id: UUID, assessment_id: UUID, textual_representation: str, embedding: Tensor):
        self.id = id
        self.assessment_id = assessment_id
        self.textual_representation = textual_representation
        self.embedding = embedding


class MediaRecordSegmentLinkEntity:
    def __init__(self, content_id: UUID, segment1_id: UUID, segment2_id: UUID):
        self.content_id = content_id
        self.segment1_id = segment1_id
        self.segment2_id = segment2_id


class SemanticSearchResultEntity:
    def __init__(self, score: float, media_record_segment_entity: VideoSegmentEntity | DocumentSegmentEntity):
        self.score = score
        self.media_record_segment_entity = media_record_segment_entity


class IngestionStateDbType(Enum):
    ENQUEUED = auto()
    PROCESSING = auto()
    DONE = auto()


class IngestionEntityTypeDbType(Enum):
    MEDIA_RECORD = auto()
    MEDIA_CONTENT = auto()
    ASSESSMENT = auto()

class EntityIngestionInfoEntity:
    def __init__(self, entity_id: UUID, entity_type: IngestionEntityTypeDbType,ingestion_state: IngestionStateDbType):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.ingestion_state = ingestion_state
