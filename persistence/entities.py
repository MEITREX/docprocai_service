from uuid import UUID


class DocumentSegmentEntity:
    def __init__(self, id: UUID, media_record_id: UUID, page_index: int, text: str, thumbnail: bytes, title: str):
        self.id = id
        self.media_record_id = media_record_id
        self.page_index = page_index
        self.text = text
        self.thumbnail = thumbnail
        self.title = title


class VideoSegmentEntity:
    def __init__(self, id: UUID, media_record_id: UUID, start_time: int, transcript: str,
                 screen_text: str, thumbnail: bytes, title: str):
        self.id = id
        self.media_record_id = media_record_id
        self.start_time = start_time
        self.transcript = transcript
        self.screen_text = screen_text
        self.thumbnail = thumbnail
        self.title = title

class MediaRecordSegmentLinkEntity:
    def __init__(self, content_id: UUID, segment1_id: UUID, segment2_id: UUID):
        self.content_id = content_id
        self.segment1_id = segment1_id
        self.segment2_id = segment2_id

class SemanticSearchResultEntity:
    def __init__(self, score: float, media_record_segment_entity: VideoSegmentEntity | DocumentSegmentEntity):
        self.score = score
        self.media_record_segment_entity = media_record_segment_entity