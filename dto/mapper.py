from functools import singledispatch
import base64
from dto import *

@singledispatch
def entity_to_dto(entity):
    raise NotImplementedError(f"from_entity not implemented for {type(entity)}")

@singledispatch
def media_record_segment_entity_to_dto(entity: MediaRecordSegmentEntity) -> MediaRecordSegmentDto:
    raise NotImplementedError(f"media_record_segment_entity_to_dto not implemented for {type(entity)}")

@entity_to_dto.register(DocumentSegmentEntity)
@media_record_segment_entity_to_dto.register(DocumentSegmentEntity)
def document_segment_entity_to_dto(entity: DocumentSegmentEntity) -> DocumentRecordSegmentDto:
    return {
        "id": entity.id,
        "mediaRecordId": entity.media_record_id,
        "page": entity.page_index,
        "text": entity.text,
        "thumbnail": "data:image/jpeg;base64," + base64.b64encode(entity.thumbnail).decode("utf-8"),
        "title": entity.title
    }

@entity_to_dto.register(VideoSegmentEntity)
@media_record_segment_entity_to_dto.register(VideoSegmentEntity)
def video_segment_entity_to_dto(entity: VideoSegmentEntity) -> VideoRecordSegmentDto:
    return {
        "id": entity.id,
        "mediaRecordId": entity.media_record_id,
        "startTime": entity.start_time,
        "screenText": entity.screen_text,
        "transcript": entity.transcript,
        "thumbnail": "data:image/jpeg;base64," + base64.b64encode(entity.thumbnail).decode("utf-8"),
        "title": entity.title
    }

@singledispatch
def semantic_search_result_entity_to_dto(entity: SemanticSearchResultEntity) -> SemanticSearchResultDto:
    raise NotImplementedError(f"semantic_search_result_entity_to_dto not implemented for {type(entity)}")

@entity_to_dto.register(MediaRecordSegmentSemanticSearchResultEntity)
@semantic_search_result_entity_to_dto.register(MediaRecordSegmentSemanticSearchResultEntity)
def media_record_semantic_search_result_entity_to_dto(entity: MediaRecordSegmentSemanticSearchResultEntity) \
    -> MediaRecordSegmentSemanticSearchResultDto:
    return {
        "score": entity.score,
        "mediaRecordSegment": media_record_segment_entity_to_dto(entity.media_record_segment_entity)
    }

@entity_to_dto.register(AssessmentSemanticSearchResultEntity)
@semantic_search_result_entity_to_dto.register(AssessmentSemanticSearchResultEntity)
def assessment_semantic_search_result_entity_to_dto(entity: AssessmentSemanticSearchResultEntity) \
    -> AssessmentSemanticSearchResultDto:
    return {
        "score": entity.score,
        "assessmentId": entity.assessment_id
    }