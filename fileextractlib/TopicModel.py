import numpy as np
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired

from persistence.MediaRecordInfoDbConnector import MediaRecordInfoDbConnector
from persistence.SegmentDbConnector import SegmentDbConnector
from persistence.entities import DocumentSegmentEntity, VideoSegmentEntity
import logging
import psycopg

_logger = logging.getLogger(__name__)


class TopicModel:
    model = BERTopic()

    def __init__(self, record_segments: list[DocumentSegmentEntity | VideoSegmentEntity], media_records):
        self.record_segments = []
        self.media_records = {}
        self.docs = []
        self.record_segments = record_segments
        self.media_records = media_records
        self.docs = []

    def create_topic_model(self):
        embeddings = []

        _logger.info("Adding segments")

        for entity in self.record_segments:
            if isinstance(entity, DocumentSegmentEntity):
                self.docs.append(entity.text)
                embeddings.append(entity.embedding)
            if isinstance(entity, VideoSegmentEntity):
                self.docs.append(entity.transcript)
                embeddings.append(entity.embedding)



        if len(self.docs) < 11:
            _logger.info("More documents needed to create topic model.")
            return

        embeddings = np.array(embeddings)

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
        mmr = MaximalMarginalRelevance(diversity=0.2)
        kbi = KeyBERTInspired()

        representation_models = [mmr, kbi]

        self.model = BERTopic(
            min_topic_size=7,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_models
        )

        self.model.fit_transform(self.docs, embeddings)

        print("Model has been fit")

    def add_tags_to_media_records(self, record_segments, media_records):
        if len(self.docs) < 11:
            _logger.info("Topic model wasn't created. More documents needed.")
            return
        document_info = self.model.get_document_info(self.docs)
        mediarecords_with_tags = {}

        i = 0
        for record in media_records:
            mediarecords_with_tags.update({record.get(id): set()})

        while i < len(record_segments):
            mediarecord_id = record_segments[i].media_record_id

            if isinstance(record_segments[i], DocumentSegmentEntity):
                if record_segments[i].text != document_info['Document'].iat[i]:
                    continue

            elif isinstance(record_segments[i], VideoSegmentEntity):
                if record_segments[i].transcript != document_info['Document'].iat[i]:
                    continue

            tags = set()
            if mediarecords_with_tags.get(mediarecord_id) is not None:
                tags = mediarecords_with_tags.get(mediarecord_id)
            tags.update(set(document_info['Representation'].iat[i]))

            mediarecords_with_tags.update({mediarecord_id: tags})
            i += 1

        return mediarecords_with_tags

if __name__ == "__main__":

    print("Connecting to DB")
    database_connection = psycopg.connect(
        "user=root password=root host=localhost port=5432 dbname=docprocai_service",
        autocommit=True,
        row_factory=psycopg.rows.dict_row
    )

    segment_database = SegmentDbConnector(database_connection)
    media_record_info_database = MediaRecordInfoDbConnector(database_connection)

    print("Loading segments and media records")

    record_segments = segment_database.get_all_media_record_segments()
    media_records = media_record_info_database.get_all_media_records()

    topic_model = TopicModel(record_segments, media_records)

    print("Running Topic model")
    topic_model.create_topic_model()

    media_records_with_tags = topic_model.add_tags_to_media_records(record_segments, media_records)
    if media_records_with_tags is not None:
        for mrid, tags in media_records_with_tags.items():
            media_record_info_database.update_media_record_tags(mrid, list(tags))

    print("Done")

