import numpy as np
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from persistence.DbConnector import DbConnector
from persistence.entities import DocumentSegmentEntity, VideoSegmentEntity


class TopicModel:
    record_segments = []
    media_records = {}
    model = BERTopic()
    docs = []

    def __init__(self, record_segments: list[DocumentSegmentEntity | VideoSegmentEntity], media_records):

        self.record_segments = record_segments
        self.media_records = media_records

    def createTopicModel(self):

        embeddings = []

        for entity in self.record_segments:
            if isinstance(entity, DocumentSegmentEntity):
                self.docs.append(entity.text)
                embeddings.append(entity.embedding)
            if isinstance(entity, VideoSegmentEntity):
                self.docs.append(entity.transcript)
                embeddings.append(entity.embedding)

        embeddings = np.array(embeddings)

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
        representational_model = MaximalMarginalRelevance(diversity=0.2)

        self.model = BERTopic(
            min_topic_size=7,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representational_model
        )
        print("Running topic model")
        topics = self.model.fit_transform(self.docs, embeddings)
        print("Finished running model")

        model_info = self.model.get_topic_info()
        model_info.to_csv('model_info.csv')
        document_info = self.model.get_document_info(self.docs)
        document_info.to_csv('document_info.csv')

        fig = self.model.visualize_topics()
        fig.write_html("topicmodel.html")

        fig2 = self.model.visualize_barchart(top_n_topics=50, n_words=100)
        fig2.write_html("barchart.html")


    def add_tags_to_mediarecords(self, record_segments, media_records):
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

            elif isinstance(record_segments[i], DocumentSegmentEntity):
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
    database = DbConnector(
        conn_info="user=root password=root host=localhost port=5431 dbname=docprocai_service")
    record_segments = database.get_all_record_segments()
    media_records = database.get_all_media_records()

    topic_model = TopicModel(record_segments, media_records)

    topic_model.createTopicModel()
    mediarecords_with_tags = topic_model.add_tags_to_mediarecords(record_segments, media_records)
    for mrid, tags in mediarecords_with_tags.items():
        database.update_media_record_tags(mrid, list(tags))
