import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from persistence.DbConnector import DbConnector
from persistence.entities import DocumentSegmentEntity, VideoSegmentEntity


class TopicModel:
    def __init__(self):
        self.database = DbConnector(
            conn_info="user=root password=root host=localhost port=5431 dbname=docprocai_service")

    def createTopicModel(self):
        query_results = self.database.get_all_record_segments()

        docs = []
        embeddings = []

        for entity in query_results:
            if isinstance(entity, DocumentSegmentEntity):
                docs.append(entity.text)
                embeddings.append(entity.embedding)
            if isinstance(entity, VideoSegmentEntity):
                docs.append(entity.transcript)
                embeddings.append(entity.embedding)

        embeddings = np.array(embeddings)

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

        model = BERTopic(
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model
        )
        print("Running topic model")
        topics = model.fit_transform(docs, embeddings)
        print("Finished running model")

        model_info = model.get_topic_info()
        model_info.to_csv('model_info.csv')
        document_info = model.get_document_info(docs)
        document_info.to_csv('document_info.csv')

        fig = model.visualize_topics()
        fig.write_html("topicmodel.html")

        fig2 = model.visualize_barchart(top_n_topics=50, n_words=100)
        fig2.write_html("barchart.html")

        self.add_tags_to_mediarecords(query_results, model, docs)

    def add_tags_to_mediarecords(self, query_results, model, docs):
        document_info = model.get_document_info(docs)

        i = 0
        mediarecords_with_tags = {}
        while i < len(query_results):
            mediarecord_id = query_results[i].media_record_id

            if isinstance(query_results[i], DocumentSegmentEntity):
                if query_results[i].text != document_info['Document'].iat[i]:
                    continue

            elif isinstance(query_results[i], DocumentSegmentEntity):
                if query_results[i].transcript != document_info['Document'].iat[i]:
                    continue

            tags = set()
            if mediarecords_with_tags.get(mediarecord_id) is not None:
                tags = mediarecords_with_tags.get(mediarecord_id)
            tags.update(set(document_info['Representation'].iat[i]))

            mediarecords_with_tags.update({mediarecord_id: tags})
            i += 1


if __name__ == "__main__":
    topic_model = TopicModel()

    topic_model.createTopicModel()
