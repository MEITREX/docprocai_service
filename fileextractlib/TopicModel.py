import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from persistence.DbConnector import DbConnector
import dto.mapper as mapper
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

        print(model.get_topic_info())
        print(model.get_topic(0, full=True))

        fig = model.visualize_topics()
        fig.write_html("topicmodel.html")

        fig2 = model.visualize_barchart(top_n_topics=50, n_words=100)
        fig2.write_html("barchart.html")

        return topics


if __name__ == "__main__":
    topic_model = TopicModel()

    topic_model.createTopicModel()
