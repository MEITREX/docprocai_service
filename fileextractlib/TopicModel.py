import argparse
import json
import os
from pathlib import Path

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

import PdfProcessor as PdfProcessor

docs = []


def createTopicModel():
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model
    )
    topics = topic_model.fit_transform(docs)
    print(topic_model.get_topic_info())
    print(topic_model.get_topic(0, full=True))

    fig = topic_model.visualize_topics()
    fig.write_html("topicmodel.html")

    fig2 = topic_model.visualize_barchart(top_n_topics=50, n_words=100)
    fig2.write_html("barchart.html")

    fig3 = topic_model.visualize_documents(docs)
    fig3.write_html("docs.html")

    return topics


def addFileToDocs(file, filetype):
    if filetype == "pdf":
        pdf_processor = PdfProcessor.PdfProcessor()
        pages = pdf_processor.process_from_path(file)
        filtered_pages = [x for x in pages if x["text"] is not None and x["text"].strip()]

        docs.append(json.dumps(filtered_pages))

    elif filetype == "json":
        with open(file, encoding="utf8") as f:
            data = json.load(f)
            for x in data:
                docs.append(x.get('transcript'))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('directory', nargs='+')
    # args = parser.parse_args()
    #
    # cwd = Path.cwd()
    #
    # print(cwd)
    # trainingdata_path = args.directory

    # print(trainingdata_path)

    path = "E:/Programmiertes/FoPro/fileextractlib/files/trainingdata"
    files = filter(lambda filepath: filepath.is_file(), Path(path).glob('*'))
    for file in files:
        print(file)
        addFileToDocs(file, "json")

    createTopicModel()
