import sys

import time
from fileextractlib.SentenceEmbeddingRunner import SentenceEmbeddingRunner
from fileextractlib.VideoData import VideoSegmentData

class LectureVideoEmbeddingGenerator:
    def generate_embeddings(self, sections: list[VideoSegmentData]):
        """
        Generates text embeddings for the passed segments.
        """
        sentence_embedding_runner = SentenceEmbeddingRunner()

        for section in sections:
            section.embedding = sentence_embedding_runner.generate_embeddings(
                [section.transcript + "\n\n" + section.screen_text])[0]
