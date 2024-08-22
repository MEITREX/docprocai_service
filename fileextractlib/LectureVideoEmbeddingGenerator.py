import sys

import time
from fileextractlib.SentenceEmbeddingRunner import SentenceEmbeddingRunner
from fileextractlib.VideoProcessor import VideoProcessor
from torch import Tensor


class LectureVideoEmbeddingGenerator:
    class Section:
        def __init__(self, start_time: int, transcript: str, screen_text: str, embedding: Tensor):
            self.start_time: int = start_time
            self.transcript: str = transcript
            self.screen_text: str = screen_text
            self.embedding: Tensor = embedding

    def generate_embeddings(self, sections: list[Section]):
        sentence_embedding_runner = SentenceEmbeddingRunner()

        for section in sections:
            section.embedding = sentence_embedding_runner.generate_embeddings(
                [section.transcript + "\n\n" + section.screen_text])[0]

        
if __name__ == "__main__":
    start_time = time.time()

    video_processor = VideoProcessor()
    video_data = video_processor.process(sys.argv[1])
    del video_processor

    generator = LectureVideoEmbeddingGenerator()
    generator.generate_embeddings(video_data.sections)
    end_time = time.time()
    print("Embedding generated successfully in " + str(end_time - start_time) + " seconds.")
