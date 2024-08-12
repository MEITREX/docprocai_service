import sys

from webvtt import WebVTT
from fileextractlib.TranscriptGenerator import TranscriptGenerator
import ffmpeg
import pytesseract
import PIL
import io
import time
import Levenshtein
from fileextractlib.SentenceEmbeddingRunner import SentenceEmbeddingRunner
from fileextractlib.VideoProcessor import VideoProcessor
from torch import Tensor


class LectureVideoEmbeddingGenerator:
    screen_text_similarity_threshold: float = 0.8

    class Section:
        def __init__(self, start_time: int, transcript: str, screen_text: str, embedding: Tensor):
            self.start_time: int = start_time
            self.transcript: str = transcript
            self.screen_text: str = screen_text
            self.embedding: Tensor = embedding

    def generate_embeddings(self, file_url: str) -> list[Section]:
        video_processor = VideoProcessor(self.screen_text_similarity_threshold)
        sections = video_processor.generate_sections(file_url)
        del video_processor

        sentence_embedding_runner = SentenceEmbeddingRunner()

        for section in sections:
            section.embedding = sentence_embedding_runner.generate_embeddings(
                [section.transcript + "\n\n" + section.screen_text])[0]

        return sections

        
if __name__ == "__main__":
    start_time = time.time()
    generator = LectureVideoEmbeddingGenerator()
    generator.generate_embeddings(sys.argv[1])
    end_time = time.time()
    print("Embedding generated successfully in " + str(end_time - start_time) + " seconds.")
