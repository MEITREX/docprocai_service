import whisper
import time
from os import path
from datetime import timedelta

class LectureVideoProcessor:
    def __init__(self):
        self.model: whisper.Whisper = whisper.load_model(name="base")

    def process(self, audio_file: str) -> str:
        start_time: float = time.time()
        result  = model.transcribe(audio=audio_file, word_timestamps=True)
        end_time: float = time.time()

        result_text: str = ""

        for segment in result["segments"]:
            segment_text: str = ""
            for word in segment["words"]:
                segment_text += word["word"] + " "
    
            result_text += str(timedelta(seconds=segment["words"][0]["start"])) + " -- " + segment_text + "\n"

        print("Processed text in " + str(end_time - start_time) +  " seconds.")
        return result_text