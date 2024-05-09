import whisper
import time
from os import path
from datetime import timedelta

from webvtt import WebVTT, Caption


class LectureVideoProcessor:
    def __init__(self):
        self.model: whisper.Whisper = whisper.load_model(name="base")

    def process(self, audio_file: str) -> str:
        start_time: float = time.time()
        result = self.model.transcribe(audio=audio_file, word_timestamps=True)
        end_time: float = time.time()

        result_text: str = ""

        vtt = WebVTT()

        for segment in result["segments"]:
            segment_text: str = ""
            for word in segment["words"]:
                segment_text += word["word"] + " "

            segment_start = timedelta(seconds=segment["words"][0]["start"])
            segment_end = timedelta(seconds=segment["words"][-1]["end"])

            if segment_start.microseconds == 0:
                segment_start = segment_start + timedelta(microseconds=1)

            if segment_end.microseconds == 0:
                segment_end = segment_end + timedelta(microseconds=1)

            caption = Caption(
                str(segment_start),
                str(segment_end),
                "-" + segment_text
            )
            vtt.captions.append(caption)
            vtt.save("test.vtt")
            result_text += str(timedelta(seconds=segment["words"][0]["start"])) + " -- " + segment_text + "\n"

        print("Processed text in " + str(end_time - start_time) + " seconds.")
        return result_text


if __name__ == "__main__":
    processor = LectureVideoProcessor()
    processor.process("E:\Programmiertes\FoPro\Test.wav")
