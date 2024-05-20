import whisper
import time
from os import path
import argparse
from datetime import timedelta
import numpy as np
import ffmpeg

from webvtt import WebVTT, Caption


class LectureVideoProcessor:
    def __init__(self):
        self.model: whisper.Whisper = whisper.load_model(name="base")

    def process(self, file_name: str) -> str:
        # load audio data from file
        try:
            sample_rate = 16000
            y, _ = (
                ffmpeg.input(file_name, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        # audio data load into numpy array
        audio_data = np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0

        start_time: float = time.time()
        result = self.model.transcribe(audio=audio_data, word_timestamps=True)
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--file")
    args = parser.parse_args()
    processor = LectureVideoProcessor()
    processor.process(args.file)
