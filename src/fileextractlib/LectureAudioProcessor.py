import time
from whispercpp import Whisper
from os import path
import argparse
from datetime import timedelta
import ffmpeg
import numpy as np
import typing

from webvtt import WebVTT, Caption

class LectureAudioProcessor:
    """
    Extracts transcripts from a lecture video or audio file.
    """

    def __init__(self):
        self.whisper = Whisper.from_pretrained("base")
        self.whisper.params.with_print_timestamps(True).build()

    def process(self, file_name: str) -> str:
        start_time: float = time.time()

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
        arr = np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0

        # transcribe audio
        result = self.whisper.transcribe(arr)
        end_time: float = time.time()

        result_text: str = ""

        print(result)
        return
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
    processor = LectureAudioProcessor()
    processor.process(args.file)
