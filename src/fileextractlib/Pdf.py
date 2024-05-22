import os
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

import whisper
import time
from os import path
import argparse
from datetime import timedelta
import numpy as np
import ffmpeg
import io

from webvtt import WebVTT, Caption


class PdfProcessor:
    """
     Can be used to convert documents in pdf format into raw text
    """

    def process(self, file_name: str) -> str:

            
        doc = convert_from_path(file_name)
        
        path, fileName = os.path.split(file_name)
        fileBaseName, fileExtension = os.path.splitext(fileName)
        os.environ['OMP_THREAD_LIMIT'] = '20'

        for page_number, page_data in enumerate(doc):
            txt = pytesseract.image_to_string(page_data).encode("utf-8")
            print("Page # {} - {}".format(str(page_number),txt))

        for segment in result["segments"]:
            segment_text: str = ""
            for word in segment["words"]:
                segment_text += word["word"]

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

        print("Processed text in " + str(end_time - start_time) + " seconds.")
        
        with io.StringIO() as f:
            vtt.write(f)
            f.seek(0)
            return f.read() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file")
    args = parser.parse_args()
    processor = PdfProcessor()
    result = processor.process(args.file)
    print(result)
