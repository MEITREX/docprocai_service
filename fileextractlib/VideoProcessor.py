from typing import Optional

from webvtt import WebVTT

from fileextractlib.ImageTemplateMatcher import ImageTemplateMatcher
from fileextractlib.TranscriptGenerator import TranscriptGenerator
import ffmpeg
import pytesseract
import PIL.Image
import io
from torch import Tensor
import Levenshtein
import sys


class VideoProcessor:
    """
    Processes lecture videos by firstly using speech-to-text to generate a transcript, and then using OCR to extract
    screen text. The screen text is then compared to the transcript to create sections of the video. Each section
    contains the transcript, screen text, and a text embedding of its contents.
    """

    class Section:
        """
        Represents a section of a video, containing the start time of the section in seconds, the transcript of the
        section, the screen text of the section, and a text embedding of the section's contents.
        """
        def __init__(self, start_time: int, transcript: str, screen_text: str, embedding: Tensor):
            self.start_time: int = start_time
            self.transcript: str = transcript
            self.screen_text: str = screen_text
            self.embedding: Tensor = embedding

    class VideoData:
        """
        Represents a video's data, containing the captions and the sections of the video.
        """
        def __init__(self, vtt: WebVTT, sections: list["VideoProcessor.Section"]):
            self.vtt: WebVTT = vtt
            self.sections: list[VideoProcessor.Section] = sections

    """
    Initializes a new VideoProcessor with the given screen text similarity threshold (range 0.0 to 1.0) and 
    minimum section length in seconds.
    """
    def __init__(self, section_image_similarity_threshold: float = 0.9, minimum_section_length: int = 15):
        self.section_image_similarity_threshold: float = section_image_similarity_threshold
        self.minimum_section_length: int = minimum_section_length

    """
    Generates sections of the video with the given file URL. Extracts spoken text and on-screen text of each section.
    """
    def process(self, file_url: str) -> VideoData:
        transcript_generator: TranscriptGenerator = TranscriptGenerator()
        vtt: WebVTT = transcript_generator.process_to_vtt(file_url)

        stream = ffmpeg.input(file_url)

        # construct ffmpeg select filter to extract a frame at each transcript caption start time
        select_filters: list[str] = []
        for caption in vtt.captions:
            start_time_seconds: int = caption.start_in_seconds

            # ffmpeg's select filter requires a start time greater than 0
            if start_time_seconds == 0:
                start_time_seconds = 1

            select_filters.append(f"lt(prev_pts*TB,{start_time_seconds})*gte(pts*TB,{start_time_seconds})")

        out, err = (stream
                    .filter_("select", "+".join(select_filters))
                    .output("-", vsync=0, format="image2pipe", vcodec="bmp")
                    .run(capture_stdout=True)
                    )

        # list of tuples, where the first element in the tuple is the BMP file's raw bytes, and the second is the
        # index of the image in relation to the captions
        bmp_files: list[tuple[bytes, int]] = []

        image_index = 0
        byte_offset = 0
        while byte_offset < len(out):
            # ensure BMP magic number is present
            if out[byte_offset:byte_offset + 2] != b'BM':
                raise ValueError("Invalid BMP file")

            # get size of bmp file in bytes
            size_in_bytes: int = int.from_bytes(out[byte_offset + 2:byte_offset + 6], byteorder='little')
            bmp_files.append((out[byte_offset:byte_offset + size_in_bytes], image_index))

            byte_offset += size_in_bytes
            image_index += 1

        # delete ffmpeg output, we don't need it anymore
        del out

        # we will now create longer sections from our captions. Captions usually have a length of a sentence or a part
        # of a sentence.
        # We extracted images at the start of each caption, now we will check when the video changes significantly and
        # create a new section, merging the captions within the timespan of that section
        sections: list[VideoProcessor.Section] = []
        current_section: Optional[VideoProcessor.Section] = None
        current_section_cropped_image: Optional[PIL.Image.Image] = None
        current_section_image: Optional[PIL.Image.Image] = None
        for bmp_file in bmp_files:
            image = PIL.Image.open(io.BytesIO(bmp_file[0]))
            image_index: int = bmp_file[1]

            # we crop the image to its center portion because in some videos the
            # lecturer might put other things, e.g. a webcam feed, in the corners
            cropped_image = image.crop((image.width * 1/6, image.height * 1/10, 
                                        image.width * 5/6, image.height * 9/10))

            if current_section is None:
                # if this is the first image, we need to create a new section
                # captions always have a leading "- ", so we remove it
                current_section = VideoProcessor.Section(
                    start_time=vtt.captions[image_index].start_in_seconds,
                    transcript=vtt.captions[image_index].text[2:],
                    screen_text=None,
                    embedding=None)
                current_section_image = image
                current_section_cropped_image = cropped_image
            else:
                # otherwise we check if the screen is similar to the current sections screen 
                # image using template matching
                matcher = ImageTemplateMatcher(template=current_section_cropped_image)

                similarity = matcher.match(cropped_image)

                if (similarity >= self.section_image_similarity_threshold
                        or current_section.start_time + self.minimum_section_length
                        > vtt.captions[image_index].start_in_seconds):
                    # if the screen is more similar than the threshold, or minimum section length 
                    # hasn't been reached yet, we append the current caption to the current section. 
                    # Captions always have a leading "- ", so we remove it
                    current_section.transcript += " " + vtt.captions[image_index].text[2:]
                else:
                    # if the screen is not similar, we create a new section
                    # Caption texts always have a leading "- ", so we remove it
                    current_section.screen_text = pytesseract.image_to_string(current_section_image)
                    sections.append(current_section)

                    current_section = VideoProcessor.Section(
                        start_time=vtt.captions[image_index].start_in_seconds,
                        transcript=vtt.captions[image_index].text[2:],
                        screen_text=None,
                        embedding=None)
                    current_section_image = image
                    current_section_cropped_image = cropped_image

        video_data = VideoProcessor.VideoData(vtt, sections)

        return video_data
