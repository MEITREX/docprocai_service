from webvtt import WebVTT
from fileextractlib.TranscriptGenerator import TranscriptGenerator
import ffmpeg
import pytesseract
import PIL
import io
from torch import Tensor
import Levenshtein


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

    def __init__(self, screen_text_similarity_threshold: float = 0.8):
        self.screen_text_similarity_threshold: float = screen_text_similarity_threshold

    """
    Generates sections of the video with the given file URL. Extracts spoken text and on-screen text of each section.
    """
    def generate_sections(self, file_url: str) -> list[Section]:
        transcript_generator: TranscriptGenerator = TranscriptGenerator()
        vtt: WebVTT = transcript_generator.process_to_vtt(file_url)

        stream = ffmpeg.input(file_url)

        # construct ffmpeg select filter to extract a frame at each transcript caption start time
        select_filters: list[str] = []
        for caption in vtt.captions:
            start_time_seconds: int = caption.start_in_seconds
            select_filters.append(f"eq(t,{start_time_seconds})")

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
        current_section = None
        for bmp_file in bmp_files:
            image = PIL.Image.open(io.BytesIO(bmp_file[0]))
            image_index: int = bmp_file[1]

            screen_text = pytesseract.image_to_string(image)

            if current_section is None:
                # if this is the first image, we need to create a new section
                # captions always have a leading "- ", so we remove it
                current_section = VideoProcessor.Section(
                    start_time=vtt.captions[image_index].start_in_seconds,
                    transcript=vtt.captions[image_index].text[2:],
                    screen_text=screen_text,
                    embedding=None)
            else:
                # otherwise we check if the screen text is similar to the previous screen text
                similarity = Levenshtein.ratio(current_section.screen_text, screen_text)

                if similarity > self.screen_text_similarity_threshold:
                    # if the screen text is similar, we append the current caption to the current section
                    # Captions always have a leading "- ", so we remove it
                    current_section.transcript += " " + vtt.captions[image_index].text[2:]
                else:
                    # if the screen text is not similar, we create a new section
                    # Caption texts always have a leading "- ", so we remove it
                    sections.append(current_section)
                    current_section = VideoProcessor.Section(
                        start_time=vtt.captions[image_index].start_in_seconds,
                        transcript=vtt.captions[image_index].text[2:0],
                        screen_text=screen_text,
                        embedding=None)

        return sections
