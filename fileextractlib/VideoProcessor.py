from typing import Optional

from webvtt import WebVTT

from fileextractlib.ImageTemplateMatcher import ImageTemplateMatcher
from fileextractlib.TranscriptGenerator import TranscriptGenerator
import ffmpeg
import tika
import tika.parser
import PIL.Image
import PIL.ImageEnhance
import io
from fileextractlib.VideoData import VideoData, VideoSegmentData


class VideoProcessor:
    """
    Processes lecture videos by firstly using speech-to-text to generate a transcript, and then using OCR to extract
    screen text. The video stream is then analyzed using computer vision to split the video into segments. Each segment
    contains the transcript, screen text, and a text embedding of its contents.
    """

    """
    Initializes a new VideoProcessor with the given screen text similarity threshold (range 0.0 to 1.0) and 
    minimum segment length in seconds.
    """
    def __init__(self, segment_image_similarity_threshold: float = 0.9, minimum_segment_length: int = 15):
        tika.initVM()

        self.segment_image_similarity_threshold: float = segment_image_similarity_threshold
        self.minimum_segment_length: int = minimum_segment_length

    """
    Generates segments of the video with the given file URL. Extracts spoken text and on-screen text of each segment.
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

        # we will now create longer segments from our captions. Captions usually have a length of a sentence or a part
        # of a sentence.
        # We extracted images at the start of each caption, now we will check when the video changes significantly and
        # create a new segment, merging the captions within the timespan of that segment
        segments: list[VideoSegmentData] = []
        current_segment: Optional[VideoSegmentData] = None
        current_segment_cropped_image: Optional[PIL.Image.Image] = None
        current_segment_image: Optional[PIL.Image.Image] = None
        for bmp_file in bmp_files:
            image = PIL.Image.open(io.BytesIO(bmp_file[0]))
            image_index: int = bmp_file[1]

            # we crop the image to its center portion because in some videos the
            # lecturer might put other things, e.g. a webcam feed, in the corners
            cropped_image = image.crop((image.width * 1/6, image.height * 1/10, 
                                        image.width * 5/6, image.height * 9/10))

            if current_segment is None:
                # if this is the first image, we need to create a new segment
                # captions always have a leading "- ", so we remove it
                # Set the screen text, thumbnail, and embedding later when we have found the whole segment
                current_segment = VideoSegmentData(
                    start_time=vtt.captions[image_index].start_in_seconds,
                    transcript=vtt.captions[image_index].text[2:],
                    screen_text=None,
                    thumbnail=None,
                    title=None,
                    embedding=None)
                current_segment_image = image
                current_segment_cropped_image = cropped_image
            else:
                # otherwise we check if the screen is similar to the current segment's screen
                # image using template matching
                matcher = ImageTemplateMatcher(template=current_segment_cropped_image, scaling_factor=0.4)

                similarity = matcher.match(cropped_image)

                if (similarity >= self.segment_image_similarity_threshold
                        or current_segment.start_time + self.minimum_segment_length
                        > vtt.captions[image_index].start_in_seconds):
                    # if the screen is more similar than the threshold, or minimum segment length
                    # hasn't been reached yet, we append the current caption to the current segment.
                    # Captions always have a leading "- ", so we remove it
                    current_segment.transcript += " " + vtt.captions[image_index].text[2:]
                else:
                    with io.BytesIO() as enhanced_image_bytes:
                        # Increase contrast on the screenshot image, this improves OCR performance for colored text
                        enhanced_image = PIL.ImageEnhance.Contrast(current_segment_image).enhance(3.0)
                        # Save image bytes
                        enhanced_image.save(enhanced_image_bytes, format="PNG")
                        enhanced_image_bytes.seek(0)
                        # Perform OCR using tika
                        tika_res = tika.parser.from_buffer(enhanced_image_bytes)["content"].strip()
                        if tika_res is not None:
                            current_segment.screen_text = tika_res.strip()
                        else:
                            current_segment.screen_text = ""

                    current_segment.thumbnail = current_segment_image
                    segments.append(current_segment)
                    # if the screen is not similar, we create a new segment
                    # Caption texts always have a leading "- ", so we remove it
                    current_segment = VideoSegmentData(
                        start_time=vtt.captions[image_index].start_in_seconds,
                        transcript=vtt.captions[image_index].text[2:],
                        screen_text=None,
                        thumbnail=None,
                        title=None,
                        embedding=None)
                    current_segment_image = image
                    current_segment_cropped_image = cropped_image

        video_data = VideoData(vtt, segments)

        return video_data
