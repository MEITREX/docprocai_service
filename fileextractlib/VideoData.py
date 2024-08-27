import PIL.Image
from torch import Tensor
from webvtt import WebVTT


class VideoSectionData:
    """
    Represents a section of a video, containing the start time of the section in seconds, the transcript of the
    section, the screen text of the section, and a text embedding of the section's contents.
    """

    def __init__(self,
                 start_time: int,
                 transcript: str,
                 screen_text: str,
                 thumbnail: PIL.Image.Image,
                 embedding: Tensor):
        self.start_time: int = start_time
        self.transcript: str = transcript
        self.screen_text: str = screen_text
        self.thumbnail: PIL.Image.Image = thumbnail
        self.embedding: Tensor = embedding


class VideoData:
    """
    Represents a video's data, containing the captions and the sections of the video.
    """

    def __init__(self, vtt: WebVTT, sections: list[VideoSectionData]):
        self.vtt: WebVTT = vtt
        self.sections: list[VideoSectionData] = sections
