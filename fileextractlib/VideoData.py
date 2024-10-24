import PIL.Image
from torch import Tensor
from webvtt import WebVTT
from typing import Optional

class VideoSegmentData:
    """
    Represents a segment of a video, containing the start time of the section in seconds, the transcript of the
    section, the screen text of the section, and a text embedding of the section's contents.
    """

    def __init__(self,
                 start_time: int,
                 transcript: str,
                 screen_text: str,
                 thumbnail: PIL.Image.Image,
                 title: Optional[str],
                 embedding: Optional[Tensor]):
        self.start_time: int = start_time
        self.transcript: str = transcript
        self.screen_text: str = screen_text
        self.thumbnail: PIL.Image.Image = thumbnail
        self.title = title
        self.embedding: Tensor = embedding


class VideoData:
    """
    Represents a video's data, containing the captions and the sections of the video.
    """

    def __init__(self, vtt: WebVTT, segments: list[VideoSegmentData], summary: list[str] = None):
        if summary is None:
            summary = []
        self.vtt: WebVTT = vtt
        self.segments: list[VideoSegmentData] = segments
        self.summary: list[str] = summary
