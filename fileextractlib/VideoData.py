import PIL.Image
from webvtt import WebVTT
from typing import Optional
from numpy.typing import NDArray
import numpy as np

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
                 embedding: Optional[NDArray[np.float64]]):
        self.start_time: int = start_time
        self.transcript: str = transcript
        self.screen_text: str = screen_text
        self.thumbnail: PIL.Image.Image = thumbnail
        self.title = title
        self.embedding: Optional[NDArray[np.float64]] = embedding


class VideoData:
    """
    Represents a video's data, containing the captions and the sections of the video.
    """

    def __init__(self,
                 length_seconds: float,
                 vtt: WebVTT,
                 segments: list[VideoSegmentData],
                 summary: Optional[list[str]] = None):
        if summary is None:
            summary = []
        self.length_seconds: float = length_seconds
        self.vtt: WebVTT = vtt
        self.segments: list[VideoSegmentData] = segments
        self.summary: list[str] = summary
