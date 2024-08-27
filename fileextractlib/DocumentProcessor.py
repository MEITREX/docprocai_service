import io

import requests
from fileextractlib.DocumentData import DocumentData
from fileextractlib.PdfProcessor import PdfProcessor
from fileextractlib.PowerPointProcessor import PowerPointProcessor


class DocumentProcessor:
    def __init__(self):
        self.pdf_processor = PdfProcessor()
        self.powerpoint_processor = PowerPointProcessor()

    def process(self, file_url: str) -> DocumentData:
        # get the binary data from the file's url
        res = requests.get(file_url)
        file_bytes = io.BytesIO(res.content)

        content_type_header = res.headers.get('content-type')
        if content_type_header is None:
            raise ValueError("Content type header not found")

        if content_type_header == "application/pdf":
            return self.pdf_processor.process_from_io(file_bytes)
        elif content_type_header == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            return self.powerpoint_processor.process_from_io(file_bytes)
