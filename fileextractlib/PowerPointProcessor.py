from fileextractlib.DocumentData import DocumentData
import os
import typing
import tempfile
import subprocess
from fileextractlib.PdfProcessor import PdfProcessor


class PowerPointProcessor:
    def __init__(self):
        self.pdf_processor = PdfProcessor()

    def process_from_io(self, file: typing.BinaryIO) -> DocumentData:
        # convert the pptx to a pdf using libreoffice. We need to do it in this roundabout way because there is
        # no library to render pptx to images, and we need images for the thumbnails

        # firstly, we need to create a temp directory to save the file to. This is necessary because libreoffice
        # cannot output a converted file to stdout AND it also cannot output a file to a specific file name, so we
        # need to create a whole directory to make sure no naming conflicts arise
        with tempfile.TemporaryDirectory() as temp_dir:
            # save the bytes of the pptx to the temp dir
            in_file_path = os.path.join(temp_dir, "file.pptx")
            with open(in_file_path, "wb") as f:
                f.write(file.read())

            # convert the pptx to pdf
            subprocess.run([
                "soffice",
                "--headless",
                "--convert-to", "pdf:impress_pdf_Export",
                "--outdir", str(os.path.abspath(temp_dir)),
                str(os.path.abspath(in_file_path))
            ])

            # use pdf processor to process the pptx in the same way we'd process a pdf
            with open(os.path.join(temp_dir, "file.pdf"), "rb") as pdf_file:
                return self.pdf_processor.process_from_io(pdf_file)
