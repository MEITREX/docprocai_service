import io
import pdf2image
import argparse
import typing
import tika
import tika.parser
from pypdf import PdfWriter, PdfReader
from fileextractlib.DocumentData import DocumentData, PageData


class PdfProcessor:
    """
     Can be used to convert documents in pdf format into raw text
    """

    def __init__(self):
        tika.initVM()

    def process_from_io(self, file: typing.BinaryIO) -> DocumentData:
        # create thumbnail images for each page
        page_images = pdf2image.convert_from_bytes(file.read())

        # split the pdf into pages, so we can extract text for each page separately
        file.seek(0)
        pdf_reader = PdfReader(file)

        # convert each page to text using tika
        pages: list[PageData] = []
        for page_index in range(len(pdf_reader.pages)):
            page_pdf_writer = PdfWriter()
            page_pdf_writer.add_page(pdf_reader.pages[page_index])

            with io.BytesIO() as page_pdf_bytes:
                page_pdf_writer.write(page_pdf_bytes)
                page_pdf_bytes.seek(0)
                page_text = tika.parser.from_buffer(page_pdf_bytes)["content"].strip()

                if page_text == "":
                    continue

                pages.append(PageData(page_index, page_text, page_images[page_index], None))

        return DocumentData(pages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file")
    args = parser.parse_args()
    processor = PdfProcessor()
    with open(args.file, "rb") as f:
        result = processor.process_from_io(args.file)
        print(result)
