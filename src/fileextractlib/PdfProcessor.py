import os
from pdf2image import convert_from_path
import pytesseract
from os import path
import argparse

class PdfProcessor:
    """
     Can be used to convert documents in pdf format into raw text
    """

    def process(self, file_name: str) -> list:

            
        doc = convert_from_path(file_name)
        os.environ['OMP_THREAD_LIMIT'] = '20'

        pages = []

        for page_number, page_data in enumerate(doc):
            text = pytesseract.image_to_string(page_data)
            pages.append({
                "page_number": page_number,
                "text": text}
            )
        
        return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file")
    args = parser.parse_args()
    processor = PdfProcessor()
    result = processor.process(args.file)
    print(result)
