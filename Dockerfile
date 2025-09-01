FROM python:3.12

# store application data in /app in the container volume
WORKDIR /app

RUN apt update -y
# poppler for pdf2image, tesseract for OCR, ffmpeg for video processing, libreoffice impress for pptx to pdf conversion
# the jre for tika
RUN apt install -y poppler-utils tesseract-ocr ffmpeg libreoffice-impress default-jre

# only copy the requirements.txt file to the container for now, so that the dependencies can be installed and this
# step can be cached by docker so dependencies don't need to be reinstalled every time the code changes
COPY requirements.txt .

# install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# copy the current directory contents into the container at /app
COPY . .

# This environment varialble must be set to connect this service with the media service. 
# ENV media_service_url="http://app-media:3001/graphql"
# ENV connection_string="user=root password=root host=database port=5432 dbname=docprocai_service"

CMD ["python", "./app.py"]
