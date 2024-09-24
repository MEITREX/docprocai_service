# DocProcAI-Service

This service is designed to process and manage uploaded lecture material (video recordings, documents, slides) to facilitate some advanced features in the MEITREX platform.

## Features
* Splitting of lecture videos into sections based on detected slide changes via computer vision
* OCR of lecture video on screen text
* Transcript & Closed Captions generation for lecture videos
* Generating of text embeddings on a per-section-basis for videos and per-page-basis for documents
* Semantic search/fetching of semantically similar sections of lecture material
* Automatic generation of section titles for the video sections generated

## Configuration
The service uses the `config.yaml` file located in the root directory for configuration.
For further information about configuration check out this file, all configuration properties are explained using in-file comments.
