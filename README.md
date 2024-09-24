# DocProcAI-Service

This service is designed to process and manage uploaded lecture material (video recordings, documents, slides) to facilitate some advanced features in the MEITREX platform.

## Features
* Splitting of lecture videos into sections based on detected slide changes via computer vision
* OCR of lecture video on screen text
* Transcript & Closed Captions generation for lecture videos
* Generating of text embeddings on a per-section-basis for videos and per-page-basis for documents
* Semantic search/fetching of semantically similar sections of lecture material
* Automatic generation of section titles for the video sections generated

## Installation
This service requires pytorch to function. As pytorch GPU-support is required for some features of this service, the pip-distributed version of pytorch cannot be used and instead a
platform-specific version has to be used.
By default, pytorch for NVIDIA CUDA 11.8 is used, as this should provide the most capability for widespread GPUs. If you need to use a different version of pytorch, you can change
the install script located in the `Dockerfile`.
Note that GPU features require a supported GPU and OS to function, especially in conjunction with Docker, as the service runs in a Docker container.
**Docker does not provide GPU-support for MacOS at this point in time, thus GPU-features of the service do not function on MacOS.**

## Configuration
The service uses the `config.yaml` file located in the root directory for configuration.
For further information about configuration check out this file, all configuration properties are explained using in-file comments.
