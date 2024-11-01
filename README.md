# DocProcAI-Service

This service is designed to process and manage uploaded lecture material (video recordings, documents, slides) to facilitate some advanced features in the MEITREX platform.

## Requirements
To run the service locally without docker or for development the following are required:
* [Python 3.12.x](https://www.python.org) 
* [ffmpeg](https://www.ffmpeg.org/about.html) 
* [LibreOffice](https://www.libreoffice.org) 
* [pytorch](https://pytorch.org/get-started/locally/)

See [Installation](#installation) for more information

## Features
* Splitting of lecture videos into sections based on detected slide changes via computer vision
* OCR of lecture video on screen text
* Transcript & Closed Captions generation for lecture videos
* Generating of text embeddings on a per-section-basis for videos and per-page-basis for documents
* Semantic search/fetching of semantically similar sections of lecture material
* Automatic generation of section titles for the video sections generated
* Automatic tag suggestion for media content and assessments

## Installation
This service requires [pytorch](https://pytorch.org/get-started/locally/) to function. As pytorch GPU-support is required for some features of this service, the pip-distributed version of pytorch cannot be used and instead a
platform-specific version has to be used.
By default, pytorch for NVIDIA CUDA 12.4 is used, as this should provide the most capability for widespread GPUs. If you need to use a different version of pytorch, you can change
the install script located in the `Dockerfile`.

> [!CAUTION]
> Note that GPU features require a supported GPU and OS to function, especially in conjunction with Docker, as the service runs in a Docker container.
> 
> Docker does not provide GPU-support for MacOS at this point in time, thus GPU-features of the service do not function on MacOS.
>
>  GPU features can be disabled using the `config.yaml`.

## Configuration
The service uses the `config.yaml` file located in the root directory for configuration.
For further information about configuration check out this file, all configuration properties are explained using in-file comments.
