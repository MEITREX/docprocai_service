# DocProcAI-Service

This service is designed to process and manage uploaded lecture material (video recordings, documents, slides) to facilitate some advanced features in the MEITREX platform.

## Features
* Splitting of lecture videos into sections based on detected slide changes via computer vision
* OCR of lecture video on screen text
* Transcript & Closed Captions generation for lecture videos
* Generating of text embeddings on a per-section-basis for videos and per-page-basis for documents
* Semantic search/fetching of semantically similar sections of lecture material
* Automatic generation of section titles for the video sections generated

For a deeper dive into the features and considerations made during development, check out our paper on *DocProcAI*.

## Installation
### Additional Installation Steps
This service requires neural network models to function at all. These models need to be downloaded and placed into a `llm_data` folder in the root. This folder is then mounted in the docker container
automatically and can the files inside can then be referenced as seen in the `config.yaml`

> [!CAUTION]
> The service cannot run without at least a sentence embedding model installed!

> [!TIP]
> The `segment_title_generator` and `document_summary_generator` tasks only require LLMs if these features are enabled in the `config.yaml`. They are enabled by default.


### GPU Acceleration
This service requires pytorch to function. As pytorch GPU-support is required for some features of this service, the pip-distributed version of pytorch cannot be used and instead a
platform-specific version has to be used.
By default, pytorch for NVIDIA CUDA 12.4 is used, as this should provide the most capability for widespread GPUs. If you need to use a different version of pytorch, you can change
the install script located in the `Dockerfile`.

> [!WARNING]
> Note that GPU features require a supported GPU and OS to function, especially in conjunction with Docker, as the service runs in a Docker container.
> 
> Docker does not provide GPU-support for MacOS at this point in time, thus GPU-features of the service do not function on MacOS.
>
>  GPU features can be disabled using the `config.yaml`. Additionally, it might be necessary to change the `docker-compose.yaml` file and remove the GPU device reservation.

## Configuration
The service uses the `config.yaml` file located in the root directory for configuration.
For further information about configuration check out this file, all configuration properties are explained using in-file comments.
