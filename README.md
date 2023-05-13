# See and Tell

## Introduction

The project aimed to assist people who are not able or do not want to experience media visually. It produces an audio description of video content and provides an ability for user to understand the context and details of the video. 

The project provides a ready-to-go pipeline that produces a video with audio-description features from an input video. As an additional enhancement for user experience, we provide a possibility to generate audio description in the context of a certain movie or serie, providing names of characters as a part of the video context. 

## Running the project

### Prerequisites

The project is build using `poetry` package manager. To install it, please follow the instructions on the [official website](https://python-poetry.org/docs/#installation). Once you have `poetry` installed, you can install all the dependencies by running:
    
    ```bash
    poetry install
    ```

### Running the pipeline

The pipeline is runnable as a module. To learn about parameters run:

    ```bash
    python -m src.pipeline -h
    ``` 

Here the example or running pipeline assuming you want to describe video "video.mp4" and save the result to "result.mp4":

    ```bash
    python -m src.pipeline video.mp4 result.mp4
    ```