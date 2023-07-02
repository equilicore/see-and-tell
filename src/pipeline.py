"""The main pipeline for the See And Tell project.

Process an input video in the following way:

1) Splits the video into frames and audio
2) Detects segments with no speech
3) Extracts descriptions for each segment
4) Recognizes faces in each segment and modifies descriptions
5) Generates audio for each description
6) Combines audios and video into a single file
"""

import os
import time
import torch
import hashlib

from . import utils
from .face.who import FaceRecognizer
from .say.caption import SpeechToText
from .listen.speech import SpeechDetector
from .describe.frame import FrameDescriptor


descriptor = FrameDescriptor(
    model_name="microsoft/git-base-textcaps",
    use_gpu=torch.cuda.is_available(),
)

def run_descriptor(*args, **kwargs) -> str: 
    global descriptor
    return descriptor(*args, **kwargs)

        
class SeeAndTell:

    def __init__(self, temp_folder: str, embeddings_folder: str = None) -> None:
        """Initialize the SeeAndTell class."""

        # Initialize the SpeechDetector class
        self.speech_detector = SpeechDetector(
            onset=0.5,
            offset=0.5,
            min_duration_off=0,
            min_duration_on=0,
        )

        self.speech_to_text = SpeechToText()
        self.use_embeddings = embeddings_folder is not None

        self.face_detector = FaceRecognizer()
        if embeddings_folder:
            self.face_detector.load_series_embeddings(embeddings_folder)

        self.temp_folder = temp_folder
        os.makedirs(self.temp_folder, exist_ok=True)

    def describe_video(self, video: str, save_to: str, from_series: str = None) -> None:
        # Step 0: Generate a hash for video name
        # and current time to ensure unique folder name

        dir_name = hashlib.md5((video + str(time.time())).encode()).hexdigest()[:8]

        def get_dir(local_dir_name: str) -> str:
            """Get the path to a directory in the temp folder."""
            os.makedirs(
                os.path.join(self.temp_folder, dir_name, local_dir_name), exist_ok=True
            )
            return os.path.join(self.temp_folder, dir_name, local_dir_name)

        def get_path(file_name: str) -> str:
            """Get the path to a file in the temp folder."""
            os.makedirs(os.path.join(self.temp_folder, dir_name), exist_ok=True)
            return os.path.join(self.temp_folder, dir_name, file_name)

        # Step 1: Extract frames and audio from video
        frames = utils.split_on_frames(video, get_dir("frames"))
        utils.split_on_audio(video, get_path("audio.mp3"))
        frames = [
            os.path.join(get_dir("frames"), frame) for frame in frames
        ]
        frames.sort()
        # Step 2: Get segments with no speech   
        segments = self.speech_detector(get_path("audio.mp3"))
        segments = utils.get_frames_with_no_speech(
            segments, utils.get_length_of_video(video)
        )
        segments = utils.split_segments(segments, 10, 1)
        segments.sort(key=lambda x: x[0])
        # Step 3: Get descriptions for each segment
        descriptions = {}
        for frame in frames:
            descriptions[frame] = (
                run_descriptor(frame)
            )
        descriptions = {i: d.lower() for i, d in descriptions.items()}

        if self.use_embeddings:
            # Step 4: Recognize faces in each segment
            desc_with_faces = []
            desc_with_faces, detections = self.face_detector(
                list(descriptions.keys()), 
                list(descriptions.values()),
                from_series
            )

            # Step 4.1: Get the most described frame for each segment
            frames_to_proceed = []
            for start, end in segments:
                most_described_frame = max(
                    [(i, detections[i]) for i in range(start, end + 1)],
                    key=lambda x: len(x[1])
                )
                frames_to_proceed.append(most_described_frame[0])         
            
            # Step 4.2: Enhance descriptions for each segment
            descriptions = [desc_with_faces[i] for i in frames_to_proceed]
        else:
            frames_to_proceed = [int(s[0]) for s in segments]
            descriptions = list(descriptions.values())
            descriptions = [descriptions[i] for i in frames_to_proceed]

        print(frames_to_proceed, descriptions)

        # Step 5: Generate audio for each description        
        audio_arrays = []
        for description in descriptions:
            audio_array = self.speech_to_text(description)
            audio_arrays.append(audio_array)

        # Step 6: Combine clips
        utils.mix_video_and_audio(video, audio_arrays, frames_to_proceed, save_to)


def run_pipeline(
        video: str,
        output: str,
        temporary_folder: str,
        embeddings_folder: str = './embeddings',
        serie: str = None
):
    """Run the pipeline on a video."""
    see_and_tell = SeeAndTell(temporary_folder, embeddings_folder)
    see_and_tell.describe_video(video, output, serie)
