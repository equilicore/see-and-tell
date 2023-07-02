from collections import defaultdict
import signal
from .describe.frame import FrameDescriptor
from .listen.speech import SpeechDetector
from .say.caption import SpeechToText
from .face.who import FaceRecognizer
from . import utils
from concurrent import futures
from multiprocessing import cpu_count

import hashlib
import time
import os
import argparse
import logging
import atexit
import torch

descriptor = FrameDescriptor(
    model_name="microsoft/git-large-r-textcaps",
    use_gpu=torch.cuda.is_available(),
)

def run_descriptor(*args, **kwargs) -> str: 
    global descriptor
    return descriptor(*args, **kwargs)

        
class SeeAndTell:

    def __init__(self, temp_folder: str, cpus: int = 1, embeddings_folder: str = None, use_gpu = False) -> None:
        """Initialize the SeeAndTell class."""

        # self.descriptor_pool = futures.ProcessPoolExecutor(
        #     max_workers=cpus,
        #     initializer=init_descriptor,
        # )

        # self.descriptor = FrameDescriptor(
        #     model_name="microsoft/git-base-textcaps"
        # )

        # Initialize the SpeechDetector class
        self.speech_detector = SpeechDetector(
            onset=0.5,
            offset=0.5,
            min_duration_off=0,
            min_duration_on=0,
        )

        self.speech_to_text = SpeechToText()

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
        # Step 2: Get segments with no speech   
        segments = self.speech_detector(get_path("audio.mp3"))
        segments = utils.get_frames_with_no_speech(
            segments, utils.get_length_of_video(video)
        )
        segments = utils.split_segments(segments, 10, 1)
        segments.sort(key=lambda x: x[0])
        # Step 3: Get descriptions for each segment
        desc_with_faces = []


        descriptions = {}
        for frame in frames:
            descriptions[frame] = (
                run_descriptor(frame)
            )

        descriptions = {i: d.lower() for i, d in descriptions.items()}

        desc_with_faces, detections = self.face_detector(
            list(descriptions.keys()), 
            list(descriptions.values()),
            from_series
        )

        frames_to_proceed = []
        for start, end in segments:
            most_described_frame = max(
                [(i, detections[i]) for i in range(start, end + 1)],
                key=lambda x: len(x[1])
            )
            frames_to_proceed.append(most_described_frame[0])         
            print(start, end)
                   
            print(most_described_frame[0], most_described_frame[1])
        
        # frames_to_proceed = [
        #     frames[segment[0] - 1]
        #     for segment in segments
        # ]

        descriptions = [desc_with_faces[i] for i in frames_to_proceed]
        
        # descriptions = ["some caption for the video"]
        # Step 4: Produce audio for each description
        audio_arrays = []
        for description in descriptions:
            audio_array = self.speech_to_text(description)
            audio_arrays.append(audio_array)
        # Step 5: Combine audio clips
        utils.mix_video_and_audio(video, audio_arrays, frames_to_proceed, save_to)



# logging.basicConfig( 
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[logging.StreamHandler()]
# )

def run_pipeline(
        video: str,
        output: str,
        temporary_folder: str,
        cpus: int = 1,
        embeddings_folder: str = './embeddings',
        serie: str = None
):
    """Run the pipeline on a video."""
    see_and_tell = SeeAndTell(temporary_folder, cpus, embeddings_folder, use_gpu=torch.cuda.is_available())

    def signal_handler(signum, frame):
        print("Caught keyboard interrupt, cancelling pending tasks...")
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)

    see_and_tell.describe_video(video, output, serie)
