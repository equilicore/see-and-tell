import signal
from .describe.frame import FrameDescriptor
from .listen.speech import SpeechDetector
from .say.caption import SpeechToText
from . import utils

import hashlib
import time
import os
import argparse
from concurrent import futures
from multiprocessing import cpu_count
import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def init_descriptor() -> FrameDescriptor:
    global descriptor
    descriptor = FrameDescriptor(
        model_name="microsoft/git-base-textcaps"
    )

def run_descriptor(*args, **kwargs) -> str: 
    global descriptor
    return descriptor(*args, **kwargs)

        
class SeeAndTell:
    def __init__(self, temp_folder: str, cpus: int = 1) -> None:
        """Initialize the SeeAndTell class."""
        self.descriptor_pools = futures.ProcessPoolExecutor(
            max_workers=cpus,
            initializer=init_descriptor,
        )

        # Initialize the SpeechDetector class
        self.speech_detector = SpeechDetector(
            onset=0.5,
            offset=0.5,
            min_duration_off=0,
            min_duration_on=0,
        )

        self.speech_to_text = SpeechToText()

        self.temp_folder = temp_folder
        os.makedirs(self.temp_folder, exist_ok=True)

    def describe_video(self, video: str, save_to: str) -> None:
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

        # Step 2: Get segments with no speech
        segments = self.speech_detector(get_path("audio.mp3"))
        segments = utils.get_frames_with_no_speech(
            segments, utils.get_length_of_video(video)
        )
        segments = utils.split_segments(segments, 10, 2)

        # Step 3: Get descriptions for each segment
        descriptions = []
        # Describe frames in parallel
        for segment in segments:
            descriptions.append(
                self.descriptor_pools.submit(
                    run_descriptor,
                    os.path.join(get_dir("frames"), frames[segment[0]]),
                    4 * (segment[1] - segment[0]),
                )
            )

        descriptions = [description.result() for description in descriptions]

        # Step 4: Produce audio for each description
        audio_arrays = []
        for description in descriptions:
            audio_array = self.speech_to_text(description)
            audio_arrays.append(audio_array)

        # Step 5: Combine audio clips
        utils.mix_video_and_audio(video, audio_arrays, segments, save_to)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("video", help="The path to the video to describe.")
    argparser.add_argument("output", help="The path to save the output video to.")
    argparser.add_argument("--temp", help="The path to save temporary files to.", default=".temp")
    argparser.add_argument("--cpus", help="The number of cpus to use.", type=int, default=1)

    args = argparser.parse_args()

    see_and_tell = SeeAndTell(args.temp, args.cpus or cpu_count())

    def signal_handler(signum, frame):
        print("Caught keyboard interrupt, cancelling pending tasks...")
        see_and_tell.descriptor_pools.shutdown(wait=False)
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)

    try:
        see_and_tell.describe_video(args.video, args.output)
    except KeyboardInterrupt:
        see_and_tell.descriptor_pools.shutdown(wait=False)