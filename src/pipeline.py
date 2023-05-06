from .describe.frame import FrameDescriptor
from .listen.speech import SpeechDetector
from .say.caption import SpeechToText
from . import utils

import hashlib
import time
import os
import argparse

class SeeAndTell:
    def __init__(self, temp_folder: str) -> None:
        """Initialize the SeeAndTell class."""
        self.image_descriptor = FrameDescriptor(
            model_name="microsoft/git-base-textcaps"
        )
        
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
        
        dir_name = hashlib.md5((video + str(time.time())).encode()).hexdigest()

        def get_dir(local_dir_name: str) -> str:
            os.makedirs(os.path.join(self.temp_folder, dir_name, local_dir_name), exist_ok=True)
            return os.path.join(self.temp_folder, dir_name, local_dir_name)
        
        def get_path(file_name: str) -> str:
            os.makedirs(os.path.join(self.temp_folder, dir_name), exist_ok=True)
            return os.path.join(self.temp_folder, dir_name, file_name)
        

        # Step 1: Extract frames and audio from video
        frames = utils.split_on_frames(video, get_dir("frames"))
        audio = utils.split_on_audio(video, get_path("audio.mp3"))
        
        # Step 2: Get segments with no speech
        segments = self.speech_detector(get_path("audio.mp3"))
        segments = utils.get_frames_with_no_speech(segments, utils.get_length_of_video(video))
        segments = utils.split_segments(segments, 10, 0)

        # Step 3: Get descriptions for each segment
        descriptions = []
        for segment in segments:
            description = self.image_descriptor(os.path.join(get_dir('frames'), frames[segment[0]]))
            descriptions.append(description)

        # Step 4: Produce audio for each description
        audio_arrays = []
        for description in descriptions:
            audio_array = self.speech_to_text(description)
            audio_arrays.append(audio_array)

        # Step 5: Combine audio clips
        utils.mix_video_and_audio(video, audio_arrays, [s[0] for s in segments], save_to)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("video", help="The path to the video to describe.")
    argparser.add_argument("output", help="The path to save the output video to.")
    argparser.add_argument("--temp", help="The path to save temporary files to.")
    args = argparser.parse_args()

    see_and_tell = SeeAndTell(args.temp)
    see_and_tell.describe_video(args.video, args.output)

