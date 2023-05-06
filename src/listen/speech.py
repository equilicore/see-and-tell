import argparse
import logging

from pyannote.audio.pipelines import VoiceActivityDetection
from ..log import get_pipeline_logger

class SpeechDetector:
    def __init__(
            self,
            model_name: str = "anilbs/segmentation",
            **kwargs
        ) -> None:
        """Initialize the SpeechDetector class."""
        self.logger = get_pipeline_logger("SpeechDetector")
        self.logger.info("Initialized SpeechDetector")
        self.pipeline = VoiceActivityDetection(segmentation=model_name)
        self.pipeline.instantiate(kwargs)
        self.logger.info("Loaded pyannote-audio model")

    def __call__(self, audio: str) -> list[tuple[float, float]]:
        """Detect speech in an audio file.

        Args:
            audio (str): The path to the audio file.

        Returns:
            list[tuple[float, float]]: A list of tuples containing the start and end times of each speech segment.
        """
        self.logger.info(f"Loading audio from {audio}")
        self.logger.info(f"Detecting speech")
        speech_segments = self.pipeline(audio)
        self.logger.info(f"Detected speech: {speech_segments}")
        return list(speech_segments.get_timeline())
    

if __name__ == "__main__":
    # Example usage
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str, help="The path to the audio file to detect speech in.")
    args = argparser.parse_args()


    logging.basicConfig(level=logging.INFO)
    speech_detector = SpeechDetector(
        onset=0.5,
        offset=0.5,
        min_duration_off=0,
        min_duration_on=0,
    )
    print(speech_detector(args.path))