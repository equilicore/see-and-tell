import argparse
import logging
import tempfile
import soundfile as sf
import pyttsx3
from ..log import get_pipeline_logger

class SpeechToText:
    def __init__(self) -> None:
        """Initialize the AudioCaptionDescriber class."""
        self.engine = pyttsx3.init()
        self.processing_dir = tempfile.TemporaryDirectory()
        self.engine.setProperty('rate', 300)
        self.logger = get_pipeline_logger("SpeechToText", "blue")
        self.logger.info("Initialized SpeechToText")

    def __call__(self, caption: str) -> str:
        """Uses tts to produce an audio file from a caption.

        Args:
            caption (str): The caption to produce an audio file from.
            save_path (str): The path to save the audio file to.

        Returns:
            str: The path to the saved audio file.
        """
        logging.info(f"Generating audio from caption {caption} ...")
        file = self.processing_dir.name + "/audio.wav"
        self.engine.save_to_file(caption, file)
        self.engine.runAndWait()
        speech, _ = sf.read(file)
        logging.info(f"Generated audio from caption: {caption}")
        return speech
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("caption", type=str, help="The caption to produce an audio file from.")
    argparser.add_argument("save_path", type=str, help="The path to save the audio file to.")
    args = argparser.parse_args()
    describer = SpeechToText()
    audio = describer(args.caption)
    sf.write(argparser.parse_args().save_path, audio, 16000)