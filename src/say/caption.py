import argparse
import tempfile
import soundfile as sf
import pyttsx3
from ..log import get_pipeline_logger

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

class SpeechToText:
    def __init__(self) -> None:
        """Initialize the AudioCaptionDescriber class."""
        self.engine = pyttsx3.init()
        self.processing_dir = tempfile.TemporaryDirectory()
        # self.engine.setProperty('rate', 300)
        self.logger = get_pipeline_logger("SpeechToText", "blue")
        self.logger.info("Initialized SpeechToText")

        
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)



    def __call__(self, caption: str) -> str:
        """Uses tts to produce an audio file from a caption.

        Args:
            caption (str): The caption to produce an audio file from.
            save_path (str): The path to save the audio file to.

        Returns:
            str: The path to the saved audio file.
        """
        inputs = self.processor(text=caption, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        # self.logger.info(f"Generating audio from caption {caption} ...")
        # file = self.processing_dir.name + "/audio.wav"
        # self.engine.save_to_file(caption, file)
        # self.engine.runAndWait()
        # speech, _ = sf.read(file)
        # self.logger.info(f"Generated audio from caption: {caption}")
        return speech.numpy()
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("caption", type=str, help="The caption to produce an audio file from.")
    argparser.add_argument("save_path", type=str, help="The path to save the audio file to.")
    args = argparser.parse_args()
    describer = SpeechToText()
    audio = describer(args.caption)
    sf.write(argparser.parse_args().save_path, audio, 16000)