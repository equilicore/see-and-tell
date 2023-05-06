import argparse
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf



class SpeechToText:
    def __init__(self) -> None:
        """Initialize the AudioCaptionDescriber class."""
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.xvector = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)

    def __call__(self, caption: str) -> str:
        """Uses tts to produce an audio file from a caption.

        Args:
            caption (str): The caption to produce an audio file from.
            save_path (str): The path to save the audio file to.

        Returns:
            str: The path to the saved audio file.
        """
        inputs = self.processor(text=caption, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], self.xvector, vocoder=self.vocoder)
        return speech.numpy()
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("caption", type=str, help="The caption to produce an audio file from.")
    argparser.add_argument("save_path", type=str, help="The path to save the audio file to.")

    describer = SpeechToText()
    audio = describer(**vars(argparser.parse_args()))
    print(audio.shape)
    sf.write(argparser.parse_args().save_path, audio, 16000)