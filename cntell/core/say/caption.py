"""Pipeline component that produces an audio file from a caption."""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

from cntell.common.component import Component, Input, Output


class Say(Component):
    """Pipeline component that produces an audio file from a caption."""

    class Captions(Input):
        texts: list[str]

    class AudioArrays(Output):
        audio: list[np.ndarray]

        class Config:
            arbitrary_types_allowed = True

    name = "Say (text-to-speech)"
    input_model = Captions
    output_model = AudioArrays

    processor: SpeechT5Processor = None
    model: SpeechT5ForTextToSpeech = None
    vocoder: SpeechT5HifiGan = None
    speaker_embeddings: torch.Tensor = None

    def prepare(self, use_dir: os.PathLike = None, use_gpu = False, *args, **kwargs) -> None:
        """Downloads TTS models and initializes them.

        Args:
            use_dir (os.PathLike, optional): The directory to use. Defaults to None.
        """
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation", data_dir=use_dir
        )
        
        self._device = torch.device('cuda' if use_gpu else 'cpu')
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(self._device)
        if use_gpu:
            self.model.to(torch.device('cuda'))
            self.vocoder.to(torch.device('cuda'))

    def _run(self, caption: Captions) -> AudioArrays:
        """Uses tts to produce an audio file from a caption.

        Args:
            caption (str): The caption to produce an audio file from.
            save_path (str): The path to save the audio file to.

        Returns:
            str: The path to the saved audio file.
        """
        out = []
        for text in caption.texts:
            inputs = self.processor(text=text, return_tensors="pt")
            speech = self.model.generate_speech(
                inputs["input_ids"].to(self._device), self.speaker_embeddings, vocoder=self.vocoder
            )
            out.append(speech.cpu().numpy())
        return self.AudioArrays(audio=out)

    def run(self, __input: input_model = None, **kwargs) -> output_model:
        return super().run(__input, **kwargs)
