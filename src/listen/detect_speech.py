import logging
import torch


class SpeechDetector:
    def __init__(
            self,
            model_name: str,
            state_dict: str = None,         
        ) -> None:
        """Initialize the SpeechDetector class."""
        self.logger = logging.getLogger("SpeechDetector")
        self.logger.info("Initialized SpeechDetector")

        self.logger.info("Loaded pyannote-audio model")