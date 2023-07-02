import os
import json

from .face_recognition import recognize_faces
from ..captions.captions_improved import generate_captions
from ..log import get_pipeline_logger

class FaceRecognizer:
    def __init__(self) -> None:
        self.embeddings: dict[str, dict] = {}
        self.logger = get_pipeline_logger("FaceRecognizer", "yellow")
        self.logger.info("Initialized FaceRecognizer")


    def load_series_embeddings(self, folder: str):
        for series in os.listdir(folder):
            self.embeddings[series.split('.')[0]] = json.load(open(os.path.join(folder, series)))


    def __call__(self, image: list[str], caption: list[str], series: str) -> list[str]:
        """Uses face recognition to identify the characters in an image.

        Args:
            image (list[str]): list of image paths
            caption (list[str]): list of captions
            embeddings (dict): dictionary of embeddings

        Returns:
            list[str]: list of characters
        """
        if series not in self.embeddings:
            return caption

        self.logger.info(f"Recognizing faces in images batch {image} ...")

        # faces = [
        #     recognize_faces(image[i], self.embeddings[series])
        #     for i in range(len(image))
        # ]

        faces = recognize_faces(image, self.embeddings[series])

        self.logger.info(f"Recognized faces in images: {[len(face) for face in faces]}")

        return generate_captions(caption, faces), faces

        

