"""Pipeline component that recognizes faces in an image.

This module provides a pipeline component that recognizes faces in an image
and returns the characters in the image. It uses the face_recognition part of
the project
"""

import os
import json

from cntell.core.face.face_recognition import recognize_faces, FR_SingletonInitializer
from cntell.common.component import Component, Input, Output


class Who(Component):
    class WhoInput(Input):
        images: list[str]
        serie: str

    class Faces(Output):
        faces: list[list[str]]

    input_model = WhoInput
    output_model = Faces

    def __init__(self, embeddings_folder: str = None) -> None:
        self.embeddings = {}
        self.embeddings_folder = embeddings_folder
        self._batch_size = 1

    def _load_series_embeddings(self, folder: str) -> dict[str, list[list[float]]]:
        for series in os.listdir(folder):
            self.embeddings[series.split(".")[0]] = json.load(
                open(os.path.join(folder, series))
            )
        return self.embeddings

    def prepare(
        self, use_dir: os.PathLike, use_gpu: bool = False, *args, **kwargs
    ) -> None:
        super().prepare(use_dir, use_gpu, *args, **kwargs)

        if self.embeddings_folder:
            self._load_series_embeddings(self.embeddings_folder)

        FR_SingletonInitializer()
        self._batch_size = kwargs.get('batch_size', 1)
        
    def has_embeddings_for(self, serie: str):
        return serie in self.embeddings
        
    def _run(self, input_data: WhoInput) -> Faces:
        out = []
        for i in range(0, len(input_data.images), self._batch_size):
            batch_images = input_data.images[i:(i+self._batch_size)]
            outputs = recognize_faces(batch_images, self.embeddings[input_data.serie])
            out.extend(outputs)
        return self.Faces(faces=out)

    def run(self, __input: input_model = None, **kwargs) -> output_model:
        return super().run(__input, **kwargs)
