"""Pipeline component that describes a frame in an image."""

import os
import torch
import pickle
import hashlib
from PIL import Image
import numpy as np


from transformers import AutoProcessor, AutoModelForCausalLM
from cntell.common.component import Component, Input, Output


class Describe(Component):
    name = "Describe (Frame Descriptor)"

    class Images(Input):
        images: torch.Tensor | str | list[str]

        class Config:
            arbitrary_types_allowed = True

        def checksum(self) -> str:
            if isinstance(self.images, str):
                with open(self.images, "rb") as f:
                    data = np.array(
                        Image.open(self.images).convert("RGB")
                    )
                    data = data.transpose(2, 0, 1)

            elif isinstance(self.images, list):
                read_images = []
                for image in self.images:
                    read_images.append(
                        np.array(Image.open(image).convert("RGB")).transpose(2, 0, 1)
                    )
                data = np.stack(read_images, axis=0)
            else:
                data = self.images.detach().cpu().numpy()

            _bytes = pickle.dumps(data)
            return hashlib.sha256(_bytes).hexdigest()

    class Captions(Output):
        captions: list[str]

    input_model = Images
    output_model = Captions

    def __init__(self, model_name: str = None) -> None:
        self.descriptor_model = None
        self.processor = None
        self.__model_name = model_name
        self.use_gpu = False

    def _run(self, image: Images) -> Captions:
        images = None
        # If list of image paths, read them
        if isinstance(image.images, list):
            read_images = []
            for image in image.images:
                read_images.append(
                    torch.Tensor(
                        np.array(Image.open(image).convert("RGB"))
                    ).permute(2, 0, 1)
                )

            images = torch.stack(read_images)
        # If a single image path, read it
        elif isinstance(image.images, str):
            images = torch.Tensor(
                np.array(Image.open(image.images).convert("RGB"))
            ).permute(2, 0, 1)
        # Use the tensor if it is already a tensor
        else:
            images = image.images

        out = []

        batch_size = self.__batch_size or 32
        for i in range(0, images.size()[0], batch_size):
            batch_images = images[i : (i + batch_size)]
            inputs = self.processor(
                images=batch_images, return_tensors="pt", padding=True
            )
            if self.use_gpu:
                inputs = inputs.pixel_values.cuda()
            else:
                inputs = inputs.pixel_values

            __out = self.descriptor_model.generate(
                pixel_values=inputs, max_length=20, num_beams=5
            )
            __out = self.processor.batch_decode(__out, skip_special_tokens=True)
            out.extend(__out)
            del inputs
            torch.cuda.empty_cache()

        return self.Captions(captions=out)

    def prepare(
        self, use_dir: os.PathLike = None, use_gpu: bool = False, *args, **kwargs
    ) -> None:
        self.descriptor_model = AutoModelForCausalLM.from_pretrained(self.__model_name)
        self.processor = AutoProcessor.from_pretrained(self.__model_name)
        self.use_gpu = use_gpu
        if use_gpu and torch.cuda.is_available():
            self.descriptor_model.cuda()

        self.__batch_size = kwargs.get("batch_size", 32)

        del self.__model_name

    def run(self, __input: Images = None, **kwargs) -> output_model:
        return super().run(__input, **kwargs)