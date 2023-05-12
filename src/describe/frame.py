import argparse
import multiprocessing as mp
import os
import torch
import PIL
import numpy as np

from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from ..log import get_pipeline_logger

class FrameDescriptor:
    def __init__(
            self,
            model_name: str,
            state_dict_path: str=None,
            use_gpu=True,
        ) -> None:
        """Initialize the FrameDescriptor class.

        Args:
            model_name (str): The name of the model to use. Must be a model from the HuggingFace model hub.
            state_dict_path (str, optional): The path to a state dict to load into the model. Defaults to None.
        """
        self.logger = get_pipeline_logger("FrameDescriptor", 'green')
        self.descriptor_model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        if state_dict_path:
            self.descriptor_model.load_state_dict(torch.load(state_dict_path))

        if use_gpu and torch.cuda.is_available():
            self.descriptor_model.cuda()

        self.use_gpu = use_gpu

        self.logger.info(f"Initialized FrameDescriptor with model {model_name} and state_dict {state_dict_path}")

    def describe_batch(self, images: list[str]) -> list[str]:
        """Describe a batch of images.

        Args:
            images (list[str]): The list of images to describe.

        Returns:
            list[str]: The list of descriptions of the images.
        """
        read_images = []
        for image in images:
            read_images.append(
                np.array(
                    PIL.Image.open(image).convert("RGB")
                )
            )


        self.logger.info(f"Processing batch of images {list(map(os.path.basename, images))}")
        inputs = self.processor(images=read_images, return_tensors="pt", padding=True)
        if self.use_gpu:
            inputs = inputs.pixel_values.cuda()
        else:
            inputs = inputs.pixel_values
        out = self.descriptor_model.generate(pixel_values=inputs, max_length=20, num_beams=5)
        out = self.processor.batch_decode(out, skip_special_tokens=True)
        return out
    

    def __call__(self, image: torch.Tensor | str, max_description_length=40) -> str:
        """Generate a description of the image.

        Args:
            image (torch.Tensor | str): The image to describe. Can be a tensor or a path to an image.
            max_description_length (int, optional): The maximum length of the description. Defaults to 20.

        Returns:
            str: The description of the image.
        """
        __image = "tensor_image"
        if isinstance(image, str):
            # Load the image if path is specified
            __image = "" + image
            image = torch.Tensor(
                np.array(
                    PIL.Image.open(image).convert("RGB")
                )
            ).permute(2, 0, 1)
                
        self.logger.info(f"Processing image {__image}")
        inputs = self.processor(images=image, return_tensors="pt")
        out = self.descriptor_model.generate(**inputs, max_length=max_description_length, num_beams=5)
        out = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        self.logger.info(f"Processed {__image}. Generated description: {out}")
        return out
    
            
        

if __name__ == "__main__":
    # Example usage
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str, help="The path to the image to describe.")
    argparser.add_argument("--use-dir", action="store_true", help="Whether to use the path as a directory.")
    args = argparser.parse_args()

    frame_desc = FrameDescriptor(
        model_name="microsoft/git-base-textcaps",
        use_gpu=True
    )

    if args.use_dir:
        # Describe all images in the directory
        images = os.listdir(args.path)
        images = list(map(lambda x: os.path.join(args.path, x), images))
        descriptions = frame_desc.describe_batch(images)
        for image, description in zip(images, descriptions):
            print(f"Image {image} has description {description}")
    else:
        frame_desc(args.path)
