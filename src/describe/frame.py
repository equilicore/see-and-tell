import argparse
import logging
import torch
import PIL
import numpy as np

from transformers import AutoProcessor, AutoModelForCausalLM
from ..log import get_pipeline_logger

class FrameDescriptor:
    def __init__(
            self,
            model_name: str,
            state_dict_path: str=None,
        ) -> None:
        """Initialize the FrameDescriptor class.

        Args:
            model_name (str): The name of the model to use. Must be a model from the HuggingFace model hub.
            state_dict_path (str, optional): The path to a state dict to load into the model. Defaults to None.
        """
        self.logger = get_pipeline_logger("FrameDescriptor")
        self.descriptor_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        if state_dict_path:
            self.descriptor_model.load_state_dict(torch.load(state_dict_path))

        self.logger.info(f"Initialized FrameDescriptor with model {model_name} and state_dict {state_dict_path}")

    def __call__(self, image: torch.Tensor | str, max_description_length=40) -> str:
        """Generate a description of the image.

        Args:
            image (torch.Tensor | str): The image to describe. Can be a tensor or a path to an image.
            max_description_length (int, optional): The maximum length of the description. Defaults to 20.

        Returns:
            str: The description of the image.
        """

        if isinstance(image, str):
            # Load the image if path is specified
            __image = "" + image
            image = torch.Tensor(
                np.array(
                    PIL.Image.open(image).convert("RGB")
                )
            ).permute(2, 0, 1)
            self.logger.info(f"Loaded image from {__image}")
        
        self.logger.info(f"Processing image")
        inputs = self.processor(images=image, return_tensors="pt")
        out = self.descriptor_model.generate(pixel_values=inputs.pixel_values, max_new_tokens=max_description_length)
        out = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        self.logger.info(f"Processed. Generated description: {out}...")

        return out
            
        

if __name__ == "__main__":
    # Example usage
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str, help="The path to the image to describe.")
    args = argparser.parse_args()

    frame_desc = FrameDescriptor(
        model_name="microsoft/git-base-textcaps"
    )

    frame_desc(args.path)
