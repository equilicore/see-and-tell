import os
import tempfile
import requests
import torch
from ..describe.frame import FrameDescriptor

def get_caption(image_path: str, use_gpu: bool):

    # Initialize the FrameDescriptor class
    frame_desc = FrameDescriptor(
        model_name="microsoft/git-large-r-textcaps",
        use_gpu=use_gpu
    )

    try:
        # Test image description using image path
        desc1 = frame_desc.describe_batch(image_path, )
        # assert isinstance(desc1, str)
        return desc1
    finally:
        pass
