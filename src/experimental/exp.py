import os
import tempfile
import requests
import torch
from describe.frame import FrameDescriptor

def get_caption(image_path: str):

    # Initialize the FrameDescriptor class
    frame_desc = FrameDescriptor(
        model_name="microsoft/git-large-r-textcaps",
        use_gpu=True
    )

    try:
        # Test image description using image path
        desc1 = frame_desc.describe_batch(image_path)
        assert isinstance(desc1, str)

        # Test image description using PyTorch tensor
        img_tensor = torch.FloatTensor(3, 224, 224).uniform_(0, 1)
        desc2 = frame_desc(img_tensor)
        return desc1
        assert isinstance(desc2, str)

    finally:
        pass
