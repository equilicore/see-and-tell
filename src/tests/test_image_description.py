import os
import tempfile
import requests
import torch
from ..describe.frame import FrameDescriptor

def test_frame_descriptor():
    # Initialize the FrameDescriptor class
    frame_desc = FrameDescriptor(
        model_name="microsoft/git-large-r-textcaps"
    )

    # Download a random image and save it to a temporary path
    img_url = "http://loremflickr.com/200/200"
    # Delete temp file before use
    

    img_path = "test_image.png"
    with requests.get(img_url, allow_redirects=True) as response, open(img_path, 'wb+') as out_file:
        out_file.write(response.content)

    try:
        # Test image description using image path
        desc1 = frame_desc(img_path)
        assert isinstance(desc1, str)

        # Test image description using PyTorch tensor
        img_tensor = torch.FloatTensor(3, 224, 224).uniform_(0, 1)
        desc2 = frame_desc(img_tensor)
        assert isinstance(desc2, str)

    finally:
        # Delete the temporary image file
        os.remove(img_path) 