# importing the necessary packages
import os
import random
from random import sample
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms as T 
from torchvision.utils import save_image
import torch

from typing import Union

# a global variable used as a default value for the resize operation in face detection.
RESIZE = 256
# a variable to denote the current working directory
HOME = os.getcwd()

from _collections_abc import Sequence

# the basic file extensions to be considered
BASIC_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp']


def all_images_in_directory(directory: Union[str, Path], img_extensions: Sequence=None) -> list[Path]:
    """This function extrats all the images with certain file extensions from the given directory

    Args:
        directory (Union[str, Path]): the path the directory 
        img_extensions (Sequence): the file extensions to consider

    Returns:
        list[Path]: a list of all paths to images in the given directory
    """

    # the default argument is the BASIC EXTENSIONS variable predefined above
    if img_extensions is None:
        img_extensions = BASIC_EXTENSIONS

    # the glob function does not support regex, the images with a specific extension should
    # be retrieved separately.

    all_images = []
    # iterate through all the extensions
    for ext in img_extensions:
        all_images.extend(directory.glob(f'*/*.{ext}'))
    
    return all_images


from numpy.linalg import norm

def cosine_similarity(A: np.array, B: np.array) -> float:
    """This function calculates the cosine similarity between two numpy arrays 

    Args:
        A (np.array): array 1
        B (np.array): array 2

    Returns:
        float: the cosine similarity between the 2 arrays
    """
    if len(A.shape) == 1 and len(B.shape) == 1 :
        return np.dot(A, B) / (norm(A) * norm(B))
    elif len(A.shape) == 1:
        return np.dot(B, A) / (norm(B, axis=1)) * norm(A)
    elif len(B.shape) == 1:
        return np.dot(A, B) / (norm(A, axis=1)) * norm(B)
    
    return A @ np.transpose(B) / (norm(A, axis=1) * norm(B, axis=1))



