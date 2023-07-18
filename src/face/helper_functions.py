"""
This scripts contains helper functions to work with the file system
"""
import os
from pathlib import Path
from _collections_abc import Sequence
from typing import Union
import itertools
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine

# # a global variable used as a default value for the resize operation in face detection.
# RESIZE = 256
# a variable to denote the current working directory
HOME = os.getcwd()

# the basic file extensions to be considered
BASIC_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp']
DEFAULT_ERROR_MSG = "MAKE SURE THE passed path satisfies the condition passed with it"


def process_save_path(save_path: Union[str, Path, None],
                      dir_ok: bool = True,
                      file_ok: bool = True,
                      condition: callable = None,
                      error_message: str = DEFAULT_ERROR_MSG) -> Union[str, Path, None]:
    """
    Checks and prepares the save_path, ensuring it is absolute and meets certain conditions.

    Args:
        save_path (str): The path to save the file or directory.
        dir_ok (bool, optional): Whether saving as a directory is allowed. Defaults to True.
        file_ok (bool, optional): Whether saving as a file is allowed. Defaults to True.
        condition (callable, optional): A condition function to check the save_path. Defaults to None.
        error_message (str, optional): Error message to raise if the condition fails. Defaults to None.

    Returns:
        str: The absolute path of the saved file or directory.

    Raises:
        AssertionError: If the save_path is invalid or does not meet the conditions.
    """

    if save_path is not None:
        # first make the save_path absolute
        save_path = save_path if os.path.isabs(save_path) else os.path.join(HOME, save_path)
        assert not \
            ((not file_ok and os.path.isfile(save_path)) or
             (not dir_ok and os.path.isdir(save_path))), \
            f'MAKE SURE NOT TO PASS A {"directory" if not dir_ok else "file"}'

        assert condition is None or condition(save_path), error_message

        # create the directory if needed
        if not os.path.isfile(save_path):
            os.makedirs(save_path, exist_ok=True)

    return save_path


def __directory_images(directory: Union[str, Path],
                       img_extensions: Sequence[str] = None) -> list[Path]:
    """
    Retrieve all images in a given directory

    Args:
        directory (Union[str, Path]): The directory path or relative path to retrieve images from.
        img_extensions (Sequence[str], optional): A sequence of image file extensions to filter the results.
        Defaults to None.

    Returns:
        list[Path]: A list of Path objects representing the image files found in the directory.

    Raises:
        AssertionError: If the path passed is not a directory.
    """

    # make sure the 'directory' argument is not None:
    assert directory is not None, "THE 'directory' ARGUMENT CANNOT BE NONE"
    # process the path: make it absolute, make sure it is a directory
    directory = process_save_path(directory, file_ok=False)
    # set the default extensions to work with
    if img_extensions is None:
        img_extensions = BASIC_EXTENSIONS

    # first let's extract all images
    all_images = [directory.glob(f'*/*.{ext}') for ext in img_extensions]
    # all_images is currently a list of lists: it needs to be flattened
    all_images = list(itertools.chain(*all_images))
    return all_images


def build_classes_paths(directory: Union[str, Path],
                        images_extensions: Sequence[str] = None) -> dict[str: list[Path]]:
    """
    Build a dictionary mapping class names to lists of image file paths within the specified directory.

    Args:
        directory (Union[str, Path]): The directory path or relative path to build the class paths from.
        images_extensions: Sequence[str]:

    Returns:
        dict[str: list[Path]]:
        A dictionary mapping class names to lists of Path objects representing the image file paths.
    """
    # before proceeding, 2 conditions must be checked
    # 1. The 'directory' must be a directory
    # 2. The inner files for the directory must be only directories

    directory = directory if os.path.isabs(directory) else os.path.join(HOME, directory)

    inner_directories = [os.path.isdir(os.path.join(directory, file)) for file in os.listdir(directory)]

    assert os.path.isdir(directory) \
           and all(inner_directories), "ALL FILES IN THE DIRECTORY MUST BE DIRECTORIES THEMSELVES"

    # first extract all the image files in the directory
    all_images = __directory_images(directory, images_extensions)

    def class_and_path(image_path: Union[str, Path]) -> tuple[str, Path]:
        """
        Extract the class name and path from the given image path.

        Args:
            image_path (Union[str, Path]): The path of the image file.

        Returns:
            tuple[str, Path]: A tuple containing the class name and Path object representing the image file path.

        """
        # first convert to Path object
        image_path = Path(image_path)
        # extract the parent's directory basename as it represents the class's name
        class_name = os.path.basename(Path(image_path).parent)
        return class_name, image_path

    classes_and_paths = {}
    for img in all_images:
        cls, img = class_and_path(img)
        if cls not in classes_and_paths:
            # use a set for computational efficiency
            classes_and_paths[cls] = set()
        classes_and_paths[cls].add(img)

    # convert the sets to lists
    for cls, paths in classes_and_paths.items():
        classes_and_paths[cls] = list(paths)

    return classes_and_paths


def cosine_similarity(A: np.array, B: np.array) -> Union[float, np.ndarray]:
    """This function calculates the cosine similarity between two numpy arrays

    Args:
        A (np.array): array 1
        B (np.array): array 2

    Returns:
        float: the cosine similarity between the 2 arrays
    """
    # before proceeding, let's make sure to reduce any extra dimensions
    A, B = np.squeeze(A), np.squeeze(B)
    if len(A.shape) == 1 and len(B.shape) == 1:
        return float(np.dot(A, B) / (norm(A) * norm(B)))
    elif len(A.shape) == 1:
        return np.dot(B, A) / (norm(B, axis=-1)) * norm(A)
    elif len(B.shape) == 1:
        return np.dot(A, B) / (norm(A, axis=-1)) * norm(B)

    return A @ np.transpose(B) / (norm(A, axis=-1) * norm(B, axis=-1))


def cos_sim(A: np.ndarray, B: np.ndarray) -> float:
    """
    This function is an iterative version of the cosine similarity function
    created above. It is slower, but guaranteed to be correct. It is used to test the function above.
    """

    # before proceeding, let's make sure to reduce any extra dimensions
    A, B = np.squeeze(A), np.squeeze(B)
    if len(A.shape) == 1 and len(B.shape) == 1:
        res = cosine(A, B)
    elif len(A.shape) == 1:
        res = np.asarray([cosine(A, b) for b in B])
    elif len(B.shape) == 1:
        res = np.asarray([cosine(B, a) for a in A])
    else:
        res = np.asarray([[cosine(a, b) for b in B] for a in A])

    return res
