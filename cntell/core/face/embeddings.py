"""
This scripts contains the functionality needed to build the embeddings used for face recognition
"""
import torch
import numpy as np
import os
import itertools
import json

from pathlib import Path
from typing import Union
from _collections_abc import Sequence
from torchvision.utils import save_image
from PIL.Image import Image
from torchvision import transforms as trans

from cntell.core.face.helper_functions import build_classes_paths, process_save_path
# from src.face.utilities import FACE_DETECTOR, ENCODER, DEVICE
from cntell.core.face.utilities import FR_SingletonInitializer

# the current directory
HOME = os.getcwd()
MIN_RESIZE = 256
RESIZE_THRESHOLDS1 = [256, 300, 360, 480]
RESIZE_THRESHOLDS2 = [480, 512, 600, 640, 750, 800, 1000]
# set torch's manual seed
torch.manual_seed(69)


def build_embeddings(images: Sequence[Union[Path, str, np.ndarray, torch.tensor]],
                     face_detector=None,
                     encoder=None,
                     keep_all: bool = False,
                     return_faces: bool = False,
                     save_faces: Union[str, Path] = None,
                     batch: bool = True,
                     ) -> Union[list[list[float]], tuple[list[list[float]], list[list[float]]]]:
    """
    This function takes a sequence of images as input and returns the embeddings of the faces detected in those
    images. It has the following parameters:

    -  images : A sequence of images: either paths to images, or actual images converted to numpy array / tensors.
    -  face_detector : The face detection algorithm.
    -  encoder : The encoder algorithm.
    -  keep_all : boolean flag: process all detected faces in a given image. Default is True
    -  return_faces : boolean flag: return faces objects or not. Default is False
    -  save_faces : An optional directory path to save the extracted face images.
    -  batch : boolean flag: process images in batches or not. Default to True .

    The function set the defaults arguments, processes any given paths, then prepare data for batch processing
    according to the ```batch``` boolean flag

    The extracted images are passed through the encoder to create the embeddings
    The function returns the embeddings along with the 'faces' according to the 'return_faces' boolean flag"""

    fr_singleton = FR_SingletonInitializer()
    # set the default arguments
    if face_detector is None:
        face_detector = fr_singleton.get_face_detector()
        # set the behavior of the face detection algorithm
        face_detector.keep_all = keep_all

    if encoder is None:
        encoder = fr_singleton.get_encoder()

    # process save_faces: it is only a directory
    save_faces = process_save_path(save_faces, file_ok=False)

    # make sure to convert the images to a format we can work with
    images = [Image.open(str(img)) if isinstance(img, (str, Path)) else img for img in images]

    if batch:
        final_images = resize_images(images)
        face_images = face_detector.forward(final_images.to(fr_singleton.get_device()))
    else:
        face_images = [face_detector.forward(img) for img in images]

    # remove any None objects
    if isinstance(face_images, list):
        face_images = [fi for fi in face_images if fi is not None]

    # flatten the output if keep_all is set to True
    if keep_all:
        # according to the documentation of the forward function of the 'mtcnn' class
        #: https://github.com/timesler/facenet-pytorch/blob/fa70227bd5f02209512f60bd10e7e66877fdb4f6/models/mtcnn.py
        # the input will be of n * image_shape where 'n' is the number of detected images

        # the idea is to flatten the list: reducing the output to 4 dimensions where each
        # inner element is an image: 3-dimensional
        face_images = list(itertools.chain(*list(face_images)))

    # now, checking if all images are of the same dimensions
    shape = face_images[0].shape
    for face in face_images:
        assert face.shape == shape

    # save the faces extracted from the images
    if save_faces is not None:
        # choose the face_images or final_face_images to
        for index, img in enumerate(face_images):
            face_save_path = os.path.join(save_faces, f'cropped_image_{index}.jpg')
            save_image(img, face_save_path)

    # encoding the faces (creating the embeddings) is performed in a single batch
    # regardless of the value of the ```batch``` argument, so if batch is set to False
    # face_images is a list of torch.tensors

    # to get the embeddings, we simply: convert the list of tensors to one batched tensor
    if not batch:
        face_images = torch.stack(face_images).to(torch.float32).to(fr_singleton.get_device())

    embeddings = encoder(face_images).detach().cpu().numpy()

    assert len(embeddings) == len(face_images) and len(embeddings.shape) == 2

    embeddings = embeddings.tolist()

    if return_faces:
        return embeddings, face_images

    return embeddings


def build_classes_embeddings(directory: Union[str, Path],
                             save_embedding: Union[str, Path] = None,
                             save_faces: Union[str, Path, None] = None,
                             images_extensions: Sequence[str] = None,
                             batch: bool = True) -> dict:
    # first let's build the dictionary that maps each class to its images
    classes_paths = build_classes_paths(directory, images_extensions=images_extensions)
    embeddings_map = {}

    if save_faces is not None:
        save_faces = save_faces if os.path.isabs(save_faces) else os.path.join(HOME, save_faces)

    for cls, images in classes_paths.items():
        embeddings_map[cls] = build_embeddings(images,
                                               keep_all=False,  # the images are assumed to have only one face
                                               save_faces=os.path.join(save_faces, cls) if save_faces else None,
                                               batch=batch)

    # process save_embedding path
    # the main condition if the path is a file, it must have the .json extension
    save_embedding = process_save_path(save_embedding,
                                       file_ok=True,
                                       dir_ok=True,
                                       condition=lambda path: not os.path.isfile(path) or str(path).endswith('.json'),
                                       error_message='MAKE SURE THE PASSED FILE IS A .json FILE')
    if save_embedding is not None:
        save_embedding = os.path.join(save_embedding, 'embeddings.json') if os.path.isdir(
            save_embedding) else save_embedding

        with open(save_embedding, 'w') as f:
            json.dump(embeddings_map, f, indent=4)

    # return the embeddings
    return embeddings_map


# this part of the file is intended to add batching to the face detection phase
# the function current does not give the desired results and should be investigated in more depth
def __to_numpy(image: [np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        return image.cpu().numpy()
    return np.asarray(image)


def __find_resize(images: Sequence[Image, np.ndarray, torch.Tensor]) -> tuple[int, int]:
    dims = [sorted(__to_numpy(i).shape)[1:] for i in images]
    min_dim1, min_dim2 = min([d[0] for d in dims]), min([d[1] for d in dims])

    def resize(dim: int, thresholds: list[int]) -> int:
        lower_dim = 2 ** int(np.log2(dim))

        if lower_dim > thresholds[-1]:
            return lower_dim

        for size in thresholds[::-1]:
            if lower_dim >= size:
                return size

        # make sure to return the minimum size through
        return max(lower_dim, thresholds[0])

    return resize(min_dim1, RESIZE_THRESHOLDS1), resize(min_dim2, RESIZE_THRESHOLDS2)


def resize_images(images: Sequence[Image, np.ndarray, torch.Tensor],
                  thresholds1: Sequence[int] = None,
                  thresholds2: Sequence[int] = None) -> Union[torch.Tensor, np.ndarray]:
    # the idea here is to resize the images while discarding those
    # that do not satisfy certain dimensionality requirements
    if thresholds1 is None:
        thresholds1 = RESIZE_THRESHOLDS1

    if thresholds2 is None:
        thresholds2 = RESIZE_THRESHOLDS2

    thresholds1 = sorted(thresholds1)
    thresholds2 = sorted(thresholds2)

    # step1: convert to np.ndarray
    np_images = [__to_numpy(i) for i in images]
    # step2: filter the images
    filtered_indices = [index for index, i in enumerate(np_images) if
                        sorted(i.shape)[1] >= thresholds1[0] and sorted(i.shape)[2] >= thresholds2[0]]

    # use the original images as pytorch.transform module's function do not accept np.ndarray as input
    images = [images[i] for i in filtered_indices]

    # step3: extract the new sizes
    batch_shape = __find_resize(images)

    # step 4: define the transformations
    t = trans.Compose([trans.Resize(size=batch_shape), trans.ToTensor()])

    # step 5: apply the transformations
    tensor = torch.stack([t(img) for img in images])
    # The current implementation will return a tensor of the following shape (batch_size, H, W, C)
    tensor = torch.permute(input=tensor, dims=(0, 2, 3, 1))
    return tensor
