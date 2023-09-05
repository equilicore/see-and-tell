"""This script contains the basis of the minimal face recognition system.
THE DIFFERENT FUNCTIONS ARE BUILT ON TOP OF facenet-pytorch library
available at: https://github.com/timesler/facenet-pytorch/blob/master
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import itertools

from pathlib import Path
from typing import Union
from PIL import Image
from torchvision.utils import save_image
from _collections_abc import Sequence

from cntell.core.face.embeddings import build_embeddings
from cntell.core.face.utilities import CONFIDENCE_THRESHOLD, REPORT_THRESHOLD, FR_SingletonInitializer
from cntell.core.face.helper_functions import cosine_similarity, process_save_path  # ,cos_sim

HOME = os.getcwd()


def __read_embeddings(p: Union[str, Path]):
    path = os.path.join(HOME, p) if os.path.isabs(p) else p
    assert os.path.basename(p).endswith('.json'), "THE FILE MUST BE a .json file"

    with open(path, 'r') as f:
        return json.load(f)


def __predict_one_image(face_embeddings: np.ndarray,
                        reference_embeddings: dict[str: np.ndarray],
                        report_threshold: float,
                        with_index: bool = False) -> Union[list[str], dict[int, str]]:
    """
    Args:
    -  face_embeddings : The embeddings of faces detected in a single image.
    -  reference_embeddings : A dictionary mapping each class to representative embeddings.
    -  report_threshold : A threshold value for similarity scores. A face will be paired to a certain class only if
        the similarity between an image's embeddings and the class' embeddings.
    -  with_index : whether to return only the classes or a dictionary {indices: classes}

    Returns:
        The most probable pairing between the detected faces and the classes predefined by the embeddings' dictionary

    """

    # the first step is to build a matrix of cos distances between each instance and all the embeddings
    sims = np.asarray(
        [[np.mean(cosine_similarity(fe, ref)) for cls, ref in reference_embeddings.items()]
         for fe in face_embeddings])

    # a couple of assert statements to make sure everything is working correctly under the hood
    assert sims.size == 0 or \
           (len(sims.shape) == 2 and all([isinstance(n, np.number) and np.abs(n) <= 1 for n in sims.flatten()]))
    # create a dataframe for easier manipulation
    if sims.size == 0:
        return {} if with_index else []

    sims = pd.DataFrame(data=sims, index=list(range(len(face_embeddings))), columns=list(reference_embeddings.keys()))

    predictions = {}

    while not sims.empty:
        # first find the minimum value in the dataframe
        max_sim = np.amax(sims.values)
        if max_sim < report_threshold:
            # the min distance is already above the threshold. The predictions are bad
            # and thus will be discarded
            break
        # locate it
        indices, columns = np.where(sims == max_sim)
        # extract the corresponding noun phrase and prediction
        best_pred = list(sims.columns)[columns[0]]
        best_index = list(sims.index)[indices[0]]

        predictions[indices[0]] = best_pred
        # don't forget to drop the column and the index from the table
        sims.drop(columns=best_pred, index=best_index, inplace=True)

    if with_index:
        return predictions

    return [cls for _, cls in predictions.items()]


def __debug_display(face_images: Union[np.ndarray, torch.Tensor],
                    face_embeddings: np.ndarray,
                    reference_embeddings: dict[str: np.ndarray]):
    for f_img, f_emb in zip(face_images, face_embeddings):
        for prediction, r_emb in reference_embeddings.items():
            # r1, r2 = cosine_similarity(r_emb, f_emb), 1 - cos_sim(r_emb, f_emb)
            cos = cosine_similarity(r_emb, f_emb)
            cos = np.mean(cos, axis=-1)
            # commented code used for testing
            # c = 1 - np.mean(cos_sim(r_emb, f_emb), axis=-1)
            # assert np.abs(c - cos) <= 10 ** -2
            print(f"{prediction}:\t{cos}")
        print("#" * 50)

        if isinstance(f_img, torch.Tensor):
            plt.imshow(f_img.detach().permute(1, 2, 0).cpu().numpy())
        elif isinstance(f_img, np.ndarray):
            plt.show(f_img)

        plt.axis('off')
        plt.show()

    print(f"\n{'#' * 100}\n")


def recognize_faces(images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]],
                    embeddings: Union[str, Path, dict],
                    possible_classes: Sequence[str] = None,
                    face_detector=None,
                    encoder=None,
                    keep_all=True,
                    confidence_threshold: float = CONFIDENCE_THRESHOLD,
                    report_threshold: float = REPORT_THRESHOLD,
                    return_bbox: bool = False,
                    save_faces: Union[str, Path, None] = None,
                    debug: bool = False,
                    ) -> list[list[str]]:
    """
    This function is used to recognize faces in given images.
    It uses pre-computed embeddings to recognize the faces and returns the predicted classes for each face.

    Args:
    -  images : A sequence of images
    -  embeddings : the path to the embeddings file, or a dictionary containing the embeddings.
    -  possible_classes : A sequence of strings representing the possible classes.
            If not provided, all classes present in the embeddings will be considered.
    -  face_detector : A face detector object. If not provided, a default face detector will be used.
    -  encoder : An encoder object. If not provided, a default encoder will be used.
    -  keep_all : A boolean indicating whether to keep all detected faces or only the best one. Default is True.
    -  confidence_threshold : the confidence threshold for face detection.
            Faces with confidence below this threshold will be discarded.
    -  report_threshold : the report threshold for face recognition.
            Faces with similarity score below this threshold will be discarded.
    -  return_bbox : whether to return bounding boxes of the detected faces. Default is False.
    -  save_faces : the path to save the detected faces. If not provided, faces will not be saved.
    -  debug : A boolean indicating whether to run the function in debug mode. Default is False.

    Returns:
    a list of strings, where each string is the predicted class for a face in the images.
    If  return_bbox  is True, the function returns a list of tuples,
    where each tuple contains a predicted class and a bounding box for a face
    """

    embeddings = __read_embeddings(embeddings) if isinstance(embeddings, (str, Path)) else embeddings
    # the possible classes
    possible_classes = list(embeddings.keys()) if possible_classes is None else possible_classes

    # make sure the classes used are part of the embeddings keys
    assert set(possible_classes).issubset(set(embeddings.keys())), "THE CLASSES PASSED MUST BE PRESENT IN THE " \
                                                                   "EMBEDDINGS"

    # limit the classes to only those selected and convert the actual embeddings to numpy arrays
    embeddings = dict([(p_c, np.asarray(embeddings[p_c])) for p_c in possible_classes])

    # define the FR_SingletonInitializer
    fr_singleton = FR_SingletonInitializer()

    if face_detector is None:
        face_detector = fr_singleton.get_face_detector()
        face_detector.keep_all = keep_all

    encoder = fr_singleton.get_encoder() if encoder is None else encoder

    # convert the passed images to working types
    def convert_image(image):
        if isinstance(image, (str, Path)):
            return Image.open(str(image))
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        # otherwise the image is a tensor and should be left as it is
        return image

    images = [convert_image(img) for img in images]

    def filter_output(output: Union[tuple[Sequence, Sequence], tuple[Sequence, Sequence, Sequence]]):
        # the output will be a tuple (a sequence of faces, a sequence of probabilities)
        try:
            if return_bbox:
                output_f, output_p, output_b = output
                return [(face, bb) for face, p, bb in zip(output_f, output_p, output_b) if p >= confidence_threshold]

            output_f, output_p = output
            return [face for face, p in zip(output_f, output_p) if p >= confidence_threshold]

        except (TypeError, ValueError):
            return []

    # extract the faces from the images
    detector_output = [face_detector.forward(image, return_prob=True) for image in images]

    if return_bbox:
        detector_boxes = [face_detector.detect(image) for image in images]
        # merge the face recognition and the
        # TODO: run the code with 'return_bbox=True' to make sure it works correctly,
        detector_output = detector_output + (detector_boxes[0],)

    # filter the model's output
    detector_output = [filter_output(o) for o in detector_output]

    # now each item in the list is a tuple (faces, bounding boxes(optional))
    # where the probabilities are higher than the minimum threshold
    # the number of faces detected in each face should be saved so the results could be matched
    # after the batched inference
    images_to_faces = {}
    start = 0
    for index, faces in enumerate(detector_output):
        length = len(faces)
        images_to_faces[index] = (start, start + length)
        start += length

    # flatten the list
    detector_output = list(itertools.chain(*detector_output))

    if return_bbox:
        faces = [d[0] for d in detector_output]
    else:
        faces = detector_output.copy()

    # consider the case where there is no face detected with high enough probability: return an empty list
    # TODO: Make sure this is an actual good fix
    if len(faces) == 0:
        return [[] for _ in images]

    faces = torch.stack(faces).to(torch.float32).to(fr_singleton.get_device())
    # convert to tensor
    face_embeddings = encoder(faces).detach().cpu().numpy()

    assert len(face_embeddings) == len(faces)

    def process_one_image(face_images,
                          face_embeddings,
                          reference_embeddings: dict,
                          save_faces: bool,
                          debug: bool):
        if debug:
            __debug_display(face_images, face_embeddings, reference_embeddings)
        # predict
        predictions = __predict_one_image(face_embeddings,
                                          reference_embeddings,
                                          report_threshold=report_threshold,
                                          with_index=True)

        # save the faces with their predictions:
        if save_faces is not None:
            save_faces = process_save_path(save_faces, file_ok=False)
            for i, img in enumerate(face_images):
                p = os.path.join(save_faces, f'img_{i}__{predictions[i] if i in predictions else "non_classified"}.jpg')
                save_image(img, p)

        return [cls for _, cls in predictions.items()]

    predictions = []
    # group the faces together by iterating through the map
    for image_index, (s, e) in images_to_faces.items():
        embs = face_embeddings[s: e]
        images = faces[s: e]
        p = process_one_image(images, embs, embeddings,
                              save_faces=os.path.join(save_faces, f'image_{image_index}') if save_faces else save_faces,
                              debug=debug)

        predictions.append(p)

    if return_bbox:
        return [[(p, do[1]) for p, do in zip(pred, d_out)] for pred, d_out in zip(predictions, detector_output)]

    return predictions


def recognize_one_image(image: Union[str, Path, np.ndarray, torch.tensor],
                        embeddings: Union[str, Path, dict],
                        possible_classes: Sequence[str] = None,
                        face_detector=None,
                        encoder=None,
                        keep_all=True,
                        confidence_threshold: float = CONFIDENCE_THRESHOLD,
                        report_threshold: float = REPORT_THRESHOLD,
                        return_bbox: bool = False,
                        save_faces: Union[str, Path, None] = None,
                        debug: bool = False,
                        ):
    embeddings = __read_embeddings(embeddings) if isinstance(embeddings, (str, Path)) else embeddings
    # the possible classes
    possible_classes = list(embeddings.keys()) if possible_classes is None else possible_classes

    # make sure the classes used are part of the embeddings keys
    assert set(possible_classes).issubset(set(embeddings.keys())), "THE CLASSES PASSED MUST BE PRESENT IN THE " \
                                                                   "EMBEDDINGS"

    # limit the classes to only those selected and convert the actual embeddings to numpy arrays
    embeddings = dict([(p_c, np.asarray(embeddings[p_c])) for p_c in possible_classes])

    # define SingletonInitializer
    fr_singleton = FR_SingletonInitializer()

    if face_detector is None:
        face_detector = fr_singleton.get_face_detector()
        face_detector.keep_all = keep_all

    encoder = fr_singleton.get_encoder() if encoder is None else encoder

    # convert the image to a type that we can work with:
    if isinstance(image, (str, Path)):
        image = Image.open(str(image))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # otherwise, (if the image is a tensor), it should be left as it is

    # extract the actual face images
    face_images, probs = face_detector.forward(image, return_prob=True)
    # convert probs to the float type
    probs = probs.astype(float)

    # detect faces and bounding boxes
    bounding_boxes = None
    if return_bbox:
        bounding_boxes, b_probs = face_detector.detect(image)
        b_probs = b_probs.astype(float)
        assert np.allclose(np.asarray(probs), np.asarray(b_probs), rtol=10 ** -3, equal_nan=True)

    # filter by probability: only keep instances with high associated probability
    try:
        if bounding_boxes is not None:
            filtered_list = [(face, bb, p)
                             for face, bb, p in zip(face_images, bounding_boxes, probs) if p >= confidence_threshold]
            # extract faces and bounding boxes only
            faces, bounding_boxes, _ = list(map(list, zip(*filtered_list)))

        else:
            filtered_list = [(face, p)
                             for face, p in zip(face_images, probs) if p >= confidence_threshold]
            # detach faces from probabilities...
            faces, _ = list(map(list, zip(*filtered_list)))

    except (TypeError, ValueError):
        # this means no faces were detected
        # return an empty list as the model did not detect any images.
        return []

    # convert the list of tensors to one batched tensor
    final_face_images = torch.stack(faces).to(torch.float32).to(fr_singleton.get_device())
    # produce the embeddings of the detected faces
    face_embeddings = encoder(final_face_images).detach().cpu().numpy()
    if debug:
        __debug_display(final_face_images, face_embeddings, embeddings)

    if save_faces is not None:
        save_faces = process_save_path(save_faces, file_ok=False)
        for i, img in enumerate(faces):
            p = os.path.join(save_faces, f'cropped_image_{i}.jpg')
            save_image(img, p)

    predictions = __predict_one_image(face_embeddings, embeddings, report_threshold=report_threshold)

    if return_bbox:
        return [(p, bb) for p, bb in zip(predictions, bounding_boxes)]

    return predictions


def display_similarity(img1: Union[str, Path, np.array],
                       img2: Union[str, Path, np.array],
                       face_detector=None,
                       encoder=None):
    # get the embeddings as well as the cropped face images
    embeddings, faces = build_embeddings([img1, img2],
                                         face_detector=face_detector,
                                         encoder=encoder,
                                         return_faces=True)

    e1, e2 = embeddings

    # calculate the similarity between the embeddings
    cos_sim = cosine_similarity(e1, e2)

    # the images are tensors. Since Pytorch uses the [C, H, W] convention while Matplotlib uses [H, W, C]
    # permuting the axis is needed.

    # display the first image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(faces[0].detach().permute(1, 2, 0).cpu().numpy())
    ax[0].set_title(f"first image")
    ax[0].axis("off")

    # display the 2nd image
    ax[1].imshow(faces[1].detach().permute(1, 2, 0).cpu().numpy())
    ax[1].set_title(f"second image")
    ax[1].axis("off")

    fig.suptitle(f"Cosine Distance {round(cos_sim, 4)}", fontsize=16)
    plt.show()
