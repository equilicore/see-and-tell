"""This file contains all the constants used across different scripts.
It was created as a solution for cyclic imports.
"""
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
RESIZE = 160
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FACE_DETECTOR = MTCNN(image_size=RESIZE,  # the default value
                      keep_all=True,  # we want all the faces in any image
                      post_process=True,  # we want all the returned images to be the same size
                      select_largest=False,  # return the face with the highest probability, not the largest
                      device=DEVICE)
# global variable used as the default encoder
ENCODER = InceptionResnetV1(pretrained='vggface2',
                            device=DEVICE).eval()

CONFIDENCE_THRESHOLD = 0.9
REPORT_THRESHOLD = 0.25
