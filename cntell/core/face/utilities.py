"""This file contains all the constants used across different scripts.
It was created as a solution for cyclic imports.
"""
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

RESIZE = 160
CONFIDENCE_THRESHOLD = 0.9
REPORT_THRESHOLD = 0.25


# we will the singleton pattern to avoid initializing FACE_DETECTOR and ENCODER object everytime

class FR_SingletonInitializer(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FR_SingletonInitializer, cls).__new__(cls)
            # set the attributes of cls
            cls.instance.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cls.instance.__encoder = InceptionResnetV1(pretrained='vggface2', device=cls.instance.__device).eval()
            cls.instance.__face_detector = MTCNN(image_size=RESIZE,  # the default value
                                                 keep_all=True,  # we want all the faces in any image
                                                 post_process=True,
                                                 # we want all the returned images to be the same size
                                                 select_largest=False,
                                                 # return the face with the highest probability, not the largest
                                                 device=cls.instance.__device)

        return cls.instance

    def get_device(self):
        return self.__device

    def get_encoder(self):
        return self.__encoder

    def get_face_detector(self):
        return self.__face_detector
