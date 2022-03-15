# this file includes two caption wrappers from invigorate
#   1. INGRESS Service
#   3. INVIGORATE Service
# Originally, this file is used to test the captioning performance of
# INVIGORATE. One can also use these services for any purpose.
import abc
from abc import ABCMeta, abstractmethod
from ingress_srv.ingress_srv import Ingress
from invigorate.invigorate_models.invigorate_ijrr_v6 import InvigorateIJRRV6
from invigorate.config.config import CLASSES

class INGRESSService():
    def __init__(self):
        self.ingress =Ingress()

class INVIGORATEService():
    def __init__(self):
        self.invigorate = InvigorateIJRRV6()

    def generate_captions(self, img, bboxes):
        bboxes, classes, scores = self.invigorate.object_detection(img, bboxes)
        class_names = [CLASSES[int(i.item())] for i in classes]
        num_obj = len(class_names)

        for i in range(num_obj):
            name = class_names[i]
            subject_tokens = name.split(' ')
            cand_captions = self.invigorate.question_captions_generation(
                img, bboxes, classes, subject=subject_tokens)


