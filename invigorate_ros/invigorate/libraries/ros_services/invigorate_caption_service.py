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
import rospy

class INGRESSService():
    def __init__(self):
        self.ingress =Ingress()

class INVIGORATEService():
    def __init__(self):
        self.invigorate = InvigorateIJRRV6()

        rospy.init_node('INVIGORATE_caption_test', anonymous=True)

    def generate_captions(self, img, bboxes):
        bboxes, classes, scores = self.invigorate.object_detection(img, bboxes)
        class_names = [CLASSES[int(i.item())] for i in classes]
        num_obj = len(class_names)

        for i in range(num_obj):
            name = class_names[i]
            subject_tokens = name.split(' ')
            self.invigorate.subject = subject_tokens # set subject
            self.invigorate.question_captions_generation(img, bboxes, classes)
            cand_captions = self.invigorate.belief['questions']
            self.invigorate.match_question_to_object(img, bboxes, classes, cand_captions)
            match_scores = self.invigorate.belief['q_matching_scores']
            match_probs = self.invigorate.belief['q_matching_prob']
            print(cand_captions)
            print(match_scores, match_probs)


