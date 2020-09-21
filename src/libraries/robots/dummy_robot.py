
import cv2
import os.path as osp

from config.config import ROOT_DIR

class DummyRobot():
    def __init__(self):
        pass

    def read_imgs(self):
        img_name = raw_input('Enter img name: ')
        img = cv2.imread(osp.join(ROOT_DIR, 'images/' + img_name))
        depth = None
        return img, depth

    def grasp(self, grasp):
        print('Dummy execution of grasp {}'.format(grasp))
        return True

    def say(self, text):
        print('Dummy execution of say: {}'.format(text))
        return True

    def listen(self, timeout=None):
        print('Dummy execution of listen')
        text = raw_input("Enter: ")
        return text
