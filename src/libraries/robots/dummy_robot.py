
import cv2
import os.path as osp

class DummyRobot():
    def __init__(self):
        pass

    def read_imgs(self):
        img_name = raw_input('Enter img name: ')
        cur_dir = osp.dirname(osp.abspath(__file__))
        img = cv2.imread(osp.join(cur_dir, '../images/' + img_name))
        depth = None
        return img, depth

    def grasp(self, grasp):
        print('Dummy execution of grasp {}'.format(grasp))
    
    def say(self, text):
        print('Dummy execution of say: {}'.format(text))

    def listen(self, timeout=None):
        print('Dummy execution of listen')
        text = raw_input("Enter: ")
        return text
