import os
import os.path as osp
import datetime

# --------------- Constants ----------------

# Directory constants
this_dir = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.join(this_dir, '../')
KDE_MODEL_PATH = osp.join(ROOT_DIR, 'model')

# Log directory
now = datetime.datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
LOG_DIR = osp.join(ROOT_DIR, "logs/{}".format(date_time))
print(LOG_DIR)

PAPER_FIGURE_ID=1
MRT_FIGURE_ID=2
DISPLAY_FIGURE_ID=3

POSITIVE_ANS = {"yes", "yeah", "yep", "sure", "certainly", "OK"}
NEGATIVE_ANS = {"no", "nope", "nah"}
PRONOUNS = {"it's", "it is", "one"}

# Action definiiton
Q2={
    "type1": "I have not found the target. Where is it?", # COMMON FORMAT
    "type2": "I have not found the target. Where is it?",         # WHEN ALL THINGS WITH PROB 0
    "type3": "Do you mean the {:s}? If not, where is the target?"  # WHEN ONLY ONE THING WITH POSITIVE PROB
}

Q1={
    "type1": "Do you mean the {:s}?"
}

GRASP_AND_END = 0
GRASP_AND_CONTINUE = 1
ASK_Q1 = 2

CLASSES = ['__background__',  # always index 0
               'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
               'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote controller', 'can', 'tape', 'knife', 'wrench', 'cup', 'charger',
               'badminton', 'wallet', 'wrist developer', 'glasses', 'plier', 'headset',
               'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

CLASSES_2 = ['__background__',  # always index 0
               'box', 'banana', 'notebook', 'apple', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote controller', 'can', 'knife', 'cup', 'wallet', 'glasses', 'toothbrush']

COCO_CLASSES = [u'__background__', u'person', u'bicycle', u'car',
u'motorcycle',
u'airplane',
u'bus',
u'train',
u'truck',
u'boat',
u'traffic light',
u'fire hydrant',
u'stop sign',
u'parking meter',
u'bench',
u'bird',
u'cat',
u'dog',
u'horse',
u'sheep',
u'cow',
u'elephant',
u'bear',
u'zebra',
u'giraffe',
u'backpack',
u'umbrella',
u'handbag',
u'tie',
u'suitcase',
u'frisbee',
u'skis',
u'snowboard',
u'sports ball',
u'kite',
u'baseball bat',
u'baseball glove',
u'skateboard',
u'surfboard',
u'tennis racket',
u'bottle',
u'wine glass',
u'cup',
u'fork',
u'knife',
u'spoon',
u'bowl',
u'banana',
u'apple',
u'sandwich',
u'orange',
u'broccoli',
u'carrot',
u'hot dog',
u'pizza',
u'donut',
u'cake',
u'chair',
u'couch',
u'potted plant',
u'bed',
u'dining table',
u'toilet',
u'tv',
u'laptop',
u'mouse',
u'remote',
u'keyboard',
u'cell phone',
u'microwave',
u'oven',
u'toaster',
u'sink',
u'refrigerator',
u'book',
u'clock',
u'vase',
u'scissors',
u'teddy bear',
u'hair drier',
u'toothbrush']

CLASSES_TO_IND = dict(zip(CLASSES, range(len(CLASSES))))

TEST = 0
EXPERIMENT = 1

# --------------- Settings ------------------
MODE = TEST # 0 for test, 1 for experiment
# EXP_SETTING = "baseline" # choose from: baseline, no_uncert, no_multistep, invigorate
# EXP_SETTING = "greedy" # choose from: baseline, no_uncert, no_multistep, invigorate
# EXP_SETTING = "heuristic" # choose from: baseline, no_uncert, no_multistep, invigorate
EXP_SETTING = "invigorate" # choose from: baseline, no_uncert, no_multistep, invigorate

# ------------- EXP Settings --------------
PARTICIPANT_NUM = 1
SCENE_NUM = 6
VER_NUM = 0
if EXP_SETTING == "baseline":
    VER_NUM = 1
elif EXP_SETTING == "greedy":
    VER_NUM = 2
elif EXP_SETTING == "heuristic":
    VER_NUM = 3
elif EXP_SETTING == "invigorate":
    VER_NUM = 4

EXP_DIR = osp.join(ROOT_DIR, "experiment")
EXP_RES_DIR = osp.join(EXP_DIR, "participant {}".format(PARTICIPANT_NUM), "{}".format(SCENE_NUM), "{}".format(VER_NUM))

# ------------ Further settings -------------
# create directory is necessary
if MODE == EXPERIMENT:
    LOG_DIR = EXP_RES_DIR # override log directory
if not osp.exists(LOG_DIR):
    os.makedirs(LOG_DIR)