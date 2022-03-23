import os
import os.path as osp
import datetime
import sys

PYTHON_VERSION = sys.version.split(".")[0]
if PYTHON_VERSION == "3":
    # overwrite the built-in methods of python 2.x
    raw_input = input
    xrange = range
# --------------- Constants ----------------

# Directory constants
this_dir = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.join(this_dir, '../')
KDE_MODEL_PATH = osp.join(ROOT_DIR, 'model')
NN_MODEL_PATH = osp.join(ROOT_DIR, 'model')

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
PRONOUNS = {"it", "it's", "one", "that", "that's"}
NLP_SERVER = "nltk"

# Action definiiton
Q2={
    "type1": "I have not found the target. Where is it?", # COMMON FORMAT
    "type2": "I have not found the target. Where is it?",         # WHEN ALL THINGS WITH PROB 0
    "type3": "Do you mean the {:s}? If not, where is the target?"  # WHEN ONLY ONE THING WITH POSITIVE PROB
}

Q1={
    "type1": "Do you mean {:s}?"
}

# planned-action type macros
GRASP_AND_END = 0
GRASP_AND_CONTINUE = 1
ASK_Q1 = 2

# executed-action type macros
EXEC_GRASP = 0
EXEC_ASK_WITH_POINTING = 1                # with pointing action
EXEC_DUMMY_ASK = 2
EXEC_ASK_WITHOUT_POINT=3    # without pointing action

# CLASSES = ['__background__',  # always index 0
#                'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
#                'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
#                'remote controller', 'can', 'tape', 'knife', 'wrench', 'cup', 'charger',
#                'badminton', 'wallet', 'wrist developer', 'glasses', 'plier', 'headset',
#                'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

CLASSES = [
'__background__',
'ball' ,
'bottle',
'cup',
'knife',
'banana',
'apple',
'carrot',
'mouse' ,
'remote' ,
'cell phone' ,
'book' ,
'scissors' ,
'teddy bear' ,
'toothbrush' ,
'box' ,
]

CLASSES_TO_IND = dict(zip(CLASSES, range(len(CLASSES))))

TEST = 1
EXPERIMENT = 0

# --------------- Settings ------------------
MODE = EXPERIMENT # 1 for test, 0 for experiment
# EXP_SETTING = "greedy" # choose from: baseline, no_uncert, no_multistep, invigorate
# EXP_SETTING = "heuristic" # choose from: baseline, no_uncert, no_multistep, invigorate
# EXP_SETTING = "no_interaction"
# EXP_SETTING = "no_multistep"
# EXP_SETTING = "no_multistep_all"
# EXP_SETTING = "invigorate_ijrr"
# EXP_SETTING = "invigorate_ijrr_v2"
# EXP_SETTING = "invigorate_ijrr_v3"
# EXP_SETTING = "invigorate_ijrr_v4"
# EXP_SETTING = "invigorate_ijrr_v5"
# EXP_SETTING = "invigorate_ijrr_v6"
EXP_SETTING = "invigorate"
# EXP_SETTING = "invigorate_vs_vilbert"

# ------------- EXP Settings --------------
PARTICIPANT_NUM = 9
SCENE_NUM = 3
VER_NUM = 0
EPSILON = 0.01
if EXP_SETTING == "baseline":
    VER_NUM = 1
elif EXP_SETTING == "greedy":
    VER_NUM = 2
elif EXP_SETTING == "heuristic":
    VER_NUM = 3
elif EXP_SETTING == "invigorate":
    VER_NUM = 4
elif EXP_SETTING == "no_interaction":
    VER_NUM = 5
elif EXP_SETTING == "no_multistep":
    VER_NUM = 6
elif EXP_SETTING == "no_multistep_all":
    VER_NUM = 7
elif EXP_SETTING == "invigorate_ijrr_v6":
    VER_NUM = 8

EXP_DIR = osp.join(ROOT_DIR, "experiment/dataset")
EXP_DATA_DIR = osp.join(ROOT_DIR, "experiment/dataset/{}".format((PARTICIPANT_NUM-1) * 10 +  SCENE_NUM))
EXP_RES_DIR = osp.join(ROOT_DIR, "experiment/result/{}".format((PARTICIPANT_NUM -1)* 10 + SCENE_NUM), "{}".format(VER_NUM))

if EXP_SETTING == "invigorate_vs_vilbert":
    EXP_DIR = osp.join(ROOT_DIR, "experiment/invigorate_vs_vilbert")
    EXP_RES_DIR = osp.join(EXP_DIR, "{}{}".format(PARTICIPANT_NUM, SCENE_NUM))

# ------------ Further settings -------------
# create directory is necessary
if MODE == EXPERIMENT:
    LOG_DIR = EXP_RES_DIR # override log directory
if not osp.exists(LOG_DIR):
    os.makedirs(LOG_DIR)