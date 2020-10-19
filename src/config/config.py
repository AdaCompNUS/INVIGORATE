import os.path as osp

# --------------- Settings ------------------
# EXP_SETTING = "baseline" # choose from: baseline, no_uncert, no_multistep, invigorate
# EXP_SETTING = "no_uncert" # choose from: baseline, no_uncert, no_multistep, invigorate
# EXP_SETTING = "no_multistep" # choose from: baseline, no_uncert, no_multistep, invigorate
EXP_SETTING = "invigorate" # choose from: baseline, no_uncert, no_multistep, invigorate
# EXP_SETTING = "no_multistep_2" # choose from: baseline, no_uncert, no_multistep, invigorate

# --------------- Constants ----------------
this_dir = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.join(this_dir, '../')

Q2={
    "type1": "I have not found the target. Where is it?", # COMMON FORMAT
    "type2": "I have not found the target. Where is it?",         # WHEN ALL THINGS WITH PROB 0
    "type3": "Do you mean the {:s}? If not, where is the target?"  # WHEN ONLY ONE THING WITH POSITIVE PROB
}

Q1={
    "type1": "Do you mean the {:s}?"
}

CLASSES = ['__background__',  # always index 0
               'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
               'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote controller', 'can', 'tape', 'knife', 'wrench', 'cup', 'charger',
               'badminton', 'wallet', 'wrist developer', 'glasses', 'plier', 'headset',
               'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

CLASSES_TO_IND = dict(zip(CLASSES, range(len(CLASSES))))