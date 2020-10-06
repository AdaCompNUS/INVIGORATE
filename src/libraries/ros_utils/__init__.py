import warnings

try:
    from rosapi.kinect_subscriber import *
    from rosapi.calibrate import *
except:
    warnings.warn("Kinect and Baxter cannot work.")

from rosapi.getco import *
from rosapi.get_grasp_ori import GetGraspOri