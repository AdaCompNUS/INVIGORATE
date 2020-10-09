import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../'))

import tf
import rospy

from libraries.robots.fetch_robot import FetchRobot, GRASP_DEPTH
from invigorate.invigorate import Invigorate

# ----------- Constants ----------
BOX_HEIGHT = 0.05

# ------------ Code ------------

rospy.init_node("calibrate_fetch")

# get ground truth
listener = tf.TransformListener()
rospy.sleep(5)
try:
    (trans, rot) = listener.lookupTransform('/base_link', '/my_tag_static', rospy.Time())
except Exception as e:
    print(e)
    sys.exit(0)

print("ground_truth: {}".format(trans))
to_continue = raw_input("Place the box, press y to continue")
if to_continue != 'y':
    sys.exit(0)

# use invigorate to detect box, the box location should be exactly at apriltag location
robot = FetchRobot()
invigorate_client = Invigorate()
expr = "box"
img, _ = robot.read_imgs()
observations = invigorate_client.perceive_img(img, expr)
if observations is None:
    print("nothing is detected, abort!!!")
    sys.exit(0)
invigorate_client.estimate_state_with_observation(observations)
action = invigorate_client.decision_making_heuristic() # action_idx.
action_type = invigorate_client.get_action_type(action)

grasps = observations['grasps']
num_box = observations['bboxes'].shape[0]
grasp_box = grasps[action % num_box][:8]
grasp, _ = robot._cal_initial_grasp(grasp_box)

diff_x = trans[0] - grasp.grasp_pose.pose.position.x
diff_y = trans[1] - grasp.grasp_pose.pose.position.y
diff_z = trans[2] + BOX_HEIGHT - GRASP_DEPTH - grasp.grasp_pose.pose.position.z + 0.01 # This is necessary

print("diff_x, y, z : {} {} {}".format(diff_x, diff_y, diff_z))