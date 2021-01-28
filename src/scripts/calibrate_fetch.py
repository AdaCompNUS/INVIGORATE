import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../'))

import tf
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf import transformations as T
import math
import tf2_ros

from rls_perception_msgs.srv import *

from config.config import *
from libraries.robots.fetch_robot import FetchRobot, GRASP_DEPTH, FETCH_GRIPPER_LENGTH
from invigorate.invigorate import Invigorate
from libraries.data_viewer.data_viewer import DataViewer

# ----------- Constants ----------
BOX_HEIGHT = 0.05

# ------------ Code ------------

rospy.init_node("calibrate_fetch")

# get ground truth
br = tf2_ros.StaticTransformBroadcaster()
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
data_viewer = DataViewer(CLASSES)
expr = "box"
bbox_segmentation_client = rospy.ServiceProxy('rls_perception_services/bbox_pc_segmention_service', BBoxSegmentation)

img, _ = robot.read_imgs()
observations = invigorate_client.perceive_img(img, expr)
if observations is None:
    print("nothing is detected, abort!!!")
    sys.exit(0)
invigorate_client.estimate_state_with_observation(observations)

img = observations['img']
bboxes = observations['bboxes']
classes = observations['classes']
rel_mat = observations['rel_mat']
rel_score_mat = observations['rel_score_mat']
target_prob = invigorate_client.belief['target_prob']
imgs = data_viewer.generate_visualization_imgs(img, bboxes, classes, rel_mat, rel_score_mat, expr, target_prob, save=False)
data_viewer.display_img(imgs['final_img'])

action = invigorate_client.decision_making_heuristic() # action_idx.
action_type, _ = invigorate_client.get_action_type(action)

# grasps = observations['grasps']
# num_box = observations['bboxes'].shape[0]
# grasp_box = grasps[action % num_box][:8]
# grasp, _ = robot._cal_initial_grasp(grasp_box)

seg_req = BBoxSegmentationRequest()
seg_req.x = bboxes[0][0] + 700
seg_req.y = bboxes[0][1] + 470
seg_req.width = bboxes[0][2] - bboxes[0][0]
seg_req.height = bboxes[0][3] - bboxes[0][1]
seg_req.angle = 0
seg_req.transform_to_reference_frame = True
seg_req.reference_frame = 'base_link'
seg_resp = bbox_segmentation_client(seg_req)
obj_pose = seg_resp.object.primitive_pose

diff_x = trans[0] - obj_pose.position.x
diff_y = trans[1] - obj_pose.position.y
# diff_z = trans[2] + BOX_HEIGHT - GRASP_DEPTH - obj_pose.pose.position.z
diff_z = trans[2] + BOX_HEIGHT / 2 - obj_pose.position.z

print("Calibration Result: diff_x, y, z : {} {} {}".format(diff_x, diff_y, diff_z))

target_pose = PoseStamped()
target_pose.header.frame_id="base_link"
target_pose.pose.position.x = trans[0]
target_pose.pose.position.y = trans[1]
target_pose.pose.position.z = trans[2] + 0.1 + FETCH_GRIPPER_LENGTH
quat = T.quaternion_from_euler(0, math.pi / 2, 0, 'rzyx') # rotate by y to make it facing downwards
                                                          # rotate by z to align with bbox orientation
target_pose.pose.orientation.x = quat[0]
target_pose.pose.orientation.y = quat[1]
target_pose.pose.orientation.z = quat[2]
target_pose.pose.orientation.w = quat[3]

print(target_pose)

grasp_t = TransformStamped()
grasp_t.header.stamp = rospy.Time.now()
grasp_t.header.frame_id = 'base_link'
grasp_t.child_frame_id = 'grasp_pose'
grasp_t.transform.translation = target_pose.pose.position
grasp_t.transform.rotation = target_pose.pose.orientation
br.sendTransform(grasp_t)  # broadcast transform so can see in rviz
raw_input("press anything to continue")

robot._move_arm_to_pose(target_pose)

raw_input("check if arm is on top of apriltag, press anything to continue")

robot.move_arm_to_home()