import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os.path as osp
import tf
from tf import transformations as t
import math
import numpy as np

from rls_perception_msgs.srv import *
from rls_control_msgs.srv import *
from geometry_msgs.msg import *
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import Grasp

from config.config import CLASSES
from libraries.data_viewer.data_viewer import DataViewer


# ------- Settings ---------
GRASP_BOX_FOR_SEG = 1
BBOX_FOR_SEG = 2
GRASP_BOX_6DOF_PICK = 3
USE_REALSENSE = True

# ------- Constants ---------
# ORIG_IMAGE_SIZE = (480, 640)
# SCALE = 0.8
# Y_OFFSET = int(ORIG_IMAGE_SIZE[0] * (1 - SCALE) / 2)
# X_OFFSET = int(ORIG_IMAGE_SIZE[1] * (1 - SCALE) / 2)
# YCROP = (Y_OFFSET, ORIG_IMAGE_SIZE[0] - Y_OFFSET)
# XCROP = (X_OFFSET, ORIG_IMAGE_SIZE[1] - X_OFFSET)
if USE_REALSENSE:
    YCROP = (180, 450)
    XCROP = (200, 500)
else:
    YCROP = (180, 450)
    XCROP = (150, 490)
FETCH_GRIPPER_LENGTH = 0.2
GRASP_DEPTH = 0.04
GRASP_POSE_X_OFFST = 0
GRIPPER_OPENING_OFFSET = 0.01

class FetchRobot():
    def __init__(self):
        self._br = CvBridge()
        self._bbox_segmentation_client = rospy.ServiceProxy('rls_perception_services/bbox_pc_segmention_service', BBoxSegmentation)
        self._pnp_client = rospy.ServiceProxy('rls_control_services/fetch/pnp', PickPlace)
        # rospy.wait_for_service('/segment_table')
        self._table_segmentor_client = rospy.ServiceProxy('/segment_table', TableSegmentation)
        self._tf_transformer = tf.TransformerROS()
        self._fetch_image_client = rospy.ServiceProxy('/rls_perception_service/fetch/rgb_image_service', RetrieveImage)

        # call pnp service to get ready
        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.GET_READY
        resp = self._pnp_client(pnp_req)  # get existing result
        if not resp.success:
            raise RuntimeError('fetch failed to get ready for pick n place!!!')

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.MOVE_ARM_TO_HOME
        resp = self._pnp_client(pnp_req)  # get existing result
        if not resp.success:
            raise RuntimeError('fetch failed to move arm to home!!!')

    def read_imgs(self):
        # resp = self._fetch_image_client()
        # img = self._br.imgmsg_to_cv2(resp.image, desired_encoding='bgr8')
        if USE_REALSENSE:
            img_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
            img = self._br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            print('img_size : {}'.format(img.shape))
            img = img[YCROP[0]:YCROP[1], XCROP[0]:XCROP[1]]
            print('img_size : {}'.format(img.shape))
        else:
            resp = self._fetch_image_client()
            img = self._br.imgmsg_to_cv2(resp.image, desired_encoding='bgr8')
            print('img_size : {}'.format(img.shape))
            img = img[YCROP[0]:YCROP[1], XCROP[0]:XCROP[1]]
            print('img_size : {}'.format(img.shape))
        # depth_img_msg = rospy.wait_for_message('/head_camera/depth/image_rect', Image)
        # depth = self._br.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        depth = None
        return img, depth

    def grasp(self, grasp):
        # return self._top_grasp(bbox, grasp)
        return self._top_grasp(grasp)

    def say(self, text):
        print('Dummy execution of say: {}'.format(text))

    def listen(self, timeout=None):
        print('Dummy execution of listen')
        text = raw_input('Enter: ')
        return text

    def _top_grasp(self, grasp):
        print('grasp_box: {}'.format(grasp))
        grasp = grasp + np.tile([XCROP[0], YCROP[0]], 4)
        x1, y1, x2, y2, x3, y3, x4, y4 = grasp.tolist()
        seg_req = BBoxSegmentationRequest()
        seg_req.x = x1
        seg_req.y = y1
        seg_req.width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        seg_req.height = math.sqrt((x4 - x1)**2 + (y4 - y1)**2)
        seg_req.angle = math.atan2(y2 - y1, x2 - x1)
        seg_req.transform_to_reference_frame = True
        seg_req.reference_frame = 'base_link'

        # resp = self._table_segmentor_client(1)  # get existing result
        # seg_req.min_z = resp.marker.pose.position.z + resp.marker.scale.z / 2 + 0.003

        print('calling bbox segmentation service')
        seg_resp = self._bbox_segmentation_client(seg_req)
        print(seg_resp.object)

        to_continue = raw_input('to_continue?')
        if to_continue != 'y':
            return False

        obj_pose = seg_resp.object.primitive_pose
        obj_width = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Y]
        obj_height = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Z]
        approach_dist = 0.1

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.EXECUTE_GRASP

        grasp = Grasp()
        grasp.grasp_pose.header = seg_resp.object.header
        grasp.grasp_pose.pose = seg_resp.object.primitive_pose
        grasp.grasp_pose.pose.position.x += GRASP_POSE_X_OFFST # HACK!!!
        grasp.grasp_pose.pose.position.z += obj_height / 2 - GRASP_DEPTH + approach_dist + FETCH_GRIPPER_LENGTH
        quat = t.quaternion_from_euler(0, math.pi / 2, seg_req.angle, 'rzyx') # rotate by y to make it facing downwards
                                                                              # rotate by z to align with bbox orientation
        grasp.grasp_pose.pose.orientation.x = quat[0]
        grasp.grasp_pose.pose.orientation.y = quat[1]
        grasp.grasp_pose.pose.orientation.z = quat[2]
        grasp.grasp_pose.pose.orientation.w = quat[3]

        grasp.pre_grasp_approach.direction.header = seg_resp.object.header
        grasp.pre_grasp_approach.direction.vector.z = -1 # top pick
        grasp.pre_grasp_approach.desired_distance = approach_dist

        grasp.post_grasp_retreat.direction.header = seg_resp.object.header
        grasp.post_grasp_retreat.direction.vector.z = 1 # top pick
        grasp.post_grasp_retreat.desired_distance = approach_dist

        pnp_req.grasp = grasp
        pnp_req.gripper_opening = obj_width + GRIPPER_OPENING_OFFSET

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            print('ERROR: robot grasp failed!!')
        return resp.success







"""
Legacy
    def _top_grasp_2(self, bbox, grasp):
        # use bbox for segmentation
        print('grasp_box: {}'.format(grasp))
        print('bbox : {}'.format(bbox))
        grasp = grasp + np.tile([XCROP[0], YCROP[0]], 4)
        bbox = bbox + np.tile([XCROP[0], YCROP[0]], 2)
        x1, y1, x2, y2, _, _, _, _ = grasp.tolist()

        seg_req = BBoxSegmentationRequest()
        seg_req.x = bbox[0]
        seg_req.y = bbox[1]
        seg_req.width = bbox[2] - bbox[0]
        seg_req.height = bbox[3] - bbox[1]
        seg_req.transform_to_reference_frame = True
        seg_req.reference_frame = 'base_link'

        resp = self._table_segmentor_client(1)  # get existing result
        seg_req.min_z = resp.marker.pose.position.z + resp.marker.scale.z / 2 + 0.003

        print('calling bbox segmentation service')
        seg_resp = self._bbox_segmentation_client(seg_req)
        print(seg_resp.object)

        to_continue = raw_input('to_continue?')
        if to_continue != 'y':
            return False

        obj_pose = seg_resp.object.primitive_pose
        obj_width = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Y]
        obj_height = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Z]
        approach_dist = 0.1

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.EXECUTE_GRASP

        grasp = Grasp()
        grasp.grasp_pose.header = seg_resp.object.header
        grasp.grasp_pose.pose = seg_resp.object.primitive_pose
        grasp.grasp_pose.pose.position.z += obj_height / 2 - GRASP_DEPTH + approach_dist + FETCH_GRIPPER_LENGTH

        angle_z = math.atan2(y2 - y1, x2 - x1)
        quat = t.quaternion_from_euler(0, math.pi / 2, angle_z, 'rzyx') # rotate by y to make it facing downwards
                                                                        # rotate by z to align with bbox orientation
        grasp.grasp_pose.pose.orientation.x = quat[0]
        grasp.grasp_pose.pose.orientation.y = quat[1]
        grasp.grasp_pose.pose.orientation.z = quat[2]
        grasp.grasp_pose.pose.orientation.w = quat[3]

        grasp.pre_grasp_approach.direction.header = seg_resp.object.header
        grasp.pre_grasp_approach.direction.vector.z = -1 # top pick
        grasp.pre_grasp_approach.desired_distance = approach_dist

        grasp.post_grasp_retreat.direction.header = seg_resp.object.header
        grasp.post_grasp_retreat.direction.vector.z = 1 # top pick
        grasp.post_grasp_retreat.desired_distance = approach_dist

        pnp_req.grasp = grasp
        pnp_req.gripper_opening = obj_width

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            print('ERROR: robot grasp failed!!')
        return resp.success

    def _6dof_grasp(self, grasp):
        # print('Dummy execution of grasp {}'.format(grasp))
        # return
        print('grasp_box: {}'.format(grasp))
        grasp = grasp + np.tile([XCROP[0], YCROP[0]], 4)

        x1, y1, x2, y2, x3, y3, x4, y4 = grasp.tolist()
        seg_req = BBoxSegmentationRequest()
        seg_req.x = x1
        seg_req.y = y1
        seg_req.width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        seg_req.height = math.sqrt((x4 - x1)**2 + (y4 - y1)**2)
        seg_req.angle = math.atan2(y2 - y1, x2 - x1)
        # seg_req.transform_to_reference_frame = True
        # seg_req.reference_frame = 'base_link'

        # resp = self._table_segmentor_client(1)  # get existing result
        # seg_req.min_z = resp.marker.pose.position.z + resp.marker.scale.z / 2

        print('calling bbox segmentation service')
        seg_resp = self._bbox_segmentation_client(seg_req)
        print(seg_resp.object)

        to_continue = raw_input('to_continue?')
        if to_continue != 'y':
            return False

        obj_pose = seg_resp.object.primitive_pose
        obj_width = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Y]
        obj_height = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Z]
        approach_dist = 0.1

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.EXECUTE_GRASP

        grasp_pose = PoseStamped()
        grasp_pose.header = seg_resp.object.header
        grasp_pose.pose = seg_resp.object.primitive_pose
        grasp_pose.pose.position.z -= obj_height / 2 - 0.01 + approach_dist + FETCH_GRIPPER_LENGTH
        quat = t.quaternion_from_euler(0, -math.pi / 2, seg_req.angle, 'rzyx') # rotate by y to make it facing downwards
                                                                               # rotate by z to align with bbox orientation
        grasp_pose.pose.orientation.x = quat[0]
        grasp_pose.pose.orientation.y = quat[1]
        grasp_pose.pose.orientation.z = quat[2]
        grasp_pose.pose.orientation.w = quat[3]

        pre_vector = Vector3Stamped()
        pre_vector.header = seg_resp.object.header
        pre_vector.vector.z = 1 # top pick

        post_vector = Vector3Stamped()
        post_vector.header = seg_resp.object.header
        post_vector.vector.z = -1 # top pick

        grasp_pose = self._tf_transformer.transformPose('base_link', grasp_pose)
        pre_vector = self._tf_transformer.transformVector3('base_link', pre_vector)
        pre_vector = self._tf_transformer.transformVector3('base_link', post_vector)

        pnp_req.grasp.grasp_pose = grasp_pose
        pnp_req.grasp.pre_grasp_approach.direction = pre_vector
        pnp_req.grasp.pre_grasp_approach.desired_distance = approach_dist
        pnp_req.grasp.post_grasp_retreat.direction = post_vector
        pnp_req.grasp.post_grasp_retreat.desired_distance = approach_dist
        pnp_req.gripper_opening = obj_width

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            print('ERROR: robot grasp failed!!')
        return resp.success
"""