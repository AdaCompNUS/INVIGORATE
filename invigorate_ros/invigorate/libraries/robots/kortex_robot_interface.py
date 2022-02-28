import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os.path as osp
import tf
from tf import transformations as T
import math
import numpy as np
import random
import os
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String
from geometry_msgs.msg import *
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import Grasp
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
import time
import random
import logging

from rls_perception_msgs.srv import *
from rls_control_msgs.srv import *
from kortex_robot.kortex_robot import KortexRobot

# from invigorate.libraries.grasp_collision_checker.grasp_collision_checker import GraspCollisionChecker
from invigorate.libraries.utils.log import LOGGER_NAME


# ------- Settings ---------
GRASP_BOX_FOR_SEG = 1
BBOX_FOR_SEG = 2
GRASP_BOX_6DOF_PICK = 3
DUMMY_LISTEN = True
DUMMY_SAY = False
DUMMY_GRASP = False

# ------- Constants ---------
ORIG_IMAGE_SIZE = (540, 960)
YCROP = (100, 440) # 540
XCROP = (300, 660) # 960

KORTEX_GRIPPER_LENGTH = 0.01
GRASP_DEPTH = 0.008
TABLE_MIN_Z = -0.1 # table is slightly lower than base_link
INITIAL_GRASP_Z_OFFSET = 0.0
GRASP_POSE_X_OFFST = 0
GRASP_POSE_Y_OFFST = 0.0
GRASP_POSE_Z_OFFST = 0.015
GRASP_WIDTH_OFFSET = 0.0
GRIPPER_OPENING_OFFSET = 0.01
GRIPPER_OPENING_MAX = 0.09
PLACE_BBOX_SIZE = 80
APPROACH_DIST = 0.1
RETREAT_DIST = 0.15
GRASP_BOX_TO_GRIPPER_OPENING = 0.0011
KORTEX_MAX_GRIPPER_OPENING = 0.14
GRIPPER_WIDTH_TO_HEIGHT_PROP = (0.025 / 0.14)

# ---------- Statics ------------
logger = logging.getLogger(LOGGER_NAME)

class KortexRobotInvigorate():
    def __init__(self):

        self._bbox_segmentation_client = rospy.ServiceProxy('rls_perception_services/kinect/bbox_pc_segmention_service', BBoxSegmentation)
        self._fetch_speaker_client = rospy.ServiceProxy("rls_control_services/fetch/speaker_google", SpeakGoogle)
        self._text_to_speech_client = rospy.ServiceProxy("rls_control_services/text_to_speech", TextToSpeech)

        self._kortex_robot = KortexRobot()
        self._kortex_robot.set_arm_velocity_scaling(0.6)

        self._br = CvBridge()
        self._tf_transformer = tf.TransformerROS()
        self._tl = tf.TransformListener()
        self.place_cnt = 0

        self._kortex_robot.arm_move_to_home()

    def clear(self):
        self.place_cnt = 0

    def _cal_initial_grasp(self, grasp_box):
        logger.debug('grasp_box: {}'.format(grasp_box))
        grasp_box = grasp_box + np.tile([XCROP[0], YCROP[0]], 4)
        # for grasp box, x1,y1 is topleft, x2,y2 is topright, x3,y3 is btmright, x4,y4 is btmleft
        x1, y1, x2, y2, x3, y3, x4, y4 = grasp_box.tolist()
        angle = math.atan2(y2 - y1, x2 - x1)
        seg_req = BBoxSegmentationRequest()
        seg_req.x = x1
        seg_req.y = y1
        seg_req.width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        seg_req.height = math.sqrt((x4 - x1)**2 + (y4 - y1)**2)
        seg_req.angle = angle
        seg_req.min_z = TABLE_MIN_Z
        seg_req.transform_to_reference_frame = True
        seg_req.reference_frame = 'base_link'

        grasp_box_width = seg_req.width
        logger.debug("grasp box width: {}".format(grasp_box_width))

        # resp = self._table_segmentor_client(1)  # get existing result
        # seg_req.min_z = resp.marker.pose.position.z + resp.marker.scale.z / 2 + 0.003

        # print('calling bbox segmentation service')
        seg_resp = self._bbox_segmentation_client(seg_req)
        # print(seg_resp.object)

        obj_pose = seg_resp.object.primitive_pose
        obj_length = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_X]
        obj_width = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Y]
        obj_height = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Z]
        print("obj_x, y, z: {} {} {}".format(obj_length, obj_width, obj_height))

        # seg request again to get grasp box center pos
        grasp_box_cx = x4 + (x2 - x4) / 2
        grasp_box_cy = y2 + (y4 - y2) / 2
        seg_req.x = grasp_box_cx - 2
        seg_req.y = grasp_box_cy - 2
        seg_req.width = 5
        seg_req.height = 5
        seg_req.angle = 0
        seg_req.min_z = TABLE_MIN_Z
        seg_req.transform_to_reference_frame = True
        seg_req.reference_frame = 'base_link'

        seg_resp = self._bbox_segmentation_client(seg_req)
        xy_pose = seg_resp.object.primitive_pose

        print("obj_pose_x: {}, obj_pose_y: {}, obj_pose_z: {}".format(obj_pose.position.x, obj_pose.position.y, obj_pose.position.z))
        print("xy_pose_x: {}, xy_pose_y: {}, xy_pose_z: {}".format(xy_pose.position.x, xy_pose.position.y, xy_pose.position.z))

        grasp = Grasp()
        grasp.grasp_pose.header.frame_id = 'base_link'
        grasp.grasp_pose.pose.position.x = xy_pose.position.x
        grasp.grasp_pose.pose.position.y = xy_pose.position.y
        # grasp.grasp_pose.pose.position.z = obj_pose.position.z + obj_height / 2 - GRASP_DEPTH
        grasp.grasp_pose.pose.position.z = xy_pose.position.z - GRASP_DEPTH
        # grasp.grasp_pose.pose.position.z -= GRASP_DEPTH
        # grasp.grasp_pose.pose.position.z += INITIAL_GRASP_Z_OFFSET
        quat = T.quaternion_from_euler(0, math.radians(180), math.radians(-90) + angle, 'rxyz') # rotate by y to make it facing downwards
                                                                            # rotate by z to align with bbox orientation
        grasp.grasp_pose.pose.orientation.x = quat[0]
        grasp.grasp_pose.pose.orientation.y = quat[1]
        grasp.grasp_pose.pose.orientation.z = quat[2]
        grasp.grasp_pose.pose.orientation.w = quat[3]

        grasp.pre_grasp_approach.direction.header = seg_resp.object.header
        grasp.pre_grasp_approach.direction.vector.z = -1 # top pick
        grasp.pre_grasp_approach.desired_distance = APPROACH_DIST

        grasp.post_grasp_retreat.direction.header = seg_resp.object.header
        grasp.post_grasp_retreat.direction.vector.z = 1 # top pick
        grasp.post_grasp_retreat.desired_distance = RETREAT_DIST

        gripper_opening = grasp_box_width * GRASP_BOX_TO_GRIPPER_OPENING
        gripper_opening = min(gripper_opening, KORTEX_MAX_GRIPPER_OPENING) # clamp

        return grasp, gripper_opening

    def _top_grasp(self, grasp_box):
        if DUMMY_GRASP:
            return False

        grasp, gripper_opening = self._cal_initial_grasp(grasp_box)

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.EXECUTE_GRASP
        pnp_req.grasp = grasp
        pnp_req.gripper_opening = gripper_opening

        grasp.grasp_pose.pose.position.x += GRASP_POSE_X_OFFST # HACK!!!
        grasp.grasp_pose.pose.position.y += GRASP_POSE_Y_OFFST # HACK!!!
        grasp.grasp_pose.pose.position.z += APPROACH_DIST + KORTEX_GRIPPER_LENGTH + 0.025 - GRIPPER_WIDTH_TO_HEIGHT_PROP * gripper_opening + GRASP_POSE_Z_OFFST #HACK!!!
        gripper_opening += GRASP_WIDTH_OFFSET # HACK!!
        dist = APPROACH_DIST
        print(grasp)

        pnp_req.grasp = grasp
        pnp_req.gripper_opening = gripper_opening
        print(gripper_opening)

        # to_cont = raw_input("to_continue?")
        # if to_cont == "n":
        #     return False

        success = self._execute_grasp(pnp_req, dist)
        num_attempt = 1
        while not success and num_attempt < 5:
            logger.error('ERROR: robot grasp failed!!, again {}'.format(num_attempt))
            # to_cont = raw_input("try again?")
            # if to_cont != "y":
            #     return False
            # success = self._execute_grasp(pnp_req, dist)
            # num_attempt += 1

            return False
        return success

    def _execute_grasp(self, pnp_req, dist):
        res = self._kortex_robot.arm_move_to_pose(pnp_req.grasp.grasp_pose)
        if not res:
            return res
        self._kortex_robot.gripper_position(pnp_req.gripper_opening)
        self._kortex_robot.arm_move_in_cartesian(dz = -dist)
        self._kortex_robot.gripper_position(0)
        self._kortex_robot.arm_move_in_cartesian(dz = dist)

        return True

    def _move_arm_to_pose(self, target_pose):
        self._arm.move_to_pose(target_pose)

    def _get_place_target_pose(self):
        # this is hard coded for demo
        target_pose = PoseStamped()
        target_pose.header.frame_id="base_link"
        target_pose.pose.position.x = 0.6
        target_pose.pose.position.y = 0.45
        target_pose.pose.position.z = 0.6
        target_pose.pose.orientation.x = 0
        target_pose.pose.orientation.y = 0
        target_pose.pose.orientation.z = 0
        target_pose.pose.orientation.w = 1.0

        return target_pose

    # --------- Public ------- #
    def read_imgs(self):
        img_msg = rospy.wait_for_message('/kinect2/qhd/image_color', Image, timeout=10)
        img = self._br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        logger.info('img_size : {}'.format(img.shape))
        img = img[YCROP[0]:YCROP[1], XCROP[0]:XCROP[1]]
        logger.info('img_size : {}'.format(img.shape))

        depth = None
        return img, depth

    def grasp(self, grasp, is_target=False):
        res = self._top_grasp(grasp)
        if not res:
            self.move_arm_to_home()
            return False

        if not is_target:
            self.place_object()
            self.move_arm_to_home()
        else:
            self.give_obj_to_human()
        return res

    def point(self, grasp):
        grasp, gripper_opening = self._cal_initial_grasp(grasp)

        grasp.grasp_pose.pose.position.x += GRASP_POSE_X_OFFST # HACK!!!
        grasp.grasp_pose.pose.position.y += GRASP_POSE_Y_OFFST # HACK!!!
        grasp.grasp_pose.pose.position.z += APPROACH_DIST + KORTEX_GRIPPER_LENGTH + 0.025 - GRIPPER_WIDTH_TO_HEIGHT_PROP * gripper_opening + GRASP_POSE_Z_OFFST #HACK!!!

        res = self._kortex_robot.arm_move_to_pose(grasp.grasp_pose)
        if not res:
            return res
        self._kortex_robot.gripper_position(self._kortex_robot.GRIPPER_CLOSED_POS)

        return True

    def place_object(self):
        logger.info('place_object')

        place_pose = PoseStamped()
        place_pose.header.frame_id = 'base_link'
        place_pose.pose.position.x = 0.3
        place_pose.pose.position.y = 0.5 - self.place_cnt * 0.1
        place_pose.pose.position.z = 0.13
        quat = T.quaternion_from_euler(0, math.radians(180), math.radians(90), 'rxyz') # rotate by y to make it facing downwards
                                                                            # rotate by z to align with bbox orientation
        place_pose.pose.orientation.x = quat[0]
        place_pose.pose.orientation.y = quat[1]
        place_pose.pose.orientation.z = quat[2]
        place_pose.pose.orientation.w = quat[3]

        res = self._kortex_robot.arm_move_to_pose(place_pose)
        if not res:
            return res
        self._kortex_robot.arm_move_in_cartesian(dz = -0.1)
        rospy.sleep(0.5)
        self._kortex_robot.gripper_position(self._kortex_robot.GRIPPER_OPENED_POS)
        self._kortex_robot.arm_move_in_cartesian(dz = 0.1)

        self.place_cnt += 1

    def give_obj_to_human(self):
        print('give_obj_to_human')

        place_pose = PoseStamped()
        place_pose.header.frame_id = 'base_link'
        place_pose.pose.position.x = 0.4
        place_pose.pose.position.y = -0.45
        place_pose.pose.position.z = 0.13
        quat = T.quaternion_from_euler(0, math.radians(180), 0, 'rxyz') # rotate by y to make it facing downwards
                                                                            # rotate by z to align with bbox orientation
        place_pose.pose.orientation.x = quat[0]
        place_pose.pose.orientation.y = quat[1]
        place_pose.pose.orientation.z = quat[2]
        place_pose.pose.orientation.w = quat[3]

        res = self._kortex_robot.arm_move_to_pose(place_pose)
        if not res:
            return res
        self._kortex_robot.arm_move_in_cartesian(dz = -0.1)
        self._kortex_robot.gripper_position(self._kortex_robot.GRIPPER_OPENED_POS)
        self._kortex_robot.arm_move_in_cartesian(dz = 0.1)

        self.move_arm_to_home()

    def say(self, text):
        if DUMMY_SAY:
            print('Dummy execution of say: {}'.format(text))
            return True
        else:
            req = TextToSpeechRequest()
            req.text = text
            req.gender = 'female'
            res = self._text_to_speech_client(req)
            if res.success:
                print(res.filename)
                os.system("play {} >/dev/null".format(res.filename))

    def listen(self, timeout=None, asking_question=False):
        if DUMMY_LISTEN:
            logger.info('Dummy execution of listen')
            target_text = raw_input('Enter: ')
        else:
            logger.info('robot is listening')
            msg = rospy.wait_for_message('/rls_perception_services/speech_recognition_google/', String)
            text = msg.data.lower()

            ## replace some word
            text = text + " " # add a space behind to facilitate replacement
            for key in WORD_REPLACE_DICT:
                for word in  WORD_REPLACE_DICT[key]:
                    text = text.replace(word, key)
            text = text.strip() # strip the space added previously

            if not asking_question:
                ## Convert text into a list
                text_list = text.split(" ")
                for i in range(len(text_list)):
                    text_list[i] = text_list[i]

                ## Check invocation word
                # if text_list[0] != ROBOT_NAME:
                #     rospy.loginfo("CmdParser: Ignore speech_command, first word '{}' is not {}".format(text_list[0], ROBOT_NAME))
                #     self._cmd_callback({"id" : robot_def.ROBOT_UNKNOWN_CMD})
                #     return

                # after invocation word check, remove robot name from text.
                text = text.replace((ROBOT_NAME + " "), "")

                target_text = ''
                for i in range(len(text_list)):
                    if text_list[i] == "pick" and i < len(text_list) - 2 and text_list[i + 1] == 'up':
                        target_text = " ".join(text_list[i+2:])

            else:
                target_text = text

        logger.info('robot heard {}'.format(target_text))

        # say acknowledgement
        resp = random.choice(POSITIVE_RESPONSE_LIST)
        self.say(resp)

        return target_text

    def move_arm_to_home(self):
        self._kortex_robot.arm_move_to_home()