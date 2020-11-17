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
import stl
import open3d as o3d
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
import fetch_api

# import sys
# this_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, osp.join(this_dir, '../../'))

from config.config import CLASSES, ROOT_DIR
from libraries.data_viewer.data_viewer import DataViewer
import libraries.utils.o3d_ros_pc_converter as pc_converter
from libraries.grasp_collision_checker.grasp_collision_checker import GraspCollisionChecker
from libraries.utils.log import LOGGER_NAME

# ------- Settings ---------
GRASP_BOX_FOR_SEG = 1
BBOX_FOR_SEG = 2
GRASP_BOX_6DOF_PICK = 3
USE_REALSENSE = True
DUMMY_LISTEN = True
DUMMY_GRASP = False

# ------- Constants ---------
CONFIG_DIR = osp.join(ROOT_DIR, "config")
GRIPPER_FILE = "gripper_link.STL"
LEFT_GRIPPER_FINGER_FILE = "l_gripper_finger_link.STL"
RIGHT_GRIPPER_FINGER_FILE = "r_gripper_finger_link.STL"
LEFT_FINGER_POSE = {"link":[0., -0.101425, 0., 0., 0., 0.],
                    "joint": [0., -0.015425, 0., 0., 0., 0.],
                    "min_max_translate":[0.0, -0.04]} # - means the direction
RIGHT_FINGER_POSE = {"link":[0., 0.101425, 0., 0., 0., 0.],
                     "joint": [0., 0.015425, 0., 0., 0., 0.],
                     "min_max_translate":[0.0, 0.04]}
ORIG_IMAGE_SIZE = (480, 640)
# SCALE = 0.8
# Y_OFFSET = int(ORIG_IMAGE_SIZE[0] * (1 - SCALE) / 2)
# X_OFFSET = int(ORIG_IMAGE_SIZE[1] * (1 - SCALE) / 2)
# YCROP = (Y_OFFSET, ORIG_IMAGE_SIZE[0] - Y_OFFSET)
# XCROP = (X_OFFSET, ORIG_IMAGE_SIZE[1] - X_OFFSET)
YCROP = (470, 1000) # 1080
XCROP = (700, 1460) # 1920
FETCH_YCROP = (180, 450) # 480
FETCH_XCROP = (150, 490) # 640

FETCH_GRIPPER_LENGTH = 0.2
FETCH_MAX_GRIPPER_OPENING = 0.1
GRASP_DEPTH = 0.01
GRASP_POSE_X_OFFST = -0.00 # -0.018
GRASP_POSE_Y_OFFST = 0.000 # 0.02
GRASP_POSE_Z_OFFST = -0.001 # -0.015
GRASP_WIDTH_OFFSET = 0.0
GRIPPER_OPENING_OFFSET = 0.01
GRIPPER_OPENING_MAX = 0.09
PLACE_BBOX_SIZE = 80
APPROACH_DIST = 0.1
RETREAT_DIST = 0.15
GRASP_BOX_TO_GRIPPER_OPENING = 0.00045
PC_DOWNSAMPLE_SIZE = 0.002

POSITIVE_RESPONSE_LIST = ["Got it", "Sure", "No Problem", "okay", "certainly", "of course"]

# ---------- Statics ------------
logger = logging.getLogger(LOGGER_NAME)

class FetchRobot():
    def __init__(self):
        self._br = CvBridge()
        self._bbox_segmentation_client = rospy.ServiceProxy('rls_perception_services/bbox_pc_segmention_service', BBoxSegmentation)
        self._pnp_client = rospy.ServiceProxy('rls_control_services/fetch/pnp', PickPlace)
        # rospy.wait_for_service('/segment_table')
        self._table_segmentor_client = rospy.ServiceProxy('/segment_table', TableSegmentation)
        self._tf_transformer = tf.TransformerROS()
        self._fetch_image_client = rospy.ServiceProxy('/rls_perception_service/fetch/rgb_image_service', RetrieveImage)
        self._fetch_pc_client = rospy.ServiceProxy('/rls_control_service/fetch/retrieve_pc_service', RetrievePointCloud)
        self._fetch_speaker_client = rospy.ServiceProxy("rls_control_services/fetch/speaker_google", SpeakGoogle)
        self._tl = tf.TransformListener()
        self._arm = fetch_api.ArmV2()

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
        self.gripper_model = self._init_gripper_model()

        self._grasp_collision_checker = GraspCollisionChecker(self.gripper_model)

    def _init_gripper_model(self):
        """
        In this function, a gripper model should be initialized.
        The model includes a list of convex hulls represented by meshes.
        Each mesh can simply include a list of points (x, y, z) and
        triangle surfaces (p1, p2, p3), or be represented by a .obj object
        (a python library named pywavefront can provide a standard
        representation for .obj object). Coordinates should be w.r.t. the
        frame of the gripper itself.
        """
        gripper_model_path = osp.join(CONFIG_DIR, GRIPPER_FILE)
        l_finger_model_path = osp.join(CONFIG_DIR, LEFT_GRIPPER_FINGER_FILE)
        r_finger_model_path = osp.join(CONFIG_DIR, RIGHT_GRIPPER_FINGER_FILE)
        gripper_mesh = stl.mesh.Mesh.from_file(gripper_model_path)
        l_finger_mesh = stl.mesh.Mesh.from_file(l_finger_model_path)
        r_finger_mesh = stl.mesh.Mesh.from_file(r_finger_model_path)
        items = 1, 4, 7
        # since the model only imposes a y axis translate on the two fingers,
        # we here only consider this translate.
        l_finger_mesh.points[:, items] += LEFT_FINGER_POSE["link"][1] + LEFT_FINGER_POSE["joint"][1]
        r_finger_mesh.points[:, items] += RIGHT_FINGER_POSE["link"][1] + RIGHT_FINGER_POSE["joint"][1]
        # gripper_model = stl.mesh.Mesh(np.concatenate([gripper_mesh.data, l_finger_mesh.data, r_finger_mesh.data]))

        # mesh = o3d.io.read_triangle_mesh(l_finger_model_path)
        # o3d.visualization.draw_geometries([mesh])

        return {"gripper": gripper_mesh, "left_finger": l_finger_mesh, "right_finger": r_finger_mesh}

    def _vis_grasp(self, scene_pc, selected_grasp):
        if selected_grasp is None:
            return

        l_finger_model_path = osp.join(CONFIG_DIR, LEFT_GRIPPER_FINGER_FILE)
        r_finger_model_path = osp.join(CONFIG_DIR, RIGHT_GRIPPER_FINGER_FILE)
        l_finger_mesh = o3d.io.read_triangle_mesh(l_finger_model_path)
        r_finger_mesh = o3d.io.read_triangle_mesh(r_finger_model_path)
        l_finger = l_finger_mesh.sample_points_uniformly(number_of_points=500)
        r_finger = r_finger_mesh.sample_points_uniformly(number_of_points=500)
        gripper_width = selected_grasp["width"]
        l_finger_points=np.asarray(l_finger.points)
        r_finger_points=np.asarray(r_finger.points)
        l_finger_points[:, 1] -= gripper_width / 2
        r_finger_points[:, 1] += gripper_width / 2
        l_finger_points[:, 0] -= 0.03
        r_finger_points[:, 0] -= 0.03
        open3d_cloud = o3d.geometry.PointCloud()
        pc_in_g = self._trans_world_points_to_gripper(scene_pc, selected_grasp)
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(pc_in_g))
        l_finger.points = o3d.utility.Vector3dVector(l_finger_points)
        r_finger.points = o3d.utility.Vector3dVector(r_finger_points)
        # o3d.visualization.draw_geometries([l_finger, r_finger, open3d_cloud])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(l_finger)
        vis.add_geometry(r_finger)
        vis.add_geometry(open3d_cloud)
        vis.run()
        vis.destroy_window()

    def _cal_initial_grasp(self, grasp_box):
        logger.debug('grasp_box: {}'.format(grasp_box))
        grasp_box = grasp_box + np.tile([XCROP[0], YCROP[0]], 4)
        # for grasp box, x1,y1 is topleft, x2,y2 is topright, x3,y3 is btmright, x4,y4 is btmleft
        x1, y1, x2, y2, x3, y3, x4, y4 = grasp_box.tolist()
        seg_req = BBoxSegmentationRequest()
        seg_req.x = x1
        seg_req.y = y1
        seg_req.width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        seg_req.height = math.sqrt((x4 - x1)**2 + (y4 - y1)**2)
        seg_req.angle = math.atan2(y2 - y1, x2 - x1)
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
        # print("obj_x, y, z: {} {} {}".format(obj_length, obj_width, obj_height))

        grasp = Grasp()
        grasp.grasp_pose.header = seg_resp.object.header
        grasp.grasp_pose.pose = seg_resp.object.primitive_pose
        grasp.grasp_pose.pose.position.z += obj_height / 2 - GRASP_DEPTH
        quat = T.quaternion_from_euler(0, math.pi / 2, seg_req.angle, 'rzyx') # rotate by y to make it facing downwards
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
        gripper_opening = min(gripper_opening, FETCH_MAX_GRIPPER_OPENING) # clamp

        return grasp, gripper_opening

    def _top_grasp(self, grasp_box):
        if DUMMY_GRASP:
            return False

        grasp, gripper_opening = self._cal_initial_grasp(grasp_box)

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.EXECUTE_GRASP
        pnp_req.grasp = grasp
        pnp_req.gripper_opening = gripper_opening

        grasp_pose_tmp = grasp.grasp_pose
        new_grasp = self._get_collision_free_grasp(grasp_pose_tmp, pnp_req.gripper_opening)
        # to_cont = raw_input("to_continue?")
        # if to_cont != "y":
        #     return False

        if new_grasp is None:
            logger.error('ERROR: robot grasp failed!!')
            return False

        grasp.grasp_pose.pose.position.x = new_grasp["pos"][0]
        grasp.grasp_pose.pose.position.y = new_grasp["pos"][1]
        grasp.grasp_pose.pose.position.z = new_grasp["pos"][2]
        grasp.grasp_pose.pose.position.x += GRASP_POSE_X_OFFST # HACK!!!
        grasp.grasp_pose.pose.position.y += GRASP_POSE_Y_OFFST # HACK!!!
        grasp.grasp_pose.pose.position.z += APPROACH_DIST + FETCH_GRIPPER_LENGTH + GRASP_POSE_Z_OFFST #HACK!!!

        pnp_req.grasp = grasp
        pnp_req.gripper_opening = new_grasp["width"] + GRASP_WIDTH_OFFSET # HACK!!!

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            logger.error('ERROR: robot grasp failed!!')
        return resp.success

    def _move_arm_to_pose(self, target_pose):
        self._arm.move_to_pose(target_pose)

    def _get_place_target_pose(self):
        # this is hard coded for demo
        target_pose = PoseStamped()
        target_pose.header.frame_id="base_link"
        target_pose.pose.position.x = 0.519
        target_pose.pose.position.y = 0.519
        target_pose.pose.position.z = 0.98
        target_pose.pose.orientation.x = 0
        target_pose.pose.orientation.y = 0
        target_pose.pose.orientation.z = 0
        target_pose.pose.orientation.w = 1.0

        return target_pose

    def _get_scene_pc(self):
        start_time = time.time()
        resp = self._fetch_pc_client()
        raw_pc = resp.pointcloud
        end_time = time.time()
        logger.debug("getting pc takes {}".format(end_time - start_time))

        try:
            # raw_pc = rospy.wait_for_message("/camera/depth_registered/points", PointCloud2, timeout=20.0)
            trans, rot = self._tl.lookupTransform('base_link', raw_pc.header.frame_id, rospy.Time(0))
            transform_mat44 = np.dot(T.translation_matrix(trans), T.quaternion_matrix(rot))
        except Exception as e:
            rospy.logerr(e)
            return None

        start_time = time.time()
        # build uv array for segmentation
        uvs = []
        # for x in range(XCROP[0], XCROP[1]):
        #     for y in range(YCROP[0], YCROP[1]):
        #         uvs.append([x, y])
        for x in range(FETCH_XCROP[0], FETCH_XCROP[1]):
            for y in range(FETCH_YCROP[0], FETCH_YCROP[1]):
                uvs.append([x, y])

        points = pcl2.read_points(raw_pc, skip_nans=True, field_names=('x', 'y', 'z'), uvs=uvs)
        end_time = time.time()
        logger.debug("read pc takes {}s".format(end_time - start_time))

        start_time = time.time()
        points_out = np.array([[p[0], p[1], p[2], 1.0] for p in points]) # num_points x 4 # NOTE: this is slow!!!
        end_time = time.time()
        logger.debug("pc transform to base_link takes {}s".format(end_time - start_time))
        points_out = np.dot(points_out, transform_mat44.T)[:, :3] # num_points x 3
        end_time = time.time()
        logger.debug("pc transform to base_link takes {}s".format(end_time - start_time))

        start_time = time.time()
        logger.info("pc shape before downsample: {}".format(len(points_out)))
        open3d_cloud = o3d.geometry.PointCloud()
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(points_out))
        downpcd = open3d_cloud.voxel_down_sample(voxel_size=PC_DOWNSAMPLE_SIZE)

        scene_pc = np.array(downpcd.points)
        logger.info("pc shape after downsample: {}".format(scene_pc.shape))

        end_time = time.time()
        logger.debug("seg and downsample pc takes {}s".format(end_time - start_time))
        return scene_pc

    def _get_collision_free_grasp(self, orig_grasp, orig_opening):
        logger.info("checking grasp collision!!!")
        scene_pc = self._get_scene_pc()
        return self._grasp_collision_checker.get_collision_free_grasp(orig_grasp, orig_opening, scene_pc, vis_grasp=True)

    # --------- Public ------- #
    def read_imgs(self):
        # resp = self._fetch_image_client()
        # img = self._br.imgmsg_to_cv2(resp.image, desired_encoding='bgr8')
        if USE_REALSENSE:
            img_msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=10)
            img = self._br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            logger.info('img_size : {}'.format(img.shape))
            img = img[YCROP[0]:YCROP[1], XCROP[0]:XCROP[1]]
            logger.info('img_size : {}'.format(img.shape))
        else:
            resp = self._fetch_image_client()
            img = self._br.imgmsg_to_cv2(resp.image, desired_encoding='bgr8')
            logger.info('img_size : {}'.format(img.shape))
            img = img[FETCH_YCROP[0]:FETCH_YCROP[1], FETCH_XCROP[0]:FETCH_XCROP[1]]
            logger.info('img_size : {}'.format(img.shape))
        # depth_img_msg = rospy.wait_for_message('/head_camera/depth/image_rect', Image)
        # depth = self._br.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        depth = None
        return img, depth

    def grasp(self, grasp, is_target=False):
        # return self._top_grasp(bbox, grasp)
        if not is_target:
            # print('sampling where to place!!')
            # target_pose = self._sample_target_pose()
            target_pose = self._get_place_target_pose()

        res = self._top_grasp(grasp)
        if not res:
            self.move_arm_to_home()
            return False

        if not is_target:
            self.place_object(target_pose)
            self.move_arm_to_home()
        else:
            self.give_obj_to_human()
        return res

    def place_object(self, target_pose):
        logger.info('place_object')
        if target_pose is None:
            logger.error('ERROR: fail to find a place to place!!')
            return False

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.PLACE
        pnp_req.place_type = PickPlaceRequest.DROP

        pnp_req.target_pose = target_pose

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            logger.error('ERROR: place_object failed!!')
        return resp.success

    def give_obj_to_human(self):
        # print('Dummy execution of give_obj_to_human')

        # Hard code for demo
        # target_pose = PoseStamped()
        # target_pose.header.frame_id="base_link"
        # target_pose.pose.position.x = 0.8
        # target_pose.pose.position.y = 0
        # target_pose.pose.position.z = 1.1
        # quat = T.quaternion_from_euler(0, math.pi / 2, 0, 'rzyx') # rotate by y to make it facing downwards
        #                                                           # rotate by z to align with bbox orientation
        # target_pose.pose.orientation.x = quat[0]
        # target_pose.pose.orientation.y = quat[1]
        # target_pose.pose.orientation.z = quat[2]
        # target_pose.pose.orientation.w = quat[3]

        # self._move_arm_to_pose(target_pose)

        target_x = 0.6
        target_y = -0.25
        try:
            (trans, rot) = self._tl.lookupTransform('/base_link', '/wrist_roll_link', rospy.Time())
        except Exception as e:
            print(e)
            return False
        dx = target_x - trans[0]
        dy = target_y - trans[1]
        self._arm.move_in_cartesian(dx=dx, dy=dy)

    def say(self, text):
        # print('Dummy execution of say: {}'.format(text))
        resp = self._fetch_speaker_client(text)
        return resp.success

    def listen(self, timeout=None):
        if DUMMY_LISTEN:
            logger.info('Dummy execution of listen')
            text = raw_input('Enter: ')
        else:
            logger.info('robot is listening')
            msg = rospy.wait_for_message('/rls_perception_services/speech_recognition_google/', String)
            text = msg.data.lower()

        logger.info('robot heard {}'.format(text))

        # say acknowledgement
        resp = random.choice(POSITIVE_RESPONSE_LIST)
        self.say(resp)

        return text

    def move_arm_to_home(self):
        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.MOVE_ARM_TO_HOME

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            logger.error('ERROR: move_arm_to_home failed!!')
        return resp.success

if __name__=="__main__":
    r = R.from_euler("zyx", [0, math.pi / 2, 0])

    ply_model_path = "../../config/visual.ply"
    scene_pc = np.array(o3d.io.read_point_cloud(ply_model_path).points)
    scene_pc[:, 2] -= 0.01
    grasps = [{
        "pos": [0, 0, 0],
        "quat":r.as_quat().tolist(),
        "width": 0.05
    }]
    robot = FetchRobot()
    robot._get_collision_free_grasp_cfg(grasps[0], scene_pc, vis = True)




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

# def vis_mesh(mesh_list, pc_list, mesh_color="r", rotation = 1):
#     # Create a new plot
#     # rotation: 0 means no rotation,
#     #           1 means rotate w.r.t. x axis for 90 degrees
#     #           2 means rotate w.r.t. y axis for 90 degrees
#     #           3 means rotate w.r.t. z axis for 90 degrees
#     figure = pyplot.figure()
#     axes = mplot3d.Axes3D(figure)

#     rot_mat_x = np.array(
#         [[1, 0, 0],
#          [0, 0, -1],
#          [0, 1, 0]]
#     )
#     rot_mat_y = np.array(
#         [[1, 0, 0],
#          [0, 0, -1],
#          [0, 1, 0]]
#     )
#     rot_mat_z = np.array(
#         [[1, 0, 0],
#          [0, 0, -1],
#          [0, 1, 0]]
#     )
#     for pc in pc_list:
#         if rotation == 1:
#             pc = np.dot(pc, rot_mat_x.T)
#         elif rotation == 2:
#             pc = np.dot(pc, rot_mat_y.T)
#         elif rotation == 3:
#             pc = np.dot(pc, rot_mat_z.T)
#         x = pc[:, 0]
#         y = pc[:, 1]
#         z = pc[:, 2]
#         axes.scatter(x, y, z)

#     for i, mesh in enumerate(mesh_list):
#         if isinstance(mesh_color, (list, tuple)):
#             c = mesh_color[i]
#         else:
#             c = mesh_color
#         if rotation == 1:
#             mesh.rotate([0.5, 0.0, 0.0], math.radians(90))
#         elif rotation == 2:
#             mesh.rotate([0.0, 0.5, 0.0], math.radians(90))
#         elif rotation == 3:
#             mesh.rotate([0.0, 0.0, 0.5], math.radians(90))
#         axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors, facecolors=c))

#     # Auto scale to the mesh size
#     scale = np.concatenate([mesh.points.flatten(-1) for mesh in mesh_list])
#     axes.auto_scale_xyz(scale, scale, scale)
#     # Show the plot to the screen
#     pyplot.show()
"""
