import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os.path as osp
import tf
from tf import transformations as t
import math
import numpy as np
import random

from rls_perception_msgs.srv import *
from rls_control_msgs.srv import *
from geometry_msgs.msg import *
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import Grasp

import sys
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(this_dir, '../../'))
from config.config import CLASSES
from libraries.data_viewer.data_viewer import DataViewer
import stl
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# ------- Settings ---------
GRASP_BOX_FOR_SEG = 1
BBOX_FOR_SEG = 2
GRASP_BOX_6DOF_PICK = 3
USE_REALSENSE = True

# ------- Constants ---------
CONFIG_DIR = "../../config"
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
if USE_REALSENSE:
    # YCROP = (180, 450)
    # XCROP = (200, 500)
    YCROP = (500, 1000)
    XCROP = (700, 1460)
else:
    YCROP = (180, 450)
    XCROP = (150, 490)
FETCH_GRIPPER_LENGTH = 0.2
GRASP_DEPTH = 0.04
GRASP_POSE_X_OFFST = 0
GRIPPER_OPENING_OFFSET = 0.01
PLACE_BBOX_SIZE = 50
APPROACH_DIST = 0.1
RETREAT_DIST = 0.15

def vis_mesh(mesh_list, pc_list, mesh_color="r", rotation = 1):
    # Create a new plot
    # rotation: 0 means no rotation,
    #           1 means rotate w.r.t. x axis for 90 degrees
    #           2 means rotate w.r.t. y axis for 90 degrees
    #           3 means rotate w.r.t. z axis for 90 degrees
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    rot_mat_x = np.array(
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]]
    )
    for pc in pc_list:
        if rotation == 1:
            pc = np.dot(pc, rot_mat_x.T)
        elif rotation == 2:
            raise NotImplementedError
        elif rotation == 3:
            raise NotImplementedError
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        axes.scatter(x, y, z)

    for i, mesh in enumerate(mesh_list):
        if isinstance(mesh_color, (list, tuple)):
            c = mesh_color[i]
        else:
            c = mesh_color
        if rotation == 1:
            mesh.rotate([0.5, 0.0, 0.0], math.radians(90))
        elif rotation == 2:
            mesh.rotate([0.0, 0.5, 0.0], math.radians(90))
        elif rotation == 3:
            mesh.rotate([0.0, 0.0, 0.5], math.radians(90))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors, facecolors=c))

    # Auto scale to the mesh size
    scale = np.concatenate([mesh.points.flatten(-1) for mesh in mesh_list])
    axes.auto_scale_xyz(scale, scale, scale)
    # Show the plot to the screen
    pyplot.show()


class FetchRobot():
    def __init__(self):
        self._br = CvBridge()
        self._bbox_segmentation_client = rospy.ServiceProxy('rls_perception_services/bbox_pc_segmention_service', BBoxSegmentation)
        self._pnp_client = rospy.ServiceProxy('rls_control_services/fetch/pnp', PickPlace)
        # rospy.wait_for_service('/segment_table')
        self._table_segmentor_client = rospy.ServiceProxy('/segment_table', TableSegmentation)
        self._tf_transformer = tf.TransformerROS()
        self._fetch_image_client = rospy.ServiceProxy('/rls_perception_service/fetch/rgb_image_service', RetrieveImage)
        self._fetch_speaker_client = rospy.ServiceProxy("rls_control_services/fetch/speaker_google", SpeakGoogle)

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
        return {"gripper": gripper_mesh, "left_finger": l_finger_mesh, "right_finger": r_finger_mesh}

    def _sample_grasps(self, grasp_cfg, sampler_cfg={"z_step": 0.005, "xy_step": 0.01, "w_step": 0.015}):
        """
        sample grasp configuration according to the rectangle representation.
        grasp: {"pos":[x,y,z], "quat": [x,y,z,w]}
        """
        def sample_dist(item, step, n_step):
            return [i * step + item for i in range(-n_step/2 + 1, n_step/2 + 1)]

        grasps = []
        x_ori, y_ori, z_ori = grasp_cfg["pos"]
        for x in sample_dist(x_ori, sampler_cfg["xy_step"], 3):
            for y in sample_dist(y_ori, sampler_cfg["xy_step"], 3):
                for z in sample_dist(z_ori, sampler_cfg["z_step"], 10):
                    for w in np.arange(0.02, 0.09, sampler_cfg["w_step"]):
                        grasps.append({"pos": [x,y,z], "quat": grasp_cfg["quat"], "width": w})
        return grasps

    def _grasp_pose_to_rotmat(self, grasp):
        x, y, z, w = grasp["quat"]
        return np.mat([
            [1 - 2*y*y - 2*z*z, 2*x*y + 2*w*z, 2*x*z-2*w*y, grasp["pos"][0]],
            [2*x*y - 2*w*z, 1 - 2*x*x - 2*z*z, 2*z*y + 2*w*x, grasp["pos"][1]],
            [2*x*z + 2*w*y, 2*z*y - 2*w*x,1 - 2*x*x - 2*y*y, grasp["pos"][2]],
            [0, 0, 0, 1]
        ]).T

    def _trans_world_points_to_gripper(self, scene_pc, grasp):
        # TODO: This function should be replaced by the one implemented in some well-known libraries.
        rot_mat = self._grasp_pose_to_rotmat(grasp)

        inv_rot_mat = rot_mat.I
        scene_pc = np.concatenate([scene_pc, np.ones((scene_pc.shape[0], 1))], axis=1)
        scene_pc = (scene_pc * inv_rot_mat.T)[:, :3]
        return scene_pc

    def _check_collison_for_cube(self, points, cube, epsilon=0.005):
        # input: a vertical cube representing a collision model, a point clouds to be checked
        # epsilon is the maximum tolerable error
        # output: whether the points collide with the cube
        dif_min = points - cube[0].reshape(1, 3) + epsilon
        dif_max = cube[1].reshape(1, 3) - points + epsilon
        return (((dif_min > 0).sum(-1) == 3) & ((dif_max > 0).sum(-1) == 3)).sum()

    def _check_grasp_collision(self, scene_pc, grasps):
        """
        given a 6-d pose of the gripper and the scene point cloud, return whether the grasp is collision-free
        collision-free grasp satisfies:
        1. some points in the point cloud are in the gripper range, i.e., the convex hull of the whole gripper
            will collide with the scene point cloud
        2. the gripper itself cannot collide with the point cloud.
        """

        base_grasp = grasps[0]
        scene_pc = self._trans_world_points_to_gripper(scene_pc, grasps[0])

        valid_grasp_inds = []
        collision_scores = []
        in_gripper_scores = []

        for ind, g in enumerate(grasps):
            gripper_width = g["width"]
            # for the left finger, the opening along the y-axis is negative
            l_finger_min = self.gripper_model["left_finger"].min_.copy()
            l_finger_max = self.gripper_model["left_finger"].max_.copy()
            l_finger_min[1] -= gripper_width / 2
            l_finger_max[1] -= gripper_width / 2
            # for the right finger, the opening along the y-axis is positive
            r_finger_min = self.gripper_model["right_finger"].min_.copy()
            r_finger_max = self.gripper_model["right_finger"].max_.copy()
            r_finger_min[1] += gripper_width / 2
            r_finger_max[1] += gripper_width / 2

            # convex hull
            gripper_min = np.minimum(l_finger_min, r_finger_min)
            gripper_max = np.maximum(l_finger_max, r_finger_max)
            # offsets w.r.t. base_grasp
            xyz_offset = np.expand_dims(np.array(base_grasp["pos"]) - np.array(g["pos"]), axis=0)
            pc_in_g = scene_pc + xyz_offset

            p_num_collided_l_finger = self._check_collison_for_cube(pc_in_g, (l_finger_min, l_finger_max))
            p_num_collided_r_finger = self._check_collison_for_cube(pc_in_g, (r_finger_min, r_finger_max))
            p_num_collided_convex_hull = self._check_collison_for_cube(pc_in_g, (gripper_min, gripper_max))
            if in_gripper_scores > 0:
                collision_score = p_num_collided_l_finger + p_num_collided_r_finger
                collision_scores.append(collision_score)
                in_gripper_scores.append(p_num_collided_convex_hull - collision_score)
                valid_grasp_inds.append(ind)

        return collision_scores, in_gripper_scores, valid_grasp_inds

    def _get_collision_free_grasp_cfg(self, grasp, scene_pc, vis=False):
        grasps = self._sample_grasps(grasp)
        collision_scores, in_gripper_scores, valid_grasp_inds = self._check_grasp_collision(scene_pc, grasps)
        # here is a trick: to balance the collision and grasping part, we minus the collided point number from the
        # number of points in between the two grippers. 2 is a factor to measure how important collision is.
        # Also, you can use some other tricks. For example, you can choose the grasp with the maximum number of points
        # in between two grippers only from the collision free grasps (collision score = 0). However, in clutter, there
        # may be no completely collision-free grasps. Also, the noisy can make this method invalid.
        selected_ind = valid_grasp_inds[np.argmax(np.array(in_gripper_scores) - 2 * np.array(collision_scores))]
        selected_grasp = grasps[selected_ind]
        if vis:
            gripper_width = selected_grasp["width"]
            gripper = stl.mesh.Mesh(self.gripper_model["gripper"].data.copy())
            l_finger = stl.mesh.Mesh(self.gripper_model["left_finger"].data.copy())
            r_finger = stl.mesh.Mesh(self.gripper_model["right_finger"].data.copy())
            items = 1, 4, 7
            l_finger.points[:, items] -= gripper_width / 2
            r_finger.points[:, items] += gripper_width / 2
            pc_in_g = self._trans_world_points_to_gripper(scene_pc, selected_grasp)
            p_num_collided_l_finger = self._check_collison_for_cube(pc_in_g, (l_finger.min_, l_finger.max_))
            p_num_collided_r_finger = self._check_collison_for_cube(pc_in_g, (r_finger.min_, r_finger.max_))
            print(p_num_collided_l_finger, p_num_collided_r_finger)
            vis_mesh([gripper, l_finger, r_finger], [pc_in_g], ["r", "b", "g"])
        return

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

    def grasp(self, grasp, is_target=False):
        # return self._top_grasp(bbox, grasp)
        if not is_target:
            print('sampling where to place!!')
            target_pose = self._sample_target_pose()

        res = self._top_grasp(grasp)
        if not res:
            return False

        if not is_target:
            self.place_object(target_pose)
            self.move_arm_to_home()
        else:
            self.give_obj_to_human()
        return res

    def place_object(self, target_pose):
        print('place_object')
        if target_pose is None:
            print('ERROR: fail to find a place to place!!')
            return False

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.PLACE
        pnp_req.place_type = PickPlaceRequest.DROP

        pnp_req.target_pose = target_pose

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            print('ERROR: place_object failed!!')
        return resp.success

    def give_obj_to_human(self):
        print('Dummy execution of give_obj_to_human')

    def say(self, text):
        # print('Dummy execution of say: {}'.format(text))
        resp = self._fetch_speaker_client(text)
        return resp.success

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

        # print('calling bbox segmentation service')
        seg_resp = self._bbox_segmentation_client(seg_req)
        # print(seg_resp.object)

        # to_continue = raw_input('to_continue?')
        # if to_continue != 'y':
        #     return False

        obj_pose = seg_resp.object.primitive_pose
        obj_width = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Y]
        obj_height = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Z]

        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.EXECUTE_GRASP

        grasp = Grasp()
        grasp.grasp_pose.header = seg_resp.object.header
        grasp.grasp_pose.pose = seg_resp.object.primitive_pose
        grasp.grasp_pose.pose.position.x += GRASP_POSE_X_OFFST # HACK!!!
        grasp.grasp_pose.pose.position.z += obj_height / 2 - GRASP_DEPTH + APPROACH_DIST + FETCH_GRIPPER_LENGTH
        quat = t.quaternion_from_euler(0, math.pi / 2, seg_req.angle, 'rzyx') # rotate by y to make it facing downwards
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

        pnp_req.grasp = grasp
        pnp_req.gripper_opening = obj_width + GRIPPER_OPENING_OFFSET

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            print('ERROR: robot grasp failed!!')
        return resp.success

    def move_arm_to_home(self):
        pnp_req = PickPlaceRequest()
        pnp_req.action = PickPlaceRequest.MOVE_ARM_TO_HOME

        resp = self._pnp_client(pnp_req)
        if not resp.success:
            print('ERROR: move_arm_to_home failed!!')
        return resp.success

    def _sample_target_pose(self):
        resp = self._table_segmentor_client(1)  # get existing result
        table_height = resp.marker.pose.position.z + resp.marker.scale.z / 2 + 0.03

        # try 10 times
        for i in range(10):
            print("_sample_target_pose, trying {} time".format(i))
            btmright_x = random.randint(PLACE_BBOX_SIZE, XCROP[0])
            btmright_y = random.randint(YCROP[0] + PLACE_BBOX_SIZE, YCROP[1])

            seg_req = BBoxSegmentationRequest()
            seg_req.x = btmright_x - PLACE_BBOX_SIZE
            seg_req.y = btmright_y - PLACE_BBOX_SIZE
            seg_req.width = PLACE_BBOX_SIZE
            seg_req.height = PLACE_BBOX_SIZE
            seg_req.transform_to_reference_frame = True
            seg_req.reference_frame = 'base_link'

            seg_resp = self._bbox_segmentation_client(seg_req)
            obj_pose = seg_resp.object.primitive_pose
            obj_height = seg_resp.object.primitive.dimensions[SolidPrimitive.BOX_Z]
            obj_pose.position.z += obj_height / 2
            if obj_pose.position.z < table_height:
                target_pose = PoseStamped()
                target_pose.header.frame_id="base_link"
                target_pose.pose = obj_pose
                print('_sample_target_pose: place_pose found!!!')
                return target_pose

        print('ERROR: _sample_target_pose: failed to find a place_pose!!!')
        return None

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
