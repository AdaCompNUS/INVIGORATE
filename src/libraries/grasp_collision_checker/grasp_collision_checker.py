import open3d as o3d
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import math
import torch
import stl
import os.path as osp
import sys
import logging

# this_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, osp.join(this_dir, '../../'))

from config.config import CLASSES, ROOT_DIR
from libraries.utils.log import LOGGER_NAME

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

# ------------- Settings -------------
ABSOLUTE_COLLISION_SCORE_THRESHOLD = 20
IN_GRIPPER_SCORE_THRESOHOLD = 20
BATCH_SIZE = 1000
VISUALIZE_GRASP = True
COLLISION_CHECK_EPSILON = 0.002

# ------------ Statics -----------
logger = logging.getLogger(LOGGER_NAME)

# ------------ Code -------------
class GraspCollisionChecker():
    def __init__(self, gripper_model):
        self._gripper_model = gripper_model

    def _vis_grasp(self, scene_pc, selected_grasp):
        if not VISUALIZE_GRASP:
            return

        if selected_grasp is None:
            return

        l_finger_model_path = osp.join(CONFIG_DIR, LEFT_GRIPPER_FINGER_FILE)
        r_finger_model_path = osp.join(CONFIG_DIR, RIGHT_GRIPPER_FINGER_FILE)
        l_finger_mesh = o3d.io.read_triangle_mesh(l_finger_model_path)
        r_finger_mesh = o3d.io.read_triangle_mesh(r_finger_model_path)
        l_finger = l_finger_mesh.sample_points_uniformly(number_of_points=500)
        r_finger = r_finger_mesh.sample_points_uniformly(number_of_points=500)
        gripper_width = selected_grasp[-1]
        l_finger_points=np.asarray(l_finger.points)
        r_finger_points=np.asarray(r_finger.points)
        l_finger_points[:, 1] -= 0.101425 + (gripper_width / 2) + (0.0115 / 2) # need to offset gripper width
        r_finger_points[:, 1] += 0.101425 + (gripper_width / 2) + (0.0115 / 2) # need to offset gripper width
        l_finger_points[:, 0] -= 0.03
        r_finger_points[:, 0] -= 0.03
        open3d_cloud = o3d.geometry.PointCloud()
        pc_in_g = self._trans_world_points_to_gripper(scene_pc, selected_grasp)
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(pc_in_g))
        l_finger.points = o3d.utility.Vector3dVector(l_finger_points)
        r_finger.points = o3d.utility.Vector3dVector(r_finger_points)
        o3d.visualization.draw_geometries([l_finger, r_finger, open3d_cloud])

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(l_finger)
        # vis.add_geometry(r_finger)
        # vis.add_geometry(open3d_cloud)
        # vis.run()
        # vis.destroy_window()


    def _sample_grasps_xyzw(self, grasp_cfg, xy=0, z=0, w=0, xy_step=0.005, z_step=0.0025, w_step=0.01):
        start_time = time.time()

        x_ori, y_ori, z_ori, width_ori = grasp_cfg[0], grasp_cfg[1], grasp_cfg[2], grasp_cfg[7]
        grasps = [grasp_cfg]

        for x in np.linspace(x_ori - xy, x_ori + xy, xy * 2 / xy_step + 1):
            for y in np.linspace(y_ori - xy, y_ori + xy, xy * 2 / xy_step + 1):
                for z1 in np.linspace(z_ori - z, z_ori + z, z * 2 / z_step + 1):
                    for width in np.linspace(width_ori - w, width_ori + w, w * 2 / w_step + 1):
                        if width >= 0.01 and width <= 0.1:
                            grasps.append(np.array([x, y, z1, grasp_cfg[3], grasp_cfg[4], grasp_cfg[5], grasp_cfg[6], width]))

        end_time = time.time()
        logger.debug("_sample_grasps_xyzw takes {}".format(end_time - start_time))
        return np.array(grasps)

    def _grasp_pose_to_rotmat(self, grasp):
        x, y, z, w = grasp[3], grasp[4], grasp[5], grasp[6]

        r = R.from_quat([x, y, z, w])
        matrix = r.as_dcm()
        translation = grasp[0:3].reshape((3, 1))
        matrix = np.concatenate([matrix, translation], axis = 1)
        matrix = np.concatenate([matrix, np.array([0, 0, 0, 1]).reshape((1, 4))], axis = 0)

        return np.mat(matrix)

    def _trans_world_points_to_gripper(self, scene_pc, grasp):
        rot_mat = self._grasp_pose_to_rotmat(grasp)

        inv_rot_mat = rot_mat.I
        scene_pc = np.concatenate([scene_pc, np.ones((scene_pc.shape[0], 1))], axis=1)
        scene_pc = (scene_pc * inv_rot_mat.T)[:, :3]

        return scene_pc

    def _trans_world_points_to_gripper_batch(self, scene_pc, grasps):
        # transfer point to respective grasp coordinates # TODO vectorize

        pc_in_g = np.zeros((len(grasps), len(scene_pc), 3), dtype=np.float) # num_grasps x num_points x 3 arrary
        for i in range(grasps.shape[0]):
            pc_in_g[i] = self._trans_world_points_to_gripper(scene_pc, grasps[i]) # num_points x 3

        # n_grasp x 4 x 4
        rot_mats = self._get_rot_mats(grasps)
        # N x 4
        scene_pc = np.concatenate([scene_pc, np.ones((scene_pc.shape[0], 1))], axis=1)
        # n_grasp x 4 x 4 -> 4 x 4 x n_grasp -> 4 x 4*n_grasp
        rot_mats = np.transpose(rot_mats, (1, 2, 0)).reshape(4, -1)
        # N x 4*n_grasp
        scene_pc = np.dot(scene_pc, rot_mats)
        # N x 4 x n_grasp
        scene_pc = scene_pc.reshape(scene_pc.shape[0], 4, -1)
        # n_grasp x N x 3
        pc_in_g = np.transpose(scene_pc, (2, 0, 1))[:, :, :3]
        return pc_in_g

    def _get_rot_mats(self, grasps):
        rot_mats = [self._grasp_pose_to_rotmat(g).I.T for g in grasps]
        return np.array(rot_mats)

    def _check_collison_for_cube(self, points, cube, epsilon=0):
        # input: a vertical cube representing a collision model, a point clouds to be checked
        # epsilon is the maximum tolerable error
        # output: whether the points collide with the cube
        dif_min = points - cube[0].reshape(1, 3) + epsilon
        dif_max = cube[1].reshape(1, 3) - points + epsilon
        return (((dif_min > 0).sum(-1) == 3) & ((dif_max > 0).sum(-1) == 3)).sum()

    def _check_collison_for_cube_batch(self, points, cube, epsilon=0, cuda=False):
        """
        @param points, np array, [num_grasps, num_points, 3]
        @parma cube, tuple, [[num_grasps, 3], [num_grasps, 3]]
        """
        num_grasps = points.shape[0]
        dif_min = points - cube[0].reshape(num_grasps, 1, 3) + epsilon # [num_grasps, num_points, 3]
        dif_max = cube[1].reshape(num_grasps, 1, 3) - points + epsilon # [num_grasps, num_points, 3]
        if cuda:
            collisions = (((dif_min > 0).sum(-1) == 3) & ((dif_max > 0).sum(-1) == 3)).sum(dim=1) # [num_grasps, 1]
        else:
            collisions = (((dif_min > 0).sum(-1) == 3) & ((dif_max > 0).sum(-1) == 3)).sum(axis=1) # [num_grasps, 1]
        return collisions

    def _check_grasp_collision(self, scene_pc, grasps, use_cuda=False):
        """
        given a 6-d pose of the gripper and the scene point cloud, return whether the grasp is collision-free
        collision-free grasp satisfies:
        1. some points in the point cloud are in the gripper range, i.e., the convex hull of the whole gripper
            will collide with the scene point cloud
        2. the gripper itself cannot collide with the point cloud.

        @param, scene_pc: n_points x 3
        @param, grasps, Nx4
        """
        if not use_cuda:
            start_time = time.time()
            num_grasps = grasps.shape[0]
            num_points = scene_pc.shape[0]

            l_finger_min_orig = self._gripper_model["left_finger"].min_.copy() # [3]
            l_finger_max_orig = self._gripper_model["left_finger"].max_.copy() # [3]
            r_finger_min_orig = self._gripper_model["right_finger"].min_.copy() # [3]
            r_finger_max_orig = self._gripper_model["right_finger"].max_.copy() # [3]

            l_finger_min = np.tile(l_finger_min_orig, (num_grasps, 1)) # num_grasps x 3
            l_finger_max = np.tile(l_finger_max_orig, (num_grasps, 1)) # num_grasps x 3
            r_finger_min = np.tile(r_finger_min_orig, (num_grasps, 1)) # num_grasps x 3
            r_finger_max = np.tile(r_finger_max_orig, (num_grasps, 1)) # num_grasps x 3

            l_finger_min[:, 1] -= grasps[:, -1] / 2 + (0.0115 /2) # need to offset grippr width
            l_finger_min[:, 0] -= 0.031                           # need to offset gripper length
            l_finger_max[:, 1] -= grasps[:, -1] / 2 + (0.0115 /2)
            l_finger_max[:, 0] -= 0.031
            r_finger_min[:, 1] += grasps[:, -1] / 2 + (0.0115 /2)
            r_finger_min[:, 0] -= 0.031
            r_finger_max[:, 1] += grasps[:, -1] / 2 + (0.0115 /2)
            r_finger_max[:, 0] -= 0.031

            gripper_min = np.minimum(l_finger_min, r_finger_min) # num_grasps x 3
            gripper_max = np.maximum(l_finger_max, r_finger_max) # num_grasps x 3
            gripper_min[:, 1] = np.minimum(l_finger_max[:, 1], r_finger_max[:, 1])
            gripper_max[:, 1] = np.maximum(l_finger_min[:, 1], r_finger_min[:, 1])

            # transfer point to respective grasp coordinates
            # n_grasp x 4 x 4
            rot_mats = self._get_rot_mats(grasps)
            # N x 4
            scene_pc = np.concatenate([scene_pc, np.ones((scene_pc.shape[0], 1))], axis=1)
            # n_grasp x 4 x 4 -> 4 x 4 x n_grasp -> 4 x 4*n_grasp
            rot_mats = np.transpose(rot_mats, (1, 2, 0)).reshape(4, -1)
            # N x 4*n_grasp
            scene_pc = np.dot(scene_pc, rot_mats)
            # N x 4 x n_grasp
            scene_pc = scene_pc.reshape(scene_pc.shape[0], 4, -1)
            # n_grasp x N x 3
            pc_in_g = np.transpose(scene_pc, (2, 0, 1))[:, :, :3]

            # check collision
            p_num_collided_l_finger = self._check_collison_for_cube_batch(pc_in_g, (l_finger_min, l_finger_max), epsilon=COLLISION_CHECK_EPSILON) # [num_grasps, 1]
            p_num_collided_r_finger = self._check_collison_for_cube_batch(pc_in_g, (r_finger_min, r_finger_max), epsilon=COLLISION_CHECK_EPSILON) # [num_grasps, 1]
            p_num_collided_convex_hull = self._check_collison_for_cube_batch(pc_in_g, (gripper_min, gripper_max), epsilon=-COLLISION_CHECK_EPSILON) # [num_grasps, 1]
            collision_scores = p_num_collided_l_finger + p_num_collided_r_finger # [num_grasps, 1]
            in_gripper_scores = p_num_collided_convex_hull # - collision_score # [num_grasps, 1]

            valid_mask = in_gripper_scores > 0
            collision_scores = collision_scores[valid_mask]
            in_gripper_scores = in_gripper_scores[valid_mask]
            valid_grasp_inds = np.flatnonzero(valid_mask)

            end_time = time.time()
            logger.debug("_check_grasp_collision takes {}".format(end_time - start_time))
            return collision_scores, in_gripper_scores, valid_grasp_inds

        else:
            # use CUDA

            start_time = time.time()
            num_grasps = grasps.shape[0]
            num_points = scene_pc.shape[0]
            grasps = torch.FloatTensor(grasps).cuda()
            scene_pc = torch.FloatTensor(scene_pc).cuda()

            l_finger_min_orig = self._gripper_model["left_finger"].min_.copy()  # [3]
            l_finger_max_orig = self._gripper_model["left_finger"].max_.copy()  # [3]
            r_finger_min_orig = self._gripper_model["right_finger"].min_.copy()  # [3]
            r_finger_max_orig = self._gripper_model["right_finger"].max_.copy()  # [3]

            l_finger_min = torch.FloatTensor(np.tile(l_finger_min_orig, (num_grasps, 1))).cuda()  # num_grasps x 3
            l_finger_max = torch.FloatTensor(np.tile(l_finger_max_orig, (num_grasps, 1))).cuda()  # num_grasps x 3
            r_finger_min = torch.FloatTensor(np.tile(r_finger_min_orig, (num_grasps, 1))).cuda()  # num_grasps x 3
            r_finger_max = torch.FloatTensor(np.tile(r_finger_max_orig, (num_grasps, 1))).cuda()  # num_grasps x 3

            l_finger_min[:, 1] -= grasps[:, -1] / 2 + (0.0115 /2) # need to offset grippr width
            l_finger_min[:, 0] -= 0.031                           # need to offset gripper length
            l_finger_max[:, 1] -= grasps[:, -1] / 2 + (0.0115 /2)
            l_finger_max[:, 0] -= 0.031
            r_finger_min[:, 1] += grasps[:, -1] / 2 + (0.0115 /2)
            r_finger_min[:, 0] -= 0.031
            r_finger_max[:, 1] += grasps[:, -1] / 2 + (0.0115 /2)
            r_finger_max[:, 0] -= 0.031

            gripper_min = torch.min(l_finger_min, r_finger_min)  # num_grasps x 3
            gripper_max = torch.max(l_finger_max, r_finger_max)  # num_grasps x 3
            gripper_min[:, 1] = torch.min(l_finger_max[:, 1], r_finger_max[:, 1])
            gripper_max[:, 1] = torch.max(l_finger_min[:, 1], r_finger_min[:, 1])

            # transfer point to respective grasp coordinates
            # n_grasp x 4 x 4
            rot_mats = torch.FloatTensor(self._get_rot_mats(grasps.cpu().numpy())).cuda()
            # N x 4
            scene_pc = torch.cat([scene_pc, torch.ones((scene_pc.shape[0], 1)).cuda()], dim=1)
            # n_grasp x 4 x 4 -> 4 x 4 x n_grasp -> 4 x 4*n_grasp
            rot_mats = rot_mats.permute(1, 2, 0).reshape(4, -1)
            # N x 4*n_grasp
            scene_pc = torch.mm(scene_pc, rot_mats)
            # N x 4 x n_grasp
            scene_pc = scene_pc.reshape(scene_pc.shape[0], 4, -1)
            # n_grasp x N x 3
            pc_in_g = scene_pc.permute(2, 0, 1)[:, :, :3]

            # check collision
            p_num_collided_l_finger = self._check_collison_for_cube_batch(pc_in_g, (l_finger_min, l_finger_max), epsilon=COLLISION_CHECK_EPSILON, cuda=True)  # [num_grasps, 1]
            p_num_collided_r_finger = self._check_collison_for_cube_batch(pc_in_g, (r_finger_min, r_finger_max), epsilon=COLLISION_CHECK_EPSILON, cuda=True)  # [num_grasps, 1]
            p_num_collided_convex_hull = self._check_collison_for_cube_batch(pc_in_g, (gripper_min, gripper_max), epsilon=-COLLISION_CHECK_EPSILON, cuda=True)  # [num_grasps, 1]
            collision_scores = p_num_collided_l_finger + p_num_collided_r_finger # [num_grasps, 1]
            in_gripper_scores = p_num_collided_convex_hull  # - collision_score # [num_grasps, 1]

            valid_mask = in_gripper_scores > 0
            collision_scores = collision_scores[valid_mask]
            in_gripper_scores = in_gripper_scores[valid_mask]
            valid_grasp_inds = torch.nonzero(valid_mask.view(-1)).view(-1)

            end_time = time.time()
            logger.debug("_check_grasp_collision takes {}".format(end_time - start_time))
            return collision_scores.cpu().numpy(), in_gripper_scores.cpu().numpy(), valid_grasp_inds.cpu().numpy()


    def _select_from_grasps(self, grasps, scene_pc, use_cuda=False):
        logger.info("_select_from_grasps: num_of_grasps: {}, num_of_points: {}".format(grasps.shape[0], scene_pc.shape[0]))
        num_of_grasps = grasps.shape[0]
        num_processed = 0

        # process in batches to prevent gpu out of memory
        collision_scores_list = []
        in_gripper_scores_list = []
        valid_grasp_inds_list = []
        while num_processed < num_of_grasps:
            if num_processed + BATCH_SIZE >= num_of_grasps:
                num_to_process = num_of_grasps - num_processed
            else:
                num_to_process = BATCH_SIZE
            logger.debug("processing grasps {} to {}".format(num_processed, num_processed + num_to_process))
            grasps_batch = grasps[num_processed:num_processed + num_to_process]
            collision_scores, in_gripper_scores, valid_grasp_inds = self._check_grasp_collision(scene_pc, grasps_batch, use_cuda=use_cuda)
            collision_scores_list.append(collision_scores)
            in_gripper_scores_list.append(in_gripper_scores)
            valid_grasp_inds_list.append(valid_grasp_inds + num_processed)
            num_processed += num_to_process

        collision_scores = np.concatenate(collision_scores_list, axis=-1)
        in_gripper_scores = np.concatenate(in_gripper_scores_list, axis=-1)
        valid_grasp_inds = np.concatenate(valid_grasp_inds_list, axis=-1)

        if len(collision_scores) == 0:
            logger.error("ERROR: no collision free grasp detected!!")
            return None

        # here is a trick: to balance the collision and grasping part, we minus the collided point number from the
        # number of points in between the two grippers. 2 is a factor to measure how important collision is.
        # Also, you can use some other tricks. For example, you can choose the grasp with the maximum number of points
        # in between two grippers only from the collision free grasps (collision score = 0). However, in clutter, there
        # may be no completely collision-free grasps. Also, the noisy can make this method invalid.
        # hard threshold
        min_collision_score = np.min(collision_scores)
        logger.info("min_collision_score: {}".format(min_collision_score))
        valid_grasp_mask = (collision_scores <= ABSOLUTE_COLLISION_SCORE_THRESHOLD) * (in_gripper_scores >= IN_GRIPPER_SCORE_THRESOHOLD)
        valid_grasp_inds = valid_grasp_inds[valid_grasp_mask]
        collision_scores = collision_scores[valid_grasp_mask]
        in_gripper_scores = in_gripper_scores[valid_grasp_mask]

        if len(in_gripper_scores) <= 0:
            return None

        selected_ind = np.argmax(in_gripper_scores) # - collision_scores
        selected_grasp = grasps[valid_grasp_inds[selected_ind]]
        logger.info("collision_score: {}, in_gripper_score: {}".format(collision_scores[selected_ind], in_gripper_scores[selected_ind]))

        return selected_grasp

    def _get_collision_free_grasp_cfg(self, grasp, scene_pc):
        collision_scores, in_gripper_scores, valid_grasp_inds = self._check_grasp_collision(scene_pc, np.expand_dims(grasp, 0))
        if len(collision_scores) <= 0:
            logger.info("original grasp has not pc in between gripper!")
        else:
            logger.info("original grasp collision: {}, in_gripper_score: {}".format(collision_scores[0], in_gripper_scores[0]))

        selected_grasp = None

        # Step1 fine tune z first
        logger.info("Adjusting z!")
        grasps = self._sample_grasps_xyzw(grasp, z=0.03)
        selected_grasp = self._select_from_grasps(grasps, scene_pc)
        if selected_grasp is None:
            logger.warn("Adjusting z failed to produce good grasp, proceed")

        # step 2 fine tune width
        if selected_grasp is None:
            logger.info("Adjusting z and w!")
            grasps = self._sample_grasps_xyzw(grasp, w=0.03, z=0.03)
            selected_grasp = self._select_from_grasps(grasps, scene_pc)
            if selected_grasp is None:
                logger.warn("Adjusting zw failed to produce good grasp, proceed")

        # step 3 fine tune xyzw
        if selected_grasp is None:
            logger.info("Adjusting xyzw!")
            grasps = self._sample_grasps_xyzw(grasp, xy=0.025, z=0.03, w=0.02, z_step=0.005)
            selected_grasp = self._select_from_grasps(grasps, scene_pc, use_cuda=True)
            if selected_grasp is None:
                logger.warn("Adjusting xyzw failed to produce good grasp, proceed")

        return selected_grasp

    def get_collision_free_grasp(self, orig_grasp, orig_opening, scene_pc, vis_grasp=False):
        logger.info("checking grasp collision!!!")

        orig_grasp_dict = {
            "pos": [orig_grasp.pose.position.x, orig_grasp.pose.position.y, orig_grasp.pose.position.z],
            "quat": [orig_grasp.pose.orientation.x, orig_grasp.pose.orientation.y, orig_grasp.pose.orientation.z, orig_grasp.pose.orientation.w],
            "width": orig_opening
        }
        logger.info("orig_grasp: {} ".format(orig_grasp_dict))

        start_time = time.time()
        # further reduce the amount of pc to around the orig grasp # TODO vectorize
        x_min, x_max = orig_grasp_dict["pos"][0] - 0.1, orig_grasp_dict["pos"][0] + 0.1
        y_min, y_max = orig_grasp_dict["pos"][1] - 0.1, orig_grasp_dict["pos"][1] + 0.1
        valid_indices = []
        for i in range(scene_pc.shape[0]):
            if scene_pc[i, 0] >= x_min and scene_pc[i, 0] <= x_max and scene_pc[i, 1] >= y_min and scene_pc[i, 1] <= y_max:
                valid_indices.append(i)
        scene_pc_seg = scene_pc[valid_indices]
        end_time = time.time()
        logger.debug("segment pc takes {}s".format(end_time - start_time))
        logger.info("pc shape after further seg: {}".format(scene_pc_seg.shape))

        orig_grasp_array = np.array([orig_grasp.pose.position.x, orig_grasp.pose.position.y, orig_grasp.pose.position.z,
                                    orig_grasp.pose.orientation.x, orig_grasp.pose.orientation.y, orig_grasp.pose.orientation.z, orig_grasp.pose.orientation.w,
                                    orig_opening])

        # visualize initial grasp
        # self._vis_grasp(scene_pc_seg, orig_grasp_array)

        start_time = time.time()
        new_grasp = self._get_collision_free_grasp_cfg(orig_grasp_array, scene_pc_seg)
        end_time = time.time()
        logger.debug("check grasp collision completed, takes {}".format(end_time - start_time))

        if new_grasp is not None:
            new_grasp_dict = {
                "pos": [new_grasp[0], new_grasp[1], new_grasp[2]],
                "quat": orig_grasp_dict["quat"],
                "width": new_grasp[-1]
            }
        else:
            new_grasp_dict = None
        logger.info("new_grasp: {}".format(new_grasp_dict))

        if vis_grasp:
            self._vis_grasp(scene_pc_seg, new_grasp)

        return new_grasp_dict

if __name__=="__main__":
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
    gripper = {"gripper": gripper_mesh, "left_finger": l_finger_mesh, "right_finger": r_finger_mesh}

    r1 = R.from_euler("zyx", [0, math.pi / 2, 0])
    r2 = R.from_euler("zyx", [0, math.pi / 3, math.pi / 3])
    grasps = np.array([[0, 0, 0] + r1.as_quat().tolist(), [0.1, 0, 0] + r2.as_quat().tolist()])
    grasps = np.tile(grasps, (1000, 1))

    scene_pc = np.random.rand(20000, 3)
    col_ck = GraspCollisionChecker(gripper)
    print(col_ck._check_grasp_collision(scene_pc, grasps))
    print(col_ck._check_grasp_collision(scene_pc, grasps, use_cuda=True))
