# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import os.path as osp
from PIL import Image
import time
import datetime

# from sklearn.manifold import TSNE
from config.config import *
from paper_fig_generator import gen_paper_fig

def create_mrt(rel_mat, class_names=None, rel_score=None):
    '''
    rel_mat: np.array of size [num_box, num_box]
             where rel_mat[i, j] is the relationship between obj i and obj j.
             1 means i is the parent of j.
             2 means i is the child of j.
             3 means i has no relation to j.
    '''

    # using relationship matrix to create manipulation relationship tree
    mrt = nx.DiGraph()

    if rel_mat.size == 0:
        # No object is detected
        return mrt
    elif (rel_mat > 0).sum() == 0:
        # No relation is detected, meaning that there is only one object in the scene
        class_names = class_names or [0]
        mrt.add_node(class_names[0])
        return mrt

    node_num = np.max(np.where(rel_mat > 0)[0]) + 1
    if class_names is None:
        # no other node information
        class_names = list(range(node_num))
    elif isinstance(class_names[0], float):
        # normalized confidence score
        class_names = ["{:d}\n{:.2f}".format(i, cls) for i, cls in enumerate(class_names)]
    else:
        # class name
        class_names = ["{:s}{:d}".format(cls, i) for i, cls in enumerate(class_names)]

    if rel_score is None:
        rel_score = np.zeros(rel_mat.shape, dtype=np.float32)

    for obj1 in xrange(node_num):
        mrt.add_node(class_names[obj1])
        for obj2 in xrange(obj1):
            if rel_mat[obj1, obj2].item() == 1:
                # OBJ1 is the father of OBJ2
                mrt.add_edge(class_names[obj2], class_names[obj1],
                             weight=np.round(rel_score[obj1, obj2].item(), decimals=2))

            if rel_mat[obj1, obj2].item() == 2:
                # OBJ2 is the father of OBJ1
                mrt.add_edge(class_names[obj1], class_names[obj2],
                             weight=np.round(rel_score[obj1, obj2].item(), decimals=2))
    return mrt

def split_long_string(in_str, len_thresh = 30):
    if in_str =='':
        return ''

    in_str = in_str.split(" ")
    out_str = ''
    len_counter = 0
    for word in in_str:
        len_counter += len(word) + 1
        if len_counter > len_thresh:
            out_str += "\n" + word + " "
            len_counter = len(word) + 1
        else:
            out_str += word + " "
    return out_str

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

# Numpy data viewer to demonstrate detection results or ground truth.
class DataViewer(object):
    def __init__(self, classes):
        self.color_pool = [(255, 207, 136), (68, 187, 92), (153, 255, 0), (68, 187, 187), (0, 153, 255), (187, 68, 163),
                           (255, 119, 119), (116, 68, 187), (68, 187, 163), (163, 187, 68), (0, 204, 255), (68, 187, 140),
                           (204, 0, 255), (255, 204, 0), (102, 0, 255), (255, 0, 0), (68, 140, 187), (187, 187, 68),
                           (0, 255, 153), (119, 255, 146), (187, 163, 68), (187, 140, 68), (255, 153, 0), (255, 255, 0),
                           (153, 0, 255), (0, 255, 204), (68, 116, 187), (0, 255, 51), (187, 68, 68), (140, 187, 68),
                           (68, 163, 187), (187, 116, 68), (163, 68, 187), (204, 255, 0), (255, 0, 204), (0, 255, 255),
                           (140, 68, 187), (0, 102, 255), (153, 214, 255), (255, 102, 0)]
        self.classes = classes
        self.num_classes = len(self.classes)
        # Extend color_pool so that it is longer than classes
        self.color_pool = (self.num_classes / len(self.color_pool) + 1) * self.color_pool
        self.class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self.ind_to_class = dict(zip(xrange(self.num_classes), self.classes))
        self.color_dict = dict(zip(self.classes, self.color_pool[:self.num_classes]))

    def draw_single_bbox(self, img, bbox, bbox_color=(163, 68, 187), text_str="", test_bg_color = None):
        if test_bg_color is None:
            test_bg_color = bbox_color
        bbox = bbox.astype(np.int)
        bbox = tuple(bbox)
        text_rd = (bbox[2], bbox[1] + 25)
        cv2.rectangle(img, bbox[0:2], bbox[2:4], bbox_color, 2)
        cv2.rectangle(img, bbox[0:2], text_rd, test_bg_color, -1)
        cv2.putText(img, text_str, (bbox[0], bbox[1] + 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
        return img

    def draw_single_grasp(self, img, grasp, test_str=None, text_bg_color=(255, 0, 0)):
        gr_c = (int((grasp[0] + grasp[4]) / 2), int((grasp[1] + grasp[5]) / 2))
        for j in range(4):
            if j % 2 == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            p1 = (int(grasp[2 * j]), int(grasp[2 * j + 1]))
            p2 = (int(grasp[(2 * j + 2) % 8]), int(grasp[(2 * j + 3) % 8]))
            cv2.line(img, p1, p2, color, 2)

        # put text
        if test_str is not None:
            text_len = len(test_str)
            text_w = 17 * text_len
            gtextpos = (gr_c[0] - text_w / 2, gr_c[1] + 20)
            gtext_lu = (gr_c[0] - text_w / 2, gr_c[1])
            gtext_rd = (gr_c[0] + text_w / 2, gr_c[1] + 25)
            cv2.rectangle(img, gtext_lu, gtext_rd, text_bg_color, -1)
            cv2.putText(img, test_str, gtextpos,
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), thickness=2)
        return img

    def draw_graspdet(self, im, dets, g_inds=None):
        """
        :param im: original image numpy array
        :param dets: detections. size N x 8 numpy array
        :param g_inds: size N numpy array
        :return: im
        """
        # make memory contiguous
        im = np.ascontiguousarray(im)
        if dets.shape[0] == 0:
            return im

        dets = dets[(dets[:,:8].sum(-1)) > 0].astype(np.int)
        num_grasp = dets.shape[0]
        for i in range(num_grasp):
            im = self.draw_single_grasp(im, dets[i], str(g_inds[i]) if g_inds is not None else None)
        return im

    def draw_objdet(self, im, dets, o_inds = None, scores = None):
        """
        :param im: original image
        :param dets: detections. size N x 5 with 4-d bbox and 1-d class
        :return: im
        """
        # make memory contiguous
        im = np.ascontiguousarray(im)
        if dets.shape[0] == 0:
            return im

        dets = dets[(dets[:,:4].sum(-1)) > 0].astype(np.int)
        num_box = dets.shape[0]

        for i in range(num_box):
            cls = self.ind_to_class[dets[i, -1]]
            if scores is not None:
                cls = "{}:{:.2f}".format(cls, scores[i])
            if o_inds is None:
                im = self.draw_single_bbox(im, dets[i][:4], self.color_dict[cls], cls)
            else:
                im = self.draw_single_bbox(im, dets[i][:4], self.color_dict[cls], '%s%d' % (cls, o_inds[i]))
        return im

    def draw_graspdet_with_owner(self, im, o_dets, g_dets, g_inds=None):
        """
        :param im: original image numpy array
        :param o_dets: object detections. size N x 5 with 4-d bbox and 1-d class
        :param g_dets: grasp detections. size N x 8 numpy array
        :param g_inds: grasp indice. size N numpy array
        :return:
        """
        im = np.ascontiguousarray(im)
        if o_dets.shape[0] > 0:
            o_inds = np.arange(o_dets.shape[0])
            im = self.draw_objdet(im, o_dets, o_inds)
            im = self.draw_graspdet(im, g_dets, g_inds)
        return im

    def draw_mrt(self, img, rel_mat, class_names = None, rel_score = None, with_img = True, rel_img_size = 300):
        if rel_mat.shape[0] == 0:
            if with_img:
                return img
            else:
                # empty relation image
                return 255. * np.ones(img.shape)

        mrt = create_mrt(rel_mat, class_names, rel_score)
        # for e in mrt.edges():
        #     print(e)

        fig = plt.figure(0, figsize=(5, 5))
        pos = nx.circular_layout(mrt)
        nx.draw(mrt, pos, with_labels=True, font_size=16,
                node_color='#FFF68F', node_shape='s', node_size=1500, labels={node:node for node in mrt.nodes()})
        edge_labels = nx.get_edge_attributes(mrt, 'weight')
        nx.draw_networkx_edge_labels(mrt, pos, edge_labels=edge_labels, font_size=16)
        # grab the pixel buffer and dump it into a numpy array
        rel_img = fig2data(fig)

        rel_img = cv2.resize(rel_img[:,:,:3], (rel_img_size, rel_img_size), interpolation=cv2.INTER_LINEAR)
        # img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_LINEAR)
        if with_img:
            if min(img.shape[:2]) < rel_img_size:
                scalar = float(rel_img_size) / min(img.shape[:2])
                img = cv2.resize(img, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
            img[:rel_img_size, :rel_img_size] = rel_img
        else:
            img = rel_img

        plt.close(0)
        return img

    def draw_caption(self, im, dets, captions):
        im = np.ascontiguousarray(im)
        if dets.shape[0] == 0:
            return im

        dets = dets[(dets[:,:4].sum(-1)) > 0].astype(np.int)
        num_box = dets.shape[0]

        for i in range(num_box):
            cls = self.ind_to_class[dets[i, -1]]
            im = self.draw_single_bbox(im, dets[i][:4], self.color_dict[cls], '{}'.format(captions[i]))
        return im

    def draw_image_caption(self, im, caption, test_bg_color=(0,0,0)):
        text_rd = (im.shape[1], 25)
        cv2.rectangle(im, (0, 0), text_rd, test_bg_color, -1)
        cv2.putText(im, caption, (0, 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
        return im

    def draw_grounding_probs(self, im, expr, dets, ground_probs):
        im = np.ascontiguousarray(im)
        # img_caption = 'user expr: {}, bg prob: {:.2f}'.format(expr, ground_probs[-1])
        self.draw_image_caption(im, expr)

        # print(im.shape)

        # get ind of highest prob
        max_prob_ind = np.argmax(ground_probs)

        # Draw bg prob
        if max_prob_ind == len(ground_probs) - 1:
            cv2.rectangle(im, (0, im.shape[0]), (100, im.shape[0]-20), (0, 255, 0), -1)
        else:
            cv2.rectangle(im, (0, im.shape[0]), (100, im.shape[0]-20), (163, 68, 187), -1)
        cv2.putText(im, '{:.2f}'.format(ground_probs[-1]), (0, im.shape[0]),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)

        ground_probs = ground_probs[:-1]
        if dets.shape[0] == 0:
            return im
        dets = dets[(dets[:,:4].sum(-1)) > 0].astype(np.int)
        assert dets.shape[0] == ground_probs.shape[0]
        num_box = dets.shape[0]

        for i in range(num_box):
            prob = '{:.2f}'.format(ground_probs[i])
            if i == max_prob_ind:
                im = self.draw_single_bbox(im, dets[i][:4], bbox_color=(0, 255, 0), text_str='{}: {}'.format(i, prob))
            else:
                im = self.draw_single_bbox(im, dets[i][:4], text_str='{}: {}'.format(i, prob))
        return im

    def add_bg_score_to_img(self, img, bg_score):
        cv2.rectangle(img, (0, img.shape[0]), (100, img.shape[0]-20), (163, 68, 187), -1)
        cv2.putText(img, '{:.2f}'.format(bg_score), (0, img.shape[0]),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
        return img

    def add_grasp_to_img(self, im, g_dets, g_inds=None):
        im = np.ascontiguousarray(im)
        im = self.draw_graspdet(im, g_dets, g_inds)
        return im

    def add_obj_classes_to_img(self, im, dets):
        im = np.ascontiguousarray(im)
        num_box = dets.shape[0]
        for i in range(num_box):
            cls = self.ind_to_class[dets[i, -1]]
            text_str = '{}: {}'.format(i, cls)
            cv2.putText(im, text_str, (0, 100+i*30),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
        return im

    def display_obj_to_grasp(self, im, bboxes, grasps, grasp_target_idx):
        im = np.ascontiguousarray(im)
        bboxes = bboxes.astype(np.int)
        im = self.draw_single_bbox(im, bboxes[grasp_target_idx][:4])
        im = self.draw_single_grasp(im, grasps[grasp_target_idx])
        return im

    def vis_action(self, action_str, shape, draw_arrow = False):
        im = 255. * np.ones(shape)
        action_str = action_str.split("\n")

        mid_line = im.shape[0] / 2
        dy = 32
        y_b = mid_line - dy * len(action_str)
        for i, string in enumerate(action_str):
            cv2.putText(im, string, (0, y_b + i * dy),
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 0), thickness=2)
        if draw_arrow:
            cv2.arrowedLine(im, (0, mid_line), (im.shape[1], mid_line), (0, 0, 0), thickness = 2, tipLength = 0.03)
        return im

    def generate_visualization_imgs(self, img, bboxes, classes, rel_mat, rel_score_mat,
        expr, target_prob, action=None, grasps=None, question_str=None, answer=None, im_id=None, tgt_size=500, save=True):

        if im_id is None:
            current_date = datetime.datetime.now()
            image_id = "{}-{}-{}-{}".format(current_date.year, current_date.month, current_date.day,
                                            time.strftime("%H:%M:%S"))

        ############ visualize
        # resize img for visualization
        scalar = float(tgt_size) / img.shape[0]
        img_show = cv2.resize(img, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
        vis_bboxes = bboxes * scalar
        vis_bboxes = np.concatenate([vis_bboxes, classes], axis=-1)
        if grasps is not None:
            grasps[:, :8] = grasps[:, :8] * scalar
        num_box = bboxes.shape[0]

        # object detection
        if grasps is not None:
            object_det_img = self.draw_graspdet_with_owner(img_show.copy(), vis_bboxes, grasps)
        else:
            object_det_img = self.draw_objdet(img_show.copy(), vis_bboxes, list(range(classes.shape[0])))

        # relationship detection
        vis_rel_score_mat = self.relscores_to_visscores(rel_score_mat)
        rel_det_img = self.draw_mrt(img_show.copy(), rel_mat, class_names=target_prob.tolist()[:-1],
                                        rel_score=vis_rel_score_mat, with_img=False, rel_img_size=500)
        rel_det_img = cv2.resize(rel_det_img, (img_show.shape[1], img_show.shape[0]))
        rel_det_img = self.add_bg_score_to_img(rel_det_img, target_prob.tolist()[-1])

        # grounding
        print("Target Probability: ")
        print(target_prob.tolist())
        ground_img = self.draw_grounding_probs(img_show.copy(), expr, vis_bboxes, target_prob)
        ground_img = self.add_obj_classes_to_img(ground_img, vis_bboxes)

        # action
        if action != None:
            target_idx = -1
            question_type = None
            print("Optimal Action:")
            if action < num_box:
                target_idx = action
                action_str = "Grasping object " + str(action) + " and ending the program"
            elif action < 2 * num_box:
                target_idx = action - num_box
                action_str = "Grasping object " + str(target_idx) + " and continuing"
            elif action < 3 * num_box:
                if question_str is not None:
                    action_str = question_str
                else:
                    action_str = integrase.Q1["type1"].format(str(target_idx - 2 * num_box) + "th object")
                question_type = "Q1_TYPE1"
            else:
                if target_prob[-1] == 1:
                    action_str = Q2["type2"]
                    question_type = "Q2_TYPE2"
                elif (target_prob[:-1] > 0.02).sum() == 1:
                    action_str = Q2["type3"].format(str(np.argmax(target_prob[:-1])) + "th object")
                    question_type = "Q2_TYPE3"
                else:
                    action_str = Q2["type1"]
                    question_type = "Q2_TYPE1"
            print(action_str)

            action_img_shape = list(img_show.shape)
            action_img = self.vis_action(split_long_string(action_str), action_img_shape)
        else:
            action_str = ''
            question_type = None
            action_img = np.zeros((img_show.shape), np.uint8)

        # grasps
        ## Only visualize this if action is grasping.
        ## Only visualize for the target object.
        if grasps is not None and target_idx >= 0:
            print("Grasping score: ")
            print(grasps[:, -1].tolist())
            ground_img = self.add_grasp_to_img(ground_img, np.expand_dims(grasps[target_idx], axis=0))

        # save result
        out_dir = LOG_DIR
        final_img = np.concatenate([
            np.concatenate([object_det_img, rel_det_img], axis=1),
            np.concatenate([ground_img, action_img], axis=1),
        ], axis=0)

        if save:
            if im_id is None:
                im_id = str(datetime.datetime.now())
                origin_name = im_id + "_origin.png"
                save_name = im_id + "_result.png"
            else:
                origin_name = im_id.split(".")[0] + "_origin.png"
                save_name = im_id.split(".")[0] + "_result.png"
            origin_path = os.path.join(out_dir, origin_name)
            save_path = os.path.join(out_dir, save_name)
            i = 1
            while (os.path.exists(save_path)):
                i += 1
                save_name = im_id.split(".")[0] + "_result_{:d}.png".format(i)
                save_path = os.path.join(out_dir, save_name)

            cv2.imwrite(origin_path, img)
            cv2.imwrite(save_path, final_img)

        return {"final_img": final_img,
                "origin_img": img_show,
                "od_img": object_det_img,
                "mrt_img": rel_det_img,
                "ground_img": ground_img,
                "action_str": split_long_string(action_str),
                "answer" : answer,
                "q_type": question_type}

    def gen_final_paper_fig(self, img, bboxes, classes, rel_mat, rel_score_mat,
        expr, target_prob, action, grasps=None, question_str=None, answer=None, im_id=None, tgt_size=500):
        imgs = self.generate_visualization_imgs(img, bboxes, classes, rel_mat, rel_score_mat, expr,
            target_prob, action, grasps, question_str, answer, im_id, tgt_size)
        gen_paper_fig(expr, [imgs])
        return True

    def relscores_to_visscores(self, rel_score_mat):
        return np.max(rel_score_mat, axis=0)

    def display_img(self, img, mode="matplotlib"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mode =="matplotlib":
            plt.axis('off')
            plt.imshow(img)
            plt.show()
        elif mode == "pil":
            im_pil = Image.fromarray(img)
            im_pil.show()

