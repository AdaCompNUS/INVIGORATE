import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import os
import datetime
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os.path as osp

import sys
sys.path.append("../vmrn_old")
from invigorate_msgs.srv import VmrDetection, VmrDetectionResponse

import model
print("model path: {}".format(model.__file__))

import _init_path
from model.utils.config import cfg, cfg_from_file
from model.utils.net_utils import rel_prob_to_mat, create_mrt
import model.AllinOne as ALL_IN_ONE

# check python paths
for p in sys.path: print(p)

# -------- Constants -------
VMRN_OLD_ROOT_DIR = '../vmrn_old'
lua_grasp_detec_file = 'demo.lua'
topn_grasp = 3

# -------- Static -------
br = CvBridge()

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

def draw_mrt(img, rel_mat, class_names=None, rel_score=None):
    if rel_mat.shape[0] == 0:
        return img

    mrt = create_mrt(rel_mat, class_names, rel_score)
    for e in mrt.edges():
        print(e)

    fig = plt.figure(0, figsize=(3, 3))
    pos = nx.circular_layout(mrt)
    nx.draw(mrt, pos, with_labels=True, arrowstyle='fancy', font_size=16,
            node_color='#FFF68F', node_shape='s', node_size=300, labels={node: node for node in mrt.nodes()})
    edge_labels = nx.get_edge_attributes(mrt, 'weight')
    nx.draw_networkx_edge_labels(mrt, pos, edge_labels=edge_labels)
    # grab the pixel buffer and dump it into a numpy array
    rel_img = fig2data(fig)

    rel_img = cv2.resize(rel_img[:, :, :3], (500, 500), interpolation=cv2.INTER_LINEAR)
    # img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    if min(img.shape[:2]) < 500:
        scalar = 500. / min(img.shape[:2])
        img = cv2.resize(img, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
    img[:500, :500] = rel_img
    plt.close(0)
    return img

def get_image_blob(im, size=600):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])

    target_size = size
    im_scale = float(target_size) / float(im_size_min)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

class vmrn_server(object):
    def __init__(self):
        cfg_file = osp.join(VMRN_OLD_ROOT_DIR, 'cfgs/vmrdcompv1_all_in_one_res101_DEMO.yml')
        cfg_from_file(cfg_file)
        net_name = 'all_in_one_1_25_1407_0.pth' # 'all_in_one_1_13_1407_3.pth'
        vmrd_classes = ('__background__',  # always index 0
                   'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
                   'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
                   'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
                   'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
                   'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch')
        load_name = os.path.join(VMRN_OLD_ROOT_DIR, 'output', 'res101', 'vmrdcompv1', net_name)
        self.net = ALL_IN_ONE.resnet(vmrd_classes, 101, pretrained=True, class_agnostic=False)
        self.net.create_architecture()
        print("load checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        self.net.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.RCNN_COMMON.POOLING_MODE = checkpoint['pooling_mode']
        print('load model successfully!')
        self.net.cuda()
        self.net.eval()

        self.color_pool = [
        (255, 0, 0), (255, 102, 0), (255, 153, 0), (255, 204, 0), (255, 255, 0),(204, 255, 0), (153, 255, 0),(0, 255, 51),
        (0, 255, 153),(0, 255, 204),(0, 255, 255),(0, 204, 255),(0, 153, 255),(0, 102, 255),(102, 0, 255),(153, 0, 255),
        (204, 0, 255),(255, 0, 204),(187, 68, 68),(187, 116, 68),(187, 140, 68),(187, 163, 68),(187, 187, 68),
        (163, 187, 68),(140, 187, 68),(68, 187, 92),(68, 187, 140),(68, 187, 163),(68, 187, 187),(68, 163, 187),
        (68, 140, 187),(68, 116, 187),(116, 68, 187),(140, 68, 187),(163, 68, 187),(187, 68, 163),(255, 119, 119),
        (255, 207, 136),(119, 255, 146),(153, 214, 255)]
        np.random.shuffle(self.color_pool)
        self.color_dict = {}
        for i, clsname in enumerate(vmrd_classes):
            self.color_dict[clsname] = self.color_pool[i]

        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.num_grasps = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)
        self.gt_grasps = torch.FloatTensor(1)
        # visual manipulation relationship matrix
        self.rel_mat = torch.FloatTensor(1)
        self.gt_grasp_inds = torch.LongTensor(1)
        # ship to cuda
        self.im_data = self.im_data.cuda()
        self.im_info = self.im_info.cuda()
        self.num_boxes = self.num_boxes.cuda()
        self.num_grasps = self.num_grasps.cuda()
        self.gt_boxes = self.gt_boxes.cuda()
        self.gt_grasps = self.gt_grasps.cuda()
        self.rel_mat = self.rel_mat.cuda()
        self.gt_grasp_inds = self.gt_grasp_inds.cuda()
        s = rospy.Service('vmrn_server', VmrDetection, self.vmrn_server_callback)
        print("Ready to detect manipulation relationships.")

    def vmrn_server_callback(self, req):
        img_msg = req.img
        img = br.imgmsg_to_cv2(img_msg)
        bbox = req.bbox
        num_box = len(bbox) / 4
        bbox = np.array(bbox).reshape((num_box, 4))
        bbox = torch.Tensor(bbox)
        self.gt_boxes.data.resize_(bbox.size()).copy_(bbox)
        self.num_boxes = torch.Tensor([self.gt_boxes.shape[0]]).type_as(self.num_boxes)
        self.gt_boxes = self.gt_boxes.unsqueeze(0)
        rel_mat, rel_score_mat, grasps = self.rel_det_with_gtbox_allinone(img)
        res = VmrDetectionResponse()
        res.rel_mat = rel_mat.astype(np.int32).reshape(-1).tolist()
        res.rel_score_mat = rel_score_mat.astype(np.float64).reshape(-1).tolist()
        res.grasps = grasps.astype(np.float64).reshape(-1).tolist()
        return res

    def rel_det_with_gtbox_allinone(self, img, im_id = None):
        im_rel = img.copy()
        img, im_scale = get_image_blob(img, size=800)
        im_info_np = np.array([[img.shape[0], img.shape[1], im_scale, im_scale]], dtype=np.float32)

        im_data_pt = torch.from_numpy(img)
        im_data_pt = im_data_pt.unsqueeze(0)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        self.im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        self.im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)

        self.gt_boxes[:, :, :4] = self.gt_boxes[:, :, :4] * im_scale
        gt = {
            'boxes': self.gt_boxes,
            'num_boxes': self.num_boxes,
            'im_info': self.im_info,
        }

        with torch.no_grad():
            rel_result, grasp_result = self.net.rel_forward_with_gtbox(self.im_data, gt)

        if im_id is None:
            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            im_id = now_time

        _, _, rels = rel_result
        rel_mat, rel_score_mat = rel_prob_to_mat(rels, self.gt_boxes.shape[1])
        # im_rel = draw_mrt(im_rel, rel_mat, rel_score=rel_score_mat)
        # if not os.path.exists("output/rob_result/result/"):
        #     os.makedirs("output/rob_result/result/")
        # cv2.imwrite('output/rob_result/result/' + im_id + 'reldet.jpg', im_rel)
        return rel_mat, rel_score_mat, grasp_result.cpu().numpy()

if __name__ == '__main__':
    rospy.init_node('vmrn_server')
    vmrn_server()
    rospy.spin()
