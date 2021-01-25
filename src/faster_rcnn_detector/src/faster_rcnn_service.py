import rospy
import numpy as np
import os
import torch
import json
import cv2
from cv_bridge import CvBridge
import sys
sys.path.append("..")
import os.path as osp

cur_dir = osp.dirname(osp.abspath(__file__))
VMRN_ROOT_DIR = osp.join(cur_dir, '../vmrn')
ROOT_DIR = osp.join(cur_dir, '../..')

import vmrn._init_path
from vmrn.model.FasterRCNN import fasterRCNN
from vmrn.model.utils.config import read_cfgs, cfg
from vmrn.model.utils.blob import prepare_data_batch_from_cvimage
from vmrn.model.utils.net_utils import rel_prob_to_mat, find_all_paths, create_mrt, objdet_inference
from vmrn.roi_data_layer.roidb import combined_roidb
from vmrn.model.utils.data_viewer import dataViewer
from vmrn.model.rpn.bbox_transform import bbox_xy_to_xywh

from invigorate_msgs.srv import ObjectDetection, ObjectDetectionResponse

class FasterRCNNService(object):
    def __init__(self, args, model_path):
        self.br = CvBridge()
        conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
        # load trained model
        trained_model = torch.load(model_path)
        # init VMRN
        _,_,_,_,cls_list = combined_roidb(args.imdbval_name, training=False)
        self.RCNN = fasterRCNN(len(cls_list), class_agnostic=trained_model['class_agnostic'], feat_name=args.net,
                    feat_list=('conv' + conv_num,), pretrained=True)
        self.RCNN.create_architecture()
        self.RCNN.load_state_dict(trained_model['model'])
        if args.cuda:
            self.RCNN.cuda()
            self.cuda = True
        self.RCNN.eval()
        # init classes
        self.classes = cls_list
        self.class_to_ind = dict(zip(self.classes, xrange(len(cls_list))))
        self.ind_to_class = dict(zip(xrange(len(cls_list)), self.classes))

        # init data viewer
        self.data_viewer = dataViewer(self.classes)
        s = rospy.Service('faster_rcnn_server', ObjectDetection, self.det_serv_callback)
        print("Ready to detect object.")

    def det_serv_callback(self, req):
        img_msg = req.img

        # detect objects
        img = self.br.imgmsg_to_cv2(img_msg)
        data_batch = prepare_data_batch_from_cvimage(img, is_cuda = True)
        dets = self.fasterRCNN_forward_process(img, data_batch, save_res=True)
        obj_box = dets[0]
        print(obj_box.shape)
        obj_cls = dets[1]
        obj_cls_scores = dets[3]
        num_obj = dets[0].shape[0]

        # optionally, get regional feature
        regional_feat = {}
        if req.get_box_feat:
            regional_feat = self.get_region_features(img, data_batch[1][0][2], obj_box, obj_cls) #
            regional_feat['Feats'] = self.convert_regional_feat_to_python_list(regional_feat['Feats'])

        res = ObjectDetectionResponse()
        res.num_box = int(num_obj)
        res.bbox = obj_box.astype(np.float64).reshape(-1).tolist()
        res.cls = obj_cls.astype(np.int32).reshape(-1).tolist()
        res.cls_scores = obj_cls_scores.astype(np.float64).reshape(-1).tolist()
        res.box_feats = json.dumps("")
        return res

    def fasterRCNN_forward_process(self, image, data_batch, save_res=False, id =""):
        with torch.no_grad():
            result  = self.RCNN(data_batch)

        rois = result[0][0][:,1:5].data
        cls_prob = result[1][0].data
        bbox_pred = result[2][0].data
        obj_boxes, obj_cls_scores = objdet_inference(cls_prob, bbox_pred, data_batch[1][0], rois,
            class_agnostic=False, for_vis=True, recover_imscale=True, with_cls_score=True)
        if save_res:
            np.save(ROOT_DIR + "/images/output/" + id + "_bbox.npy", obj_boxes)

        obj_classes = obj_boxes[:, -1]
        obj_boxes = obj_boxes[:, :-1]
        num_box = obj_boxes.shape[0]
        obj_cls_name = []
        for cls in obj_classes:
            obj_cls_name.append(self.ind_to_class[cls])

        if save_res:
            obj_det_img = self.data_viewer.draw_objdet(image.copy(),
                np.concatenate((obj_boxes, np.expand_dims(obj_classes, 1)), axis = 1), o_inds=list(range(num_box)))
            cv2.imwrite(ROOT_DIR + "/images/output/" + id + "object_det.png", obj_det_img)

        return obj_boxes, obj_classes, obj_cls_name, obj_cls_scores

    def get_region_features(self, image, im_scales, obj_boxes, obj_classes):
        obj_boxes = obj_boxes[:, :4]

        # add to dets
        dets = []
        det_id = 0
        for idx, obj_class in enumerate(obj_classes):
            # detections: list of (n, 5), [xyxyc]
            x1, y1, x2, y2 = obj_boxes[idx]  # TODO check xywh or xyxy
            det = {'det_id': det_id,
                   'box': [x1, y1, x2-x1+1, y2-y1+1],
                   # 'category_name': category_name,
                   'category_id': obj_class,
                   # 'score': sc
                   }
            dets += [det]
            det_id += 1
        Dets = {det['det_id']: det for det in dets}
        det_ids = [det['det_id'] for det in dets]

        # Compute features
        # (n, 1024, 7, 7), (n, 2048, 7, 7) TODO
        print(obj_boxes)
        obj_boxes = torch.from_numpy(obj_boxes).to(dtype=torch.float32).cuda()
        obj_boxes = obj_boxes.unsqueeze(0)
        # print(obj_boxes)
        # img_scale = data_batch[1][0][2]
        print("img_scale {}".format(im_scales))
        pool5, fc7 = self.RCNN.box_to_spatial_fc7(self.RCNN.get_base_feat_cache(), obj_boxes, im_scales)
        print('pool5 shape {}'.format(pool5.shape))
        print('fc7 shape {}'.format(fc7.shape))
        lfeats = self.compute_lfeats(det_ids, Dets, image) # location feature against the image
        dif_lfeats = self.compute_dif_lfeats(det_ids, Dets) # location feature against five objects of the same category
        cxt_fc7, cxt_lfeats, cxt_det_ids = self.fetch_cxt_feats(det_ids, Dets, fc7)  # relational feature

        lfeats = torch.from_numpy(lfeats).cuda()
        dif_lfeats = torch.from_numpy(dif_lfeats).cuda()
        cxt_lfeats = torch.from_numpy(cxt_lfeats).cuda()

        # return
        data = {}
        data['det_ids'] = det_ids
        data['dets'] = dets
        # data['masks'] = masks
        data['cxt_det_ids'] = cxt_det_ids
        data['Feats'] = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                         'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats}

        return data

    def compute_lfeats(self, det_ids, Dets, im):
        '''
        object's location in image
        '''
        # Compute (n, 5) lfeats for given det_ids
        lfeats = np.empty((len(det_ids), 5), dtype=np.float32)
        for ix, det_id in enumerate(det_ids):
            det = Dets[det_id]
            x, y, w, h = det['box']
            ih, iw = im.shape[0], im.shape[1]
            lfeats[ix] = np.array(
                [[x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)]], np.float32)
        return lfeats

    def fetch_neighbour_ids(self, ref_det_id, Dets):
        '''
        For a given ref_det_id, we return
        - st_det_ids: same-type neighbouring det_ids (not including itself)
        - dt_det_ids: different-type neighbouring det_ids
        Ordered by distance to the input det_id
        '''
        ref_det = Dets[ref_det_id]
        x, y, w, h = ref_det['box']
        rx, ry = x+w/2, y+h/2

        def compare(det_id0, det_id1):
            x, y, w, h = Dets[det_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x, y, w, h = Dets[det_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer --> former
            if (rx-ax0)**2 + (ry-ay0)**2 <= (rx-ax1)**2 + (ry-ay1)**2:
                return -1
            else:
                return 1

        det_ids = list(Dets.keys())  # copy in case the raw list is changed
        det_ids = sorted(det_ids, cmp=compare)
        st_det_ids, dt_det_ids = [], []
        for det_id in det_ids:
            if det_id != ref_det_id:
                if Dets[det_id]['category_id'] == ref_det['category_id']:
                    st_det_ids += [det_id]
                else:
                    dt_det_ids += [det_id]
        return st_det_ids, dt_det_ids

    def compute_dif_lfeats(self, det_ids, Dets, topK=5):
        '''
        object's location wrt to other objects of the same category
        '''
        # return ndarray float32 (#det_ids, 5*topK)
        dif_lfeats = np.zeros((len(det_ids), 5*topK), dtype=np.float32)
        for i, ref_det_id in enumerate(det_ids):
            # reference box
            rbox = Dets[ref_det_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2] / \
                2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            # candidate boxes
            st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id, Dets)
            for j, cand_det_id in enumerate(st_det_ids[:topK]):
                cbox = Dets[cand_det_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = \
                    np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx) / \
                            rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return dif_lfeats

    def fetch_cxt_feats(self, det_ids, Dets, spatial_fc7, topK=5, with_st=1):
        '''
        object's location wrt to other objects of the different category

        Arguments:
        - det_ids    : list of det_ids
        - Dets       : each det is {det_id, box, category_id, category_name}
        - spatial_fc7: (#det_ids, 2048, 7, 7) cuda
        Return
        - cxt_feats  : cuda (#det_ids, topK, feat_dim)
        - cxt_lfeats : ndarray (#det_ids, topK, 5)
        - cxt_det_ids: [[det_id]] of size (#det_ids, topK), padded with -1
        Note we use neighbouring objects for computing context objects, zeros padded.
        '''
        fc7 = spatial_fc7.mean(3).mean(2)  # (n, 2048)
        cxt_feats = spatial_fc7.data.new(len(det_ids), topK, 2048).zero_()
        cxt_lfeats = np.zeros((len(det_ids), topK, 5), dtype=np.float32)
        cxt_det_ids = -np.ones((len(det_ids), topK),
                               dtype=np.int32)  # (#det_ids, topK)
        for i, ref_det_id in enumerate(det_ids):
            # reference box
            rbox = Dets[ref_det_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2] / \
                2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            # candidate boxes
            st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id, Dets)
            if with_st > 0:
                cand_det_ids = dt_det_ids + st_det_ids
            else:
                cand_det_ids = dt_det_ids
            cand_det_ids = cand_det_ids[:topK]
            for j, cand_det_id in enumerate(cand_det_ids):
                cand_det = Dets[cand_det_id]
                cbox = cand_det['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                cxt_lfeats[i, j, :] = np.array(
                    [(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
                cxt_feats[i, j, :] = fc7[det_ids.index(cand_det_id)]
                cxt_det_ids[i, j] = cand_det_id
        cxt_det_ids = cxt_det_ids.tolist()
        return cxt_feats, cxt_lfeats, cxt_det_ids

    def convert_regional_feat_to_python_list(self, feats):
        pool5 = feats['pool5'].detach().cpu().numpy().tolist()
        fc7 = feats['fc7'].detach().cpu().numpy().tolist()
        lfeats = feats['lfeats'].detach().cpu().numpy().tolist()
        dif_lfeats = feats['dif_lfeats'].detach().cpu().numpy().tolist()
        cxt_fc7 = feats['cxt_fc7'].detach().cpu().numpy().tolist()
        cxt_lfeats = feats['cxt_lfeats'].detach().cpu().numpy().tolist()

        feats['pool5'] = pool5
        feats['fc7'] = fc7
        feats['lfeats'] = lfeats
        feats['dif_lfeats'] = dif_lfeats
        feats['cxt_fc7'] = cxt_fc7
        feats['cxt_lfeats'] = cxt_lfeats

        return feats

if __name__ == '__main__':
    rospy.init_node('faster_rcnn_server')
    # we need to read configs of VMRN that were used in training and also need to be used in this demo
    args = read_cfgs()
    args.cuda = True
    fasterrcnn_service = FasterRCNNService(args, os.path.join(VMRN_ROOT_DIR, "output" , "vmrdext" , "res101", "faster_rcnn_1_13_18301.pth"))
    rospy.spin()



