import sys
import os
import yaml
import PIL
import rospy
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt

python2_path_backup = []
for p in sys.path:
    if "python2.7" in p:
        sys.path.remove(p)
        python2_path_backup.append(p)
import cv2
for p in python2_path_backup:
    sys.path.append(p)

from cv_bridge import CvBridge
from PIL import Image

# import os.path as osp
# cur_dir = osp.dirname(osp.abspath(__file__))
# sys.path.append(osp.join(cur_dir, '../vqa_maskrcnn_benchmark'))
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.structures.bounding_box import BoxList

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
# from invigorate import INVIGORATE

from invigorate_msgs.srv import ObjectDetectionResponse, VmrDetectionResponse, GroundingResponse
from invigorate_msgs.srv import ObjectDetection, VmrDetection, Grounding

SENTENCE_COMPLETION=True

# -------- Static -------
# br = CvBridge()
# def test_cv_bridge(br):
#     rand_img = np.random.rand(255,255,3)
#     img_msg = br.cv2_to_imgmsg(rand_img)
#     recovered_img = br.imgmsg_to_cv2(img_msg)
#     assert (1 - (recovered_img == rand_img)).sum() == 0
# test_cv_bridge(br)

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

class VMRDObjDetector():

    def __init__(self):
        rospy.loginfo('waiting for services...')
        rospy.wait_for_service('faster_rcnn_server')
        self._obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        self._br = CvBridge()

        self.classes = ['__background__',  # always index 0
               'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
               'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote controller', 'can', 'tape', 'knife', 'wrench', 'cup', 'charger',
               'badminton', 'wallet', 'wrist developer', 'glasses', 'plier', 'headset',
               'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

    def detect_objs(self, img):
        obj_result = self._faster_rcnn_client(img)
        num_box = obj_result[0]
        if num_box == 0:
            return None, None, None, None

        bboxes = np.array(obj_result[1]).reshape(num_box, -1)
        print('_object_detection: \n{}'.format(bboxes))
        classes = np.array(obj_result[2]).reshape(num_box, 1)
        class_scores = np.array(obj_result[3]).reshape(num_box, -1)
        bboxes, classes, class_scores, num_box = self._bbox_filter(bboxes, classes, class_scores)

        class_names = [self.classes[i[0]] for i in classes]
        print('_object_detection classes: {}'.format(class_names))
        return bboxes, classes, class_scores, num_box, class_names

    def _faster_rcnn_client(self, img):
        img_msg = self._br.cv2_to_imgmsg(img)
        res = self._obj_det(img_msg, False)
        return res.num_box, res.bbox, res.cls, res.cls_scores

    def _bbox_filter(self, bbox, cls, cls_scores):
        # apply NMS
        keep = nms(torch.from_numpy(bbox[:, :-1]), torch.from_numpy(bbox[:, -1]), 0.7)
        num_box = keep.shape[0]
        keep = keep.view(-1).numpy().tolist()
        for i in range(bbox.shape[0]):
            if i not in keep and bbox[i][-1] > 0.9:
                keep.append(i)
        bbox = bbox[keep]
        cls = cls[keep]
        cls_scores = cls_scores[keep]
        return bbox, cls, cls_scores, num_box

class FeatureExtractor(object):
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser()
        self.detection_model = self._build_detection_model()

    def get_parser(self):
        parser = SimpleNamespace(model_file='../vilbert/data/detectron_model.pth',
                                 config_file='../vilbert/data/detectron_config.yaml',
                                 batch_size=1,
                                 num_features=100,
                                 feature_name="fc6",
                                 confidence_threshold=0,
                                 background=False,
                                 partition=0)
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_preprocess(self, im):
        # input: an opencv image represented by numpy array with BGR channels

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
            self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes, start_index:], dim=1)
            cls_prob = torch.max(scores[keep_boxes, start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "cls_prob": scores[keep_boxes].cpu().numpy(),
                }
            )

        return feat_list, info_list

    def get_batch_proposals(self, images, im_scales, im_infos, proposals):
        proposals_batch = []
        for idx, img_info in enumerate(im_infos):
            boxes_tensor = torch.from_numpy(proposals[idx]).to("cuda")
            orig_image_size = (img_info["width"], img_info["height"])
            boxes = BoxList(boxes_tensor, orig_image_size)
            image_size = (images.image_sizes[idx][1], images.image_sizes[idx][0])
            boxes = boxes.resize(image_size)
            proposals_batch.append(boxes)
        return proposals_batch

    def _load_img_from_paths(self, image_paths):

        imgs = []

        for image_path in image_paths:
            img = Image.open(image_path)
            im = np.array(img).astype(np.float32)
            if len(im.shape) == 3:
                # from RGB to BGR
                im = im[:, :, ::-1]
            imgs.append(im)

        return imgs


    def _get_img_blobs_from_cv_imgs(self, imgs):

        img_tensor, im_scales, im_infos = [], [], []

        for im in imgs:
            im, im_scale, im_info = self._image_preprocess(im)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        return img_tensor, im_scales, im_infos


    def _get_img_blobs_from_paths(self, image_paths):

        return self._get_img_blobs_from_cv_imgs(self._load_img_from_paths(image_paths))


    def get_detectron_features(self, input, proposals=None):

        img_tensor, im_scales, im_infos = input
        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")

        if proposals[0] is not None:
            proposals = self.get_batch_proposals(current_img_list, im_scales, im_infos, proposals)
            with torch.no_grad():
                output = self.detection_model(current_img_list, proposals=proposals)
        else:
            with torch.no_grad():
                output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i: i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = file_base_name + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)

    def extract_features_from_path(self, image_path, proposals=None):

        img_tensor, im_scales, im_infos = self._get_img_blobs_from_paths([image_path])

        features, infos = self.get_detectron_features((img_tensor, im_scales, im_infos), proposals=[proposals])

        return features, infos

    def extract_features_from_cv_image(self, img, proposals=None):

        img_tensor, im_scales, im_infos = self._get_img_blobs_from_cv_imgs([img])

        features, infos = self.get_detectron_features((img_tensor, im_scales, im_infos), proposals=[proposals])

        return features, infos


class VilbertServer(object):
    def __init__(self):

        self.args = self._init_args()
        rospy.loginfo("Loading BERT tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.bert_model, do_lower_case=self.args.do_lower_case
        )

        rospy.loginfo("Initializing Feature Extractor")
        self.feature_extractor = FeatureExtractor()

        rospy.loginfo("Initializing ViLBERT model")
        self.model = self._init_models()

        self.expr_max_length = 20
        self.num_regions = 101
        self.classes, self.color_dict = self._init_obj_cls()

        # self.obj_det_server = rospy.Service('faster_rcnn_server', ObjectDetection, self.objdet_server_callback)
        # self.rel_vmrn_server = rospy.Service('vilbert_obr_server', VmrDetection, self.vmrn_server_callback)
        self.ground_server = rospy.Service('vilbert_grounding_service', Grounding, self.vilbert_grounding_service_callback)
        # self.caption_server = rospy.Service('vilbert_captioning_server', ViLBERTCaptioning, self.vilbert_captioning_server_callback)

        print("Ready to detect manipulation relationships.")

    def vilbert_grounding_service_callback(self, req):
        img_msg = req.img
        img = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                         dtype=np.uint8, buffer=img_msg.data)  # bypass python3 cv_bridge
        img = img.astype(np.float32)
        bbox = req.bboxes
        num_box = int(len(bbox) / 4)
        bbox = np.array(bbox).reshape((num_box, 4))
        expr = req.expr
        grounding_scores = self.grounding(img, bbox, expr)
        grounding_scores = grounding_scores.reshape(-1).tolist()
        res = GroundingResponse()
        res.grounding_scores = grounding_scores
        return res

    def tokenize_batch(self, batch):
        return [self.tokenizer.convert_tokens_to_ids(sent) for sent in batch]

    def untokenize_batch(self, batch):
        return [self.tokenizer.convert_ids_to_tokens(sent) for sent in batch]

    def extract_features(self, img, bboxes=None):

        if bboxes is not None:
            features, infos = self.feature_extractor.extract_features_from_cv_image(img, proposals=bboxes[:, :4])
            self_features, self_infos = self.feature_extractor.extract_features_from_cv_image(img)
            features[0] = torch.cat([features[0], self_features[0]], dim=0)[:self.num_regions - 1]

            def combine_infos(infos, self_infos, num_regions):
                info = infos[0]
                self_info = self_infos[0]
                self_info["bbox"] = np.concatenate([info["bbox"], self_info["bbox"]], axis=0)[:num_regions]
                self_info["cls_prob"] = np.concatenate([info["cls_prob"], self_info["cls_prob"]], axis=0)[:num_regions]
                self_info["objects"] = np.concatenate([info["objects"], self_info["objects"]], axis=0)[:num_regions]
                self_info["num_boxes"] = num_regions
                return self_infos

            infos = combine_infos(infos, self_infos, self.num_regions - 1)

        else:
            features, infos = self.feature_extractor.extract_features_from_cv_image(img)

        return features, infos

    def vmr_detection(self, img, bboxes):
        features, spatials, image_mask = self._prepare_visual_input(img, bboxes)
        co_attention_mask = torch.zeros((1, self.num_regions, self.expr_max_length)).cuda()
        # TODO: here 19 represents the vmr detection for VMRD. 3D-VMRD will be supported later
        task_tokens = torch.from_numpy(np.array([19])).cuda().unsqueeze(0)
        text, text_mask, text_seg_ids = self._initialize_sent_for_vmr_det()

        num_box = bboxes.shape[0]
        ref_ids = (torch.arange(num_box).cuda().long() + 1)
        type_ids = self._ref_id_to_type_id(ref_ids, num_region=self.num_regions)

        with torch.no_grad():
            _, _, _, _, _, rel_pred_scores, _, _, _, _ = self.model(
                text, features, spatials, text_seg_ids, type_ids, text_mask, image_mask, co_attention_mask,
                task_tokens,
                output_all_attention_masks=True
            )

        rel_pred_scores = rel_pred_scores.reshape(self.num_regions-1, self.num_regions-1, 3)
        rel_pred_scores = rel_pred_scores[:num_box, :num_box] * (1. - torch.eye(num_box).unsqueeze(-1)).cuda()

        im_height, im_width = img.shape[:2]
        bboxes /= np.array([[im_width, im_height, im_width, im_height]])
        overlaps = iou(torch.tensor(bboxes).float().cuda(), spatials[0][1:num_box+1, :4].float())
        maps = torch.argmax(overlaps, 1)

        rel_pred_scores = rel_pred_scores[maps][:, maps]
        rel_pred_scores = rel_pred_scores.reshape((num_box, num_box, 3))
        return rel_pred_scores.cpu().numpy()

    def caption_generation(self, img, bboxes, startswith=None):
        num_box = bboxes.shape[0]

        features, spatials, image_mask = self._prepare_visual_input(img, bboxes)
        text, text_mask, text_seg_ids = self._initialize_sent_for_captioning(num_box, startswith)
        co_attention_mask = torch.zeros((1, self.num_regions, self.expr_max_length)).cuda()
        # TODO: here 21 represents the captioning for refcoco. Other datasets will be supported later
        task_tokens = torch.from_numpy(np.array([21])).cuda().unsqueeze(0)

        stop_token = self.tokenizer.sep_token_id
        tokens = []
        ref_id = (torch.arange(num_box).cuda().long() + 1)

        for i, r_id in enumerate(ref_id):

            cap = text[i].unsqueeze(0)
            t_mask = text_mask[i].unsqueeze(0)
            t_seg_ids = text_seg_ids[i].unsqueeze(0)
            if SENTENCE_COMPLETION:
                current_position = t_mask.sum()
            else:
                current_position = 1
            pred_token = -1

            while (pred_token != stop_token):
                region_type_id = self._ref_id_to_type_id(r_id, self.num_regions)
                with torch.no_grad():
                    _, _, _, _, _, _, _, linguisic_prediction, _, _ = self.model(
                        cap, features, spatials, t_seg_ids, region_type_id, t_mask, image_mask, co_attention_mask,
                        task_tokens,
                        output_all_attention_masks=True
                    )
                _, pred_token = torch.max(linguisic_prediction, dim=-1)
                pred_token = pred_token.item()
                cap[:, current_position] = pred_token
                t_mask[:, current_position] = 1
                current_position += 1
            cap_len = (cap > 0).sum().item()
            tokens.append(self.tokenizer.decode(cap[0][:cap_len].cpu().tolist()))

        return tokens

    def grounding(self, img, bboxes, expr):
        features, spatials, image_mask = self._prepare_visual_input(img, bboxes)
        text, text_mask, text_seg_ids = self._prepare_txt_input(expr)
        co_attention_mask = torch.zeros((1, self.num_regions, self.expr_max_length)).cuda()
        # TODO: here 9 represents the grounding for refcoco. Other datasets will be supported later
        task_tokens = torch.from_numpy(np.array([9])).cuda().unsqueeze(0)

        with torch.no_grad():
            _, _, _, _, _, _, vision_logit, _, _, _ = self.model(
                text, features, spatials, text_seg_ids, None, text_mask, image_mask, co_attention_mask,
                task_tokens,
                output_all_attention_masks=True # allow us to visualize the attention across regions and word
            )

        num_box = bboxes.shape[0]
        grounding_val = vision_logit[0][1:].view(-1)

        im_height, im_width = img.shape[:2]
        bboxes /= np.array([[im_width, im_height, im_width, im_height]])
        overlaps = iou(torch.tensor(bboxes).float().cuda(), spatials[0][1:, :4].float())
        maps = torch.argmax(overlaps, 1)
        print(torch.max(maps))
        print(grounding_val.shape)
        return grounding_val[maps].cpu().numpy()

    def _init_args(self):
        args = SimpleNamespace(
            from_pretrained="../vilbert/save/multi_task_model.bin",
            bert_model="bert-base-uncased",
            config_file="../vilbert/config/bert_base_6layer_6conect.json",
            max_seq_length=101,
            train_batch_size=1,
            do_lower_case=True,
            predict_feature=False,
            seed=42,
            num_workers=0,
            baseline=False,
            img_weight=1,
            distributed=False,
            objective=1,
            visual_target=0,
            dynamic_attention=False,
            task_specific_tokens=True,
            tasks='1',
            save_name='',
            in_memory=False,
            batch_size=1,
            local_rank=-1,
            split='mteval',
            clean_train_sets=True
            )
        return args

    def _init_models(self):
        with open('../vilbert/invigorate_tasks.yml', 'r') as f:
            task_cfg = edict(yaml.safe_load(f))
        task_names = []
        for i, task_id in enumerate(self.args.tasks.split('-')):
            task = 'TASK' + task_id
            name = task_cfg[task]['name']
            task_names.append(name)

        config = BertConfig.from_json_file(self.args.config_file)
        default_gpu = True
        if self.args.predict_feature:
            config.v_target_size = 2048
            config.predict_feature = True
        else:
            config.v_target_size = 1601
            config.predict_feature = False

        if self.args.task_specific_tokens:
            config.task_specific_tokens = True

        if self.args.dynamic_attention:
            config.dynamic_attention = True

        config.visualization = True

        # initialize INVIGORATE models
        model = VILBertForVLTasks.from_pretrained(self.args.from_pretrained, config=config, default_gpu=default_gpu)
        model.eval()
        cuda = torch.cuda.is_available()
        if cuda: model = model.cuda()
        return model

    def _init_obj_cls(self):
        classes = ('__background__',  # always index 0
                        'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
                        'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
                        'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
                        'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
                        'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch')

        color_pool = [
            (255, 0, 0), (255, 102, 0), (255, 153, 0), (255, 204, 0), (255, 255, 0), (204, 255, 0), (153, 255, 0),
            (0, 255, 51),
            (0, 255, 153), (0, 255, 204), (0, 255, 255), (0, 204, 255), (0, 153, 255), (0, 102, 255), (102, 0, 255),
            (153, 0, 255),
            (204, 0, 255), (255, 0, 204), (187, 68, 68), (187, 116, 68), (187, 140, 68), (187, 163, 68), (187, 187, 68),
            (163, 187, 68), (140, 187, 68), (68, 187, 92), (68, 187, 140), (68, 187, 163), (68, 187, 187),
            (68, 163, 187),
            (68, 140, 187), (68, 116, 187), (116, 68, 187), (140, 68, 187), (163, 68, 187), (187, 68, 163),
            (255, 119, 119),
            (255, 207, 136), (119, 255, 146), (153, 214, 255)]
        np.random.shuffle(color_pool)
        color_dict = {}
        for i, clsname in enumerate(classes):
            color_dict[clsname] = color_pool[i]

        return classes, color_dict

    def _prepare_txt_input(self, query):
        tokens = self.tokenizer.encode(query)
        tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        if len(tokens) < self.expr_max_length:
            # Note here we pad in front of the sentence
            padding = [0] * (self.expr_max_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        text = torch.from_numpy(np.array(tokens)).cuda().unsqueeze(0)
        input_mask = torch.from_numpy(np.array(input_mask)).cuda().unsqueeze(0)
        segment_ids = torch.from_numpy(np.array(segment_ids)).cuda().unsqueeze(0)
        return text, input_mask, segment_ids

    def _prepare_visual_input(self, img, bboxes):

        features, infos = self.extract_features(img, bboxes)
        num_image = len(infos)

        feature_list = []
        image_location_list = []
        image_mask_list = []
        for i in range(num_image):
            image_w = infos[i]['image_width']
            image_h = infos[i]['image_height']
            feature = features[i]
            num_boxes = feature.shape[0]

            g_feat = torch.sum(feature, dim=0) / num_boxes
            num_boxes = num_boxes + 1
            feature = torch.cat([g_feat.view(1, -1), feature], dim=0)
            boxes = infos[i]['bbox']
            image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
            image_location[:, :4] = boxes
            image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                    image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))
            image_location[:, 0] = image_location[:, 0] / float(image_w)
            image_location[:, 1] = image_location[:, 1] / float(image_h)
            image_location[:, 2] = image_location[:, 2] / float(image_w)
            image_location[:, 3] = image_location[:, 3] / float(image_h)
            g_location = np.array([0, 0, 1, 1, 1])
            image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
            image_mask = [1] * (int(num_boxes))

            feature_list.append(feature)
            image_location_list.append(torch.tensor(image_location))
            image_mask_list.append(torch.tensor(image_mask))

        features = torch.stack(feature_list, dim=0).float().cuda()
        spatials = torch.stack(image_location_list, dim=0).float().cuda()
        image_mask = torch.stack(image_mask_list, dim=0).byte().cuda()

        return features, spatials, image_mask

    def _initialize_sent_for_captioning(self, num_box, startswith=None):
        # initialize sentences for all boxes

        text = torch.zeros(self.expr_max_length).long().cuda().unsqueeze(0)
        text_mask = torch.zeros(self.expr_max_length).long().cuda().unsqueeze(0)
        text_seg_ids = torch.zeros(self.expr_max_length).long().cuda().unsqueeze(0)

        start_token = self.tokenizer.cls_token_id
        if startswith is not None:
            assert len(startswith) == num_box
            for i in range(num_box):
                cls = startswith[i]
                is_token = self.tokenizer.encode("is")
                cls_token = self.tokenizer.encode(cls)
                cls_token_len = len(cls_token)
                cap = text[0:1].clone()
                cap[0, 1:] = 0
                cap[0, 1: (cls_token_len + 1)] = torch.tensor(cls_token).type_as(cap)
                # cap[0, (cls_token_len + 1)] = is_token[0]
                mask = text_mask[0:1].clone()
                mask[0, :(cls_token_len + 1)] = 1
                mask[0, (cls_token_len + 1):] = 0
                text = torch.cat([text, cap], dim=0)
                text_mask = torch.cat([text_mask, mask], dim=0)
                text_seg_ids = torch.cat([text_seg_ids, text_seg_ids.clone()], dim=0)
        else:
            for i in range(num_box):
                cap = text[0:1].clone()
                cap[:, 0] = start_token
                cap[:, 1:] = 0
                text = torch.cat([text, cap], dim=0)
                mask = text_mask[0:1].clone()
                mask[:, 0] = 1
                mask[:, 1:] = 0
                text_mask = torch.cat([text_mask, mask], dim=0)
                text_seg_ids = torch.cat([text_seg_ids, text_seg_ids.clone()], dim=0)

        return text[1:], text_mask[1:], text_seg_ids[1:]

    def _initialize_sent_for_vmr_det(self):
        # initialize dummy sentences for vmr det.

        text = torch.zeros(self.expr_max_length).long().cuda().unsqueeze(0)
        text_mask = torch.zeros(self.expr_max_length).long().cuda().unsqueeze(0)
        text_seg_ids = torch.zeros(self.expr_max_length).long().cuda().unsqueeze(0)

        start_token = self.tokenizer.cls_token_id
        stop_token = self.tokenizer.sep_token_id

        text[:, 0] = start_token
        text[:, 1] = stop_token
        text_mask[:, :2] = 1

        return text, text_mask, text_seg_ids

    def _ref_id_to_type_id(self, r_id, num_region):
        typeid = torch.zeros((1, num_region)).type_as(r_id)
        typeid[0, 0] = 3
        typeid[0, r_id] = 1
        return typeid

# for testing this code
class VilbertReq():
    def __init__(self):
        pass

if __name__=="__main__":

    # ovs = iou(torch.tensor(infos[0]["bbox"][:num_box]).float(),
    #                   torch.tensor(bboxes[:, :4]).float())
    # input_to_det = torch.argmax(ovs, dim=1).tolist()
    #
    # infos[0]["ref_classes"] = [class_names[input_to_det[i]] for i in range(num_box)]
    # features[0] = features[0][:infos[0]["num_boxes"]]

    # image_path = '../images/nus_validation/11.jpeg'
    # img = PIL.Image.open(image_path).convert('RGB')
    # img = torch.tensor(np.array(img))
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()
    # query = "the white mouse"

    rospy.init_node('vilbert_server')
    vilbert_server = VilbertServer()
    rospy.spin()

    # obj_detector = VMRDObjDetector()
    # bboxes, classes, class_scores, num_box, class_names = obj_detector.detect_objs(cv2.imread(image_path))

    # req = VilbertReq()
    # req.img = cv2.imread(image_path)
    # req.bbox = bboxes[:, :4].reshape(-1).tolist()
    # req.expr = query
    # res = vilbert_server.vmrn_server_callback(req)
    # res = vilbert_server.vilbert_captioning_server_callback(req)
    # res = vilbert_server.vilbert_grounding_server_callback(req)
