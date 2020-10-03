
import rospy

from ingress_srv.ingress_srv import Ingress
from config.config import CLASSES

def caption_generation_client(img, bbox, target_box_id):
    # dbg_print(bbox)
    ingress_client = Ingress()
    top_caption, top_context_box_idx = ingress_client.generate_rel_captions_for_box(img, bbox.tolist(), target_box_id)
    return top_caption, top_context_box_idx

def form_rel_caption_sentence(obj_cls, cxt_obj_cls, rel_caption):
    obj_name = CLASSES[int(obj_cls)]
    cxt_obj_name = CLASSES[int(cxt_obj_cls)]

    if cxt_obj_cls == 0:
        rel_caption_sentence = '{} {}'.format(obj_name, rel_caption)
    else:
        rel_caption_sentence = '{} {} of {}'.format(obj_name, rel_caption, cxt_obj_name)
    rel_caption_sentence = rel_caption_sentence.replace('.', '')
    return rel_caption_sentence

def generate_caption(img_cv, bboxes, classes, target_box_ind):
    # input validity check
    if bboxes.shape[1] == 5:
        # filter out class scores if necessary
        bboxes = bboxes[:, :4]

    print('generating caption for object {}'.format(target_box_ind))
    top_caption, top_context_box_idxs = caption_generation_client(img_cv, bboxes, target_box_ind)
    print('top_caption: {}'.format(top_caption))
    print('top_context_box_idxs: {}'.format(top_context_box_idxs))
    if top_context_box_idxs == len(bboxes):
        # top context is background
        caption_sentence = form_rel_caption_sentence(classes[target_box_ind], 0, top_caption)
    else:
        caption_sentence = form_rel_caption_sentence(classes[target_box_ind], classes[top_context_box_idxs], top_caption)
    print('caption_sentence: {}'.format(caption_sentence))
    return caption_sentence