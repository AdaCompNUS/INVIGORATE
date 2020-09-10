
import rospy

from ingress_srv.ingress_srv import Ingress

CLASSES = ['__background__',  # always index 0
                'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
                'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
                'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
                'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
                'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']

def caption_generation_client(img, bbox, target_box_id):
    # dbg_print(bbox)
    ingress_client = Ingress()
    top_caption, top_context_box_idx = ingress_client.generate_rel_captions_for_box(img, bbox.tolist(), target_box_id)
    return top_caption, top_context_box_idx

def form_rel_caption_sentence(obj_cls, cxt_obj_cls, rel_caption):
    obj_name = CLASSES[int(obj_cls)]
    cxt_obj_name = CLASSES[int(cxt_obj_cls)]
    
    if cxt_obj_cls == 0:
        rel_caption_sentence = '{} {} (of image)'.format(obj_name, rel_caption)
    else:
        rel_caption_sentence = '{} {} of {}'.format(obj_name, rel_caption, cxt_obj_name)
    rel_caption_sentence = rel_caption_sentence.replace('.', '')
    return rel_caption_sentence

def generate_caption(img_cv, bboxes, target_box_ind):
    bbox_2d = bboxes[:, :-1]
    classes = bboxes[:, -1]
    print(classes)

    print('generating caption for object {}'.format(target_box_ind))
    top_caption, top_context_box_idxs = caption_generation_client(img_cv, bbox_2d, target_box_ind)
    print('top_caption: {}'.format(top_caption))
    print('top_context_box_idxs: {}'.format(top_context_box_idxs))
    if top_context_box_idxs == len(bbox_2d) - 1:
        # top context is background
        caption_sentence = form_rel_caption_sentence(classes[target_box_ind], 0, top_caption)
    else:
        caption_sentence = form_rel_caption_sentence(classes[target_box_ind], classes[top_context_box_idxs], top_caption)
    print('caption_sentence: {}'.format(caption_sentence))
    return caption_sentence