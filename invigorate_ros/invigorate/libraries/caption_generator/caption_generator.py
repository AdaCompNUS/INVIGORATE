import numpy as np
import rospy
import nltk
import warnings
try:
    import stanza
    stanford_nlp_server = stanza.Pipeline("en")
except:
    warnings.warn("No NLP models are loaded.")

from ingress_srv.ingress_srv import Ingress
from invigorate.config.config import CLASSES
from ..utils.expr_processor import ExprssionProcessor
from collections_extended import setlist

expr_proc = ExprssionProcessor()
# all class words
CLASS_WORD_BAG = set([])
for c in expr_proc.CLASSES:
    CLASS_WORD_BAG = CLASS_WORD_BAG.union(c.split(' '))
for syn in expr_proc.SYNSETS_WORD_BAG:
    CLASS_WORD_BAG = CLASS_WORD_BAG.union(syn)

# RSS Version - purely relational captions

def caption_generation_client(img, bbox, target_box_id, cap_type='rel'):
    # dbg_print(bbox)
    ingress_client = Ingress()
    if cap_type == 'rel':
        top_caption, top_context_box_idx = \
            ingress_client.generate_rel_captions_for_box(img, bbox.tolist(), target_box_id)
        return top_caption, top_context_box_idx
    elif cap_type == 'self':
        dense_caps, _, _, _ = \
            ingress_client.generate_all_captions_for_boxes(img, bbox.tolist())
        return dense_caps[target_box_id], None
    else:
        raise NotImplementedError

def form_rel_caption_sentence_rss(obj_cls, cxt_obj_cls, rel_caption, subject):
    obj_name = subject
    cxt_obj_name = CLASSES[int(cxt_obj_cls)]

    if rel_caption.startswith("at") or rel_caption.startswith("on") or rel_caption.startswith("in"):
        # if cxt_obj_cls == 0:
        #     rel_caption_sentence = '{} {}'.format(obj_name, rel_caption)
        # else:
        #     rel_caption_sentence = '{} {} of {}'.format(obj_name, rel_caption, cxt_obj_name)
        rel_caption_sentence = 'the {} {}'.format(obj_name, rel_caption)
    else:
        # HACK to fix the rare bug where the caption is actually semantic
        rel_caption_sentence = 'the {} {}'.format(rel_caption, obj_name)
    rel_caption_sentence = rel_caption_sentence.replace('.', '')
    return rel_caption_sentence

def generate_caption(img_cv, bboxes, classes, target_box_ind, subject, cap_type='rel'):
    # input validity check
    if bboxes.shape[1] == 5:
        # filter out class scores if necessary
        bboxes = bboxes[:, :4]

    print('generating caption for object {}'.format(target_box_ind))
    top_caption, top_context_box_idxs = caption_generation_client(
        img_cv, bboxes, target_box_ind, cap_type=cap_type)
    print('top_caption: {}'.format(top_caption))
    if cap_type == 'rel':
        print('top_context_box_idxs: {}'.format(top_context_box_idxs))
        if top_context_box_idxs == len(bboxes):
            # top context is background
            caption_sentence = form_rel_caption_sentence_rss(
                classes[target_box_ind], 0, top_caption, subject)
        else:
            caption_sentence = form_rel_caption_sentence_rss(
                classes[target_box_ind], classes[top_context_box_idxs], top_caption, subject)
        print('caption_sentence: {}'.format(caption_sentence))
    elif cap_type == 'self':
        caption_sentence = form_self_referential_questions(
            classes[target_box_ind], top_caption, None)[0]
    else:
        raise NotImplementedError
    return caption_sentence

# IJRR Version

def form_rel_caption_sentence(obj_cls, cxt_obj_cls, rel_caption, subject):
    obj_name = CLASSES[int(obj_cls)]
    cxt_obj_name = CLASSES[int(cxt_obj_cls)]

    if rel_caption.startswith("at") or rel_caption.startswith("on") or rel_caption.startswith("in"):
        # if cxt_obj_cls == 0 or cxt_obj_cls == obj_cls:
        #     rel_caption_sentence = 'the {} {}'.format(obj_name, rel_caption)
        # else:
        #     rel_caption_sentence = 'the {} {} of the {}'.format(obj_name, rel_caption, cxt_obj_name)
        rel_caption_sentence = 'the {} {}'.format(obj_name, rel_caption)
        q_flag = True
    else:
        rel_caption = expr_proc.clean_sentence(rel_caption)
        rel_caption = rel_caption.split(' ')
        rel_caption = list(setlist(rel_caption))
        rel_caption_pre = []
        for w in rel_caption:
            if w not in CLASS_WORD_BAG:
                rel_caption_pre.append(w)
        rel_caption = ' '.join(rel_caption_pre)
        # HACK to fix the rare bug where the caption is actually semantic
        rel_caption_sentence = 'the {} {}'.format(rel_caption, obj_name)
        q_flag = False
    rel_caption_sentence = rel_caption_sentence.replace('.', '')
    return rel_caption_sentence, q_flag

def form_mixed_caption_sentence(obj_cls, cxt_obj_cls, rel_caption, subject, self_caption):
    obj_name = self_caption
    cxt_obj_name = CLASSES[int(cxt_obj_cls)]

    if rel_caption.startswith("at") or rel_caption.startswith("on") or rel_caption.startswith("in"):
        # if cxt_obj_cls == 0 or cxt_obj_cls == obj_cls:
        #     rel_caption_sentence = 'the {} {}'.format(self_caption, rel_caption)
        # else:
        #     rel_caption_sentence = '{} {} of the {}'.format(self_caption, rel_caption, cxt_obj_name)
        rel_caption_sentence = '{} {}'.format(self_caption, rel_caption)
    else:
        rel_caption = expr_proc.clean_sentence(rel_caption)
        rel_caption = rel_caption.split(' ')
        rel_caption = list(setlist(rel_caption))
        rel_caption_pre = []
        for w in rel_caption:
            if w not in CLASS_WORD_BAG:
                rel_caption_pre.append(w)
        rel_caption = ' '.join(rel_caption_pre)
        # HACK to fix the rare bug where the caption is actually semantic
        rel_caption_sentence = '{} {}'.format(rel_caption, self_caption)
    rel_caption_sentence = rel_caption_sentence.replace('.', '')
    return rel_caption_sentence

def all_captions_generation_client(img, bbox):
    # dbg_print(bbox)
    ingress_client = Ingress()

    dense_caps, rel_caps, context_idxs, rel_cap_probs = \
        ingress_client.generate_all_captions_for_boxes(img, bbox.tolist())

    selected_rel_caps = []
    selected_context_idxs = []
    for rcaps, ctxs, probs in zip(rel_caps, context_idxs, rel_cap_probs):
        cap_idx = np.argmax(probs)
        selected_rel_caps.append(rcaps[cap_idx].strip())
        selected_context_idxs.append(ctxs[cap_idx])

    return dense_caps, selected_rel_caps, selected_context_idxs

def form_self_referential_questions(classes, dense_caps, subject):
    if isinstance(dense_caps, str):
        assert isinstance(classes, int)
        dense_caps = [dense_caps]
        classes = [int(classes)]

    for i in range(len(dense_caps)):
        sent = dense_caps[i]
        text = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(text)

        formed_cap = ['the']
        for token, postag in pos_tags:
            if postag in {'JJ'} and token not in {'remote'}:
                formed_cap.append(token)
        formed_cap.append(CLASSES[int(classes[i])])
        formed_cap = list(setlist(formed_cap))
        dense_caps[i] = ' '.join(formed_cap)

    return dense_caps


def generate_all_captions(img_cv, bboxes, classes, subject):
    # input validity check
    if bboxes.shape[1] == 5:
        # filter out class scores if necessary
        bboxes = bboxes[:, :4]

    dense_caps, rel_caps, context_idxs = all_captions_generation_client(img_cv, bboxes)
    print('generated self-referential captions: {}'.format(dense_caps))
    print('generated relational captions: {}'.format(rel_caps))
    print('context box indices'.format(context_idxs))

    formed_dense_caps = form_self_referential_questions(classes, dense_caps, subject)
    formed_rel_caps = []
    rel_cap_flags = []
    for ind in range(len(bboxes)):
        if context_idxs[ind] == len(bboxes):
            # top context is background
            caption_sentence, q_flag = form_rel_caption_sentence(classes[ind], 0, rel_caps[ind], subject)
        else:
            caption_sentence, q_flag = form_rel_caption_sentence(classes[ind], classes[context_idxs[ind]],
                                                         rel_caps[ind], subject)
        formed_rel_caps.append(caption_sentence)
        rel_cap_flags.append(q_flag)

    formed_mixes_caps = []
    for ind in range(len(bboxes)):
        if not rel_cap_flags[ind]:
            formed_mixes_caps.append(None)
        else:
            if context_idxs[ind] == len(bboxes):
                # top context is background
                caption_sentence = form_mixed_caption_sentence(classes[ind], 0, rel_caps[ind], subject, formed_dense_caps[ind])
            else:
                caption_sentence = form_mixed_caption_sentence(classes[ind], classes[context_idxs[ind]],
                                                             rel_caps[ind], subject, formed_dense_caps[ind])
            formed_mixes_caps.append(caption_sentence)

    print('formed self-referential captions: {}'.format(formed_dense_caps))
    print('formed relational captions: {}'.format(formed_rel_caps))
    print('formed mixed captions: {}'.format(formed_mixes_caps))

    return zip(list(formed_dense_caps), list(formed_rel_caps), list(formed_mixes_caps))
    # return zip(list(formed_dense_caps), list(formed_rel_caps))