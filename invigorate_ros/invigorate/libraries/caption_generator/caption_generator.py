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

def caption_generation_client(img, bbox, target_box_id):
    # dbg_print(bbox)
    ingress_client = Ingress()
    top_caption, top_context_box_idx = ingress_client.generate_rel_captions_for_box(img, bbox.tolist(), target_box_id)
    return top_caption, top_context_box_idx

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
        # HACK to fix the rare bug where the caption is actually semantic
        rel_caption_sentence = '{} {}'.format(rel_caption, self_caption)
    rel_caption_sentence = rel_caption_sentence.replace('.', '')
    return rel_caption_sentence

def generate_caption(img_cv, bboxes, classes, target_box_ind, subject):
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
        caption_sentence, _ = form_rel_caption_sentence(classes[target_box_ind], 0, top_caption, subject)
    else:
        caption_sentence, _ = form_rel_caption_sentence(classes[target_box_ind], classes[top_context_box_idxs], top_caption, subject)
    print('caption_sentence: {}'.format(caption_sentence))
    return caption_sentence

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

def form_self_referential_questions(classes, dense_caps, subject, nlp_server="nltk"):
    for i in range(len(dense_caps)):
        sent = dense_caps[i]
        if nlp_server == "nltk":
            text = nltk.word_tokenize(sent)
            pos_tags = nltk.pos_tag(text)
        elif nlp_server == "stanza":
            doc = stanford_nlp_server(sent)
            pos_tags = [(d.text, d.xpos) for d in doc.sentences[0].words]
        else:
            raise NotImplementedError

        formed_cap = ['the']
        for token, postag in pos_tags:
            if postag in {'JJ'}:
                formed_cap.append(token)
        formed_cap.append(CLASSES[int(classes[i])])

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

    # return zip(list(formed_dense_caps), list(formed_rel_caps), list(formed_mixes_caps))
    return zip(list(formed_dense_caps), list(formed_rel_caps))