import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(this_dir, '../'))

import nltk
from config.config import *

def _find_subject(expr):
    text = nltk.word_tokenize(expr)
    pos_tags = nltk.pos_tag(text)

    subj_tokens = []
    for i, (token, postag) in enumerate(pos_tags):
        if postag in {"NN"}:
            subj_tokens.append(token)
            for j in range(i + 1, len(pos_tags)):
                token, postag = pos_tags[j]
                if postag in {"NN"}:
                    subj_tokens.append(token)
                else:
                    break
            return subj_tokens

    return subj_tokens

def _initialize_cls_filter(subject):
    subj_str = ''.join(subject)
    cls_filter = []
    for cls in CLASSES:
        if cls in subj_str:
            cls_filter.append(cls)
    assert len(cls_filter) <= 1
    return cls_filter

def _process_user_answer(answer, subject):
    answer = answer.lower()

    is_subject_informative = len(_initialize_cls_filter(subject)) > 0
    if is_subject_informative:
        subject = " ".join(subject)
        # replace the pronoun in the answer with the subject given by the user
        for pronoun in PRONOUNS:
            if pronoun in answer:
                answer = answer.replace(pronoun, subject)

    answer = answer.replace(",", " ")  # delete all , in the answer
    answer = answer.replace(".", " ")  # delete all . in the answer
    answer = answer.replace("!", " ")  # delete all . in the answer
    answer = ' '.join(answer.split()).strip().split(' ')

    response = None
    for neg_ans in NEGATIVE_ANS:
        if neg_ans in answer:
            response = False
            answer.remove(neg_ans)

    for pos_ans in POSITIVE_ANS:
        if pos_ans in answer:
            assert response is None, "A positive answer should not appear with a negative answer"
            response = True
            answer.remove(pos_ans)

    answer = ' '.join(answer)

    return response, answer

expr = "the apple"
subject = _find_subject(expr)
answer = "yes"
print(_process_user_answer(answer, subject))
