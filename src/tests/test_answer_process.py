import sys
import os.path as osp
this_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(this_dir, '../'))
py_version = sys.version.split(".")[0]
if py_version == "3":
    raw_input = input

import nltk
from config.config import *
import warnings

try:
    import stanza
    stanford_nlp_server = stanza.Pipeline("en")
except:
    warnings.warn("stanza needs python 3.6 or higher. "
                  "please update your python version and run 'pip install stanza'")

def postag_analysis(sent, mode="nltk"):
    if mode == "nltk":
        text = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(text)
    elif mode == "stanza":
        doc = stanford_nlp_server(sent)
        pos_tags = [(d.text, d.xpos) for d in doc.sentences[0].words]
    else:
        raise NotImplementedError
    return pos_tags

def _find_subject(expr):
    pos_tags = postag_analysis(expr)

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
        cls_str = ''.join(cls.split())
        if cls_str in subj_str or subj_str in cls_str:
            cls_filter.append(cls)
    assert len(cls_filter) <= 1
    return cls_filter

def _process_user_answer(answer, subject):
    answer = answer.lower()

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

mode = "nltk"
expr = "the remote"
print(postag_analysis(expr, mode))
subject = _find_subject(expr)
while(True):
    answer = raw_input("input a sentence: ")
    response, answer = _process_user_answer(answer, subject)
    print(answer)
    print(postag_analysis(answer, mode))

# subject = _find_subject(expr)
# answer = "yes"
# print(_process_user_answer(answer, subject))
