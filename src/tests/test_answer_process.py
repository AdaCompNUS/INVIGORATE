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

    # 1. Try to find the first noun phrase before any preposition
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
        elif postag in {"IN", "TO", "RP"}:
            break

    # 2. Otherwise, return all words before the first prepostion
    assert subj_tokens == []
    for i, (token, postag) in enumerate(pos_tags):
        if postag in {"IN", "TO", "RP"}:
            break
        if postag in {"DT"}:
            continue
        subj_tokens.append(token)

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

def _process_user_answer(answer, subject_tokens):
    # preprocess the sentence
    # 1. make all letters lowercase
    answer = answer.lower()
    # 2. delete all ",", "." and "!" in the answer
    answer = answer.replace(",", " ")
    answer = answer.replace(".", " ")
    answer = answer.replace("!", " ")
    # 3. delete redundant spaces
    answer = ' '.join(answer.split()).strip().split(' ')

    # extract the response utterence
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

    # postprocess the sentence
    subject = " ".join(subject_tokens)
    # replace the pronoun in the answer with the subject given by the user
    for pronoun in PRONOUNS:
        if pronoun in answer:
            answer = [w if w != pronoun else subject for w in answer]

    answer = ' '.join(answer)

    # if the answer starts without any subject, add the subject
    subj_cand = []
    for token, postag in postag_analysis(answer):
        if postag in {"IN", "TO", "RP"}:
            break
        if postag in {"DT"}:
            continue
        subj_cand.append(token)
    subj_cand = set(subj_cand)
    if len(subj_cand.intersection(set(subject_tokens))) == 0:
        answer = " ".join(subject_tokens + answer.split(" "))

    return response, answer

mode = "nltk"
expr = raw_input("input an expression: ")
print(postag_analysis(expr, mode))
subject = _find_subject(expr)
print("Parsed subject: {}".format(" ".join(subject)))
while(True):
    answer = raw_input("input an answer: ")
    response, answer = _process_user_answer(answer, subject)
    print("Processed Answer: {}".format(answer))
    # print(postag_analysis(answer, mode))

# subject = _find_subject(expr)
# answer = "yes"
# print(_process_user_answer(answer, subject))
