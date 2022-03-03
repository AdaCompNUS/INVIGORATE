import sys
import os.path as osp

this_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(this_dir, '../'))
py_version = sys.version.split(".")[0]
if py_version == "3":
    raw_input = input

import warnings
from libraries.utils import ExprssionProcessor

expr_proc = ExprssionProcessor('nltk')

def test_complete_expr(subject=''):
    while(True):
        expr = raw_input("input an expression: ")
        if not subject:
            subject_tokens = expr_proc.find_subject(expr)
        else:
            subject_tokens = subject.split()
        print(expr_proc.complete_expression(expr, subject_tokens))

def test_find_subject():
    while (True):
        expr = raw_input("input an expression: ")
        print(expr_proc.find_subject(expr, expr_proc.CLASSES))

def test_complete_answer():
    expr = raw_input("input an expression: ")
    print(expr_proc.postag_analysis(expr))
    subject = expr_proc.find_subject(expr)
    print("Parsed subject: {}".format(" ".join(subject)))
    while(True):
        answer = raw_input("input an answer: ")
        response, answer = expr_proc.process_user_answer(answer, subject)
        print("Processed Answer: {}".format(answer))

def test_merge_expressions():
    sent1 = 'the phone on the right'
    sent2 = 'the red smart phone'
    subject_tokens = expr_proc.find_subject(sent1)
    print(expr_proc.merge_expressions(sent1, sent2, subject_tokens))

def test_semantic_similarity():
    subject = "remote"
    e1 = "the big red {:s} on the right".format(subject, subject)
    exprs = ["the {:s} on the left".format(subject),
             "the left {:s}".format(subject),
             "the leftmost {:s}".format(subject),
             "the red {:s} to the left".format(subject),
             "the red {:s} on top of the book".format(subject),
             "the red right {:s}".format(subject),
             "the red {:s} on the right".format(subject),
             "the red {:s}".format(subject),
             "the red {:s} on the right of the banana".format(subject),
             "the {:s}".format(subject),
             "the {:s} next to the banana".format(subject),
             "the second {:s}".format(subject),
             "the {:s} on top".format(subject),
             "the white {:s}".format(subject),
             "the black {:s}".format(subject),
             "the {:s} in white".format(subject),
             "the {:s} in black".format(subject),
             "the {:s} in red".format(subject)]
    print(expr_proc.semantic_similarity(e1, exprs, (0, 0.5, 0.5)))

test_merge_expressions()
# test_complete_expr()
#  test_find_subject()