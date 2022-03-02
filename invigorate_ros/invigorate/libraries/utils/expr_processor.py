import collections
import warnings
# try:
#     import stanza
# except:
#     warnings.warn("Stanford NLP Server is not available. NLTK will be used")
import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.metrics import edit_distance
import numpy as np
import re

# TODO: here it is possible to introduce state of the art NN-based NL processor, e.g., BERT.

class Lesk:
    # from https://www.kaggle.com/antriksh5235/semantic-similarity-using-wordnet/notebook
    def __init__(self, sentence):
        self.sentence = sentence
        self.meanings = {}
        for word in sentence:
            self.meanings[word] = ''

    def getSenses(self, word):
        # print word
        return wn.synsets(word.lower())

    def getGloss(self, senses):

        gloss = {}

        for sense in senses:
            gloss[sense.name()] = []

        for sense in senses:
            gloss[sense.name()] += word_tokenize(sense.definition())

        return gloss

    def getAll(self, word):
        senses = self.getSenses(word)

        if senses == []:
            return {word.lower(): senses}

        return self.getGloss(senses)

    def Score(self, set1, set2):
        # Base
        overlap = 0

        # Step
        for word in set1:
            if word in set2:
                overlap += 1

        return overlap

    def overlapScore(self, word1, word2):

        gloss_set1 = self.getAll(word1)
        if self.meanings[word2] == '':
            gloss_set2 = self.getAll(word2)
        else:
            # print 'here'
            gloss_set2 = self.getGloss([wn.synset(self.meanings[word2])])

        # print gloss_set2

        score = {}
        for i in gloss_set1.keys():
            score[i] = 0
            for j in gloss_set2.keys():
                score[i] += self.Score(gloss_set1[i], gloss_set2[j])

        bestSense = None
        max_score = 0
        for i in gloss_set1.keys():
            if score[i] > max_score:
                max_score = score[i]
                bestSense = i

        return bestSense, max_score

    def lesk(self, word, sentence):
        maxOverlap = 0
        context = sentence
        word_sense = []
        meaning = {}

        senses = self.getSenses(word)

        for sense in senses:
            meaning[sense.name()] = 0

        for word_context in context:
            if not word == word_context:
                score = self.overlapScore(word, word_context)
                if score[0] == None:
                    continue
                meaning[score[0]] += score[1]

        if senses == []:
            return word, None, None

        self.meanings[word] = max(meaning.keys(), key=lambda x: meaning[x])

        return word, self.meanings[word], wn.synset(self.meanings[word]).definition()

class ExprssionProcessor:

    def __init__(self,
                 nlp_server='nltk'):
        self.nlp_server = nlp_server
        self.stemmer = PorterStemmer()

        # TODO: implement a version based on the Stanford NLP Toolkit.
        # if self.nlp_server == "stanza":
        #     try:
        #         self.stanford_nlp_server = stanza.Pipeline("en")
        #     except:
        #         warnings.warn("stanza needs python 3.6 or higher. "
        #                       "please update your python version "
        #                       "and run 'pip install stanza'")

        self.PRONOUNS = {"it", "it's", "one", "that", "that's"}
        self.POSITIVE_ANS = {"yes", "yeah", "yep", "sure", "certainly", "OK"}
        self.NEGATIVE_ANS = {"no", "nope", "nah"}
        self.STOP_WORDS = nltk.corpus.stopwords.words()
        self.CLASSES = ['__background__','sports ball', 'bottle', 'cup', 
                        'knife', 'banana', 'apple', 'carrot', 'mouse', 
                        'remote', 'cell phone', 'book', 'scissors', 
                        'teddy bear', 'toothbrush', 'box', ]

    # ------------- expression preprocess -------------
    def stem(self, tag_q):
        stem_q = []
        for token in tag_q:
            stem_q.append(self.stemmer.stem(token))
        return stem_q

    def extract_cls_filter(self, subject):
        subj_str = ''.join(subject)
        cls_filter = []
        for cls in self.CLASSES:
            cls_str = ''.join(cls.split())
            if cls_str in subj_str or subj_str in cls_str:
                cls_filter.append(cls)
        assert len(cls_filter) <= 1
        return cls_filter

    def postag_analysis(self, expr):
        text = word_tokenize(expr)
        pos_tags = pos_tag(text)
        return pos_tags

    def find_subject(self, expr, classes=None):
        expr = self._clean_sentence(expr, clean_stop=False)
        pos_tags = self.postag_analysis(expr)

        subj_tokens = []

        # 1. Try to find the first noun phrase before any preposition
        for i, (token, postag) in enumerate(pos_tags):
            if postag in {"NN"} and classes is not None:
                assert isinstance(classes, collections.Sequence)
                for c in classes:
                    if token in c:
                        subj_tokens.append(c)
                if subj_tokens:
                    return subj_tokens

            elif postag in {"NN"} and classes is None:
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

        # 2. Otherwise, return all words before the first preposition
        assert subj_tokens == []
        for i, (token, postag) in enumerate(pos_tags):
            if postag in {"IN", "TO", "RP"}:
                break
            if postag in {"DT"}:
                continue
            subj_tokens.append(token)

        return subj_tokens

    def is_included(self, expr, old_expr):
        expr = self._clean_sentence(expr)
        old_expr = self._clean_sentence(old_expr)
        return set(expr.split(' ')).issubset(set(old_expr.split(' ')))

    def process_user_answer(self, answer, subject_tokens):
        # preprocess the sentence
        answer = self._clean_sentence(answer, clean_stop=False)
        answer = answer.split(' ')

        # extract the response utterance
        response = None
        for neg_ans in self.NEGATIVE_ANS:
            if neg_ans in answer:
                response = False
                answer.remove(neg_ans)

        for pos_ans in self.POSITIVE_ANS:
            if pos_ans in answer:
                assert response is None, \
                    "A positive answer should not " \
                    "appear with a negative answer"
                response = True
                answer.remove(pos_ans)

        # postprocess the sentence
        answer = self.complete_answer_expression(' '.join(answer), subject_tokens)
        return response, answer

    def merge_expressions(self, expr, new_expr, subject_tokens):
        new_expr = self.complete_answer_expression(new_expr, subject_tokens)
        new_pre, _, new_post = self._split_expr_by_subject(new_expr, subject_tokens)
        old_pre, _, old_post = self._split_expr_by_subject(expr, subject_tokens)

        old_pre_clean = set(
            self._clean_sentence(old_pre).split(' '))
        new_pre_clean = set(
            self._clean_sentence(new_pre).split(' '))
        merged_pre = list(old_pre_clean.union(new_pre_clean))
        while '' in merged_pre:
            merged_pre.remove('')
        merged_pre = ' '.join(['the'] + merged_pre)

        merged_post = old_post
        merged_post_clean = set(
            self._clean_sentence(merged_post).split(' ')).union({''})
        new_post_clean = set(
            self._clean_sentence(new_post).split(' ')).union({''})
        if merged_post_clean.issubset(new_post_clean):
            merged_post = new_post
        elif new_post_clean.issubset(merged_post_clean):
            pass
        else:
            merged_post = ' '.join([merged_post, 'and', new_post])

        return ' '.join([merged_pre] + subject_tokens + [merged_post]).strip()

    def complete_answer_expression(self, answer, subject_tokens):
        # replace the pronoun in the answer with the subject given by the user
        answer = self._clean_sentence(answer, clean_stop=False)
        answer = answer.split()
        subject = ' '.join(subject_tokens)
        answer = [w if w not in self.PRONOUNS else subject for w in answer]
        answer = ' '.join(answer)

        # if the answer starts without any subject, add the subject
        subj_cand = []
        for token, postag in self.postag_analysis(answer):
            if postag in {"IN", "TO", "RP"}:
                break
            if postag in {"DT"}:
                continue
            subj_cand.append(token)
        subj_cand = set(subj_cand)
        if len(subj_cand.intersection(set(subject_tokens))) == 0 and len(answer) > 0:
            answer = " ".join(subject_tokens + answer.split(" "))
        return answer

    def _split_expr_by_subject(self, expr, subject_tokens):
        """
        input of this function should include the subject tokens,
        e.g., the expressions processed by _complete_answer_expression
        """
        if expr:
            for sub in subject_tokens:
                assert sub in expr, "the subject should be included" \
                                    "in the given expression. \n expression: " \
                                    "{:s} \n problematic sub: {:s}".format(
                    expr, sub
                )

        pre_phrase = []
        post_phrase = []
        subject_phrase = []

        expr_words = expr.split(' ')

        pre_flag = True
        for w in expr_words:
            if w in subject_tokens:
                subject_phrase.append(w)
                pre_flag = False
                continue

            if pre_flag:
                pre_phrase.append(w)
            else:
                post_phrase.append(w)

        return ' '.join(pre_phrase), ' '.join(subject_phrase), ' '.join(post_phrase)

    # ----------- word distance --------------
    def word_path_dist(self, set1, set2):
        return wn.path_similarity(set1, set2)

    def word_wup_dist(self, set1, set2):
        return wn.wup_similarity(set1, set2)

    def word_edit_dist(self, word1, word2):
        if float(edit_distance(word1, word2)) == 0.0:
            return 0.0
        return 1.0 / float(edit_distance(word1, word2))

    # ------------ sentence distance -------------
    def semantic_similarity(self, q1, q2, weight=(0, 0.5, 0.5)):

        if sum(weight) != 1.:
            warnings.warn('The sum of all semantic similarity weights is not equal to 1.')

        type_weights = dict(zip(('meteor', 'path', 'wup'), weight))

        def sentence_means(pos_tag):
            sentence = []
            for i, word in enumerate(pos_tag):
                if 'NN' in word[1] or 'JJ' in word[1] or \
                        'VB' in word[1] or 'IN' in word[1]:
                    sentence.append(word[0])

            sense = Lesk(sentence)
            means = []
            for word in sentence:
                means.append(sense.lesk(word, sentence))
            return means

        def preprocess_expr(expr, mode='meteor'):
            expr = self._clean_sentence(expr)
            if mode == 'meteor':
                return self.postag_analysis(expr)
            else:
                return sentence_means(self.postag_analysis(expr))

        sim_scores = []
        for s_type, w in type_weights.items():
            if w == 0.:
                continue
            q1_processed = preprocess_expr(q1, s_type)
            if isinstance(q2, collections.Sequence):
                q2_processed = [preprocess_expr(q, s_type) for q in q2]
            else:
                q2_processed = preprocess_expr(q2, s_type)

            dist_func = getattr(self, '_sent_' + s_type + '_dist')
            scores = np.array(dist_func(q1_processed, q2_processed))
            sim_scores.append(w * scores)

        sim_scores = np.array(sim_scores).sum(0)
        return sim_scores

    def _sent_meteor_dist(self, q1, q2):
        q1 = ' '.join([w[1] for w in q1])
        if not isinstance(q2, collections.Sequence):
            q2 = ' '.join([w[1] for w in q2])
            return single_meteor_score(q2, q1)
        else:
            q2 = [' '.join([w[1] for w in q]) for q in q2]
            return [single_meteor_score(q, q1) for q in q2]

    def _sent_path_dist(self, q1, q2):
        if isinstance(q2, collections.Sequence):
            return [self._overall_sim(
                q1, q, self._word_wise_sent_path_dist(q1, q)) for q in q2]
        else:
            return self._overall_sim(q1, q2, self._word_wise_sent_path_dist(q1, q2))

    def _sent_wup_dist(self, q1, q2):
        if isinstance(q2, collections.Sequence):
            return [self._overall_sim(
                q1, q, self._word_wise_sent_wup_dist(q1, q)) for q in q2]
        else:
            return self._overall_sim(q1, q2, self._word_wise_sent_wup_dist(q1, q2))

    def _word_wise_sent_path_dist(self, q1, q2):

        R = np.zeros((len(q1), len(q2)))

        for i in range(len(q1)):
            for j in range(len(q2)):
                if q1[i][1] == None or q2[j][1] == None:
                    sim = self.word_edit_dist(q1[i][0], q2[j][0])
                else:
                    sim = self.word_path_dist(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

                if sim == None:
                    sim = self.word_edit_dist(q1[i][0], q2[j][0])

                R[i, j] = sim

        return R

    def _word_wise_sent_wup_dist(self, q1, q2):

        R = np.zeros((len(q1), len(q2)))

        for i in range(len(q1)):
            for j in range(len(q2)):
                if q1[i][1] == None or q2[j][1] == None:
                    sim = self.word_edit_dist(q1[i][0], q2[j][0])
                else:
                    sim = self.word_wup_dist(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

                if sim == None:
                    sim = self.word_edit_dist(q1[i][0], q2[j][0])

                R[i, j] = sim

        return R

    def _overall_sim(self, q1, q2, R):

        sum_X = 0.0
        sum_Y = 0.0

        for i in range(len(q1)):
            max_i = 0.0
            for j in range(len(q2)):
                if R[i, j] > max_i:
                    max_i = R[i, j]
            sum_X += max_i

        for i in range(len(q1)):
            max_j = 0.0
            for j in range(len(q2)):
                if R[i, j] > max_j:
                    max_j = R[i, j]
            sum_Y += max_j

        if (float(len(q1)) + float(len(q2))) == 0.0:
            return 0.0

        overall = (sum_X + sum_Y) / (2 * (float(len(q1)) + float(len(q2))))

        return overall

    def _clean_sentence(self, expr, clean_stop=True):
        "remove chars that are not letters or numbers, downcase, then remove stop words"
        regex = re.compile('([^\s\w]|_)+')
        sentence = regex.sub('', expr).lower()
        sentence = sentence.split(" ")
        if clean_stop:
            sentence = self._clean_stop_words(sentence)
        sentence = " ".join(sentence)
        return sentence

    def _clean_stop_words(self, word_list):
        for word in list(word_list):
            if word in self.STOP_WORDS:
                word_list.remove(word)
        return word_list