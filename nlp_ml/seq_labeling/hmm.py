import functools
from itertools import chain

import numpy as np

# joint probability of series x and label series y
joint_probability = 0
dataset_example = [[("I", "pronoun"), ("am", "verb"), ("a", "article"), ("student", "noun")],
                   [("You", "noun"), ("are", "verb"), ("a", "article"), ("teacher", "noun")]]


class HmmWordLabel:
    def __init__(self):
        self.dataset = None
        self.words = None
        self.labels = None

        self.vitervi_max_previous_ys = None

    def set_dataset(self, dataset):
        self.dataset = dataset
        flatten_dataset = list(chain.from_iterable(dataset))
        self.words = set(map(lambda l: l[0], flatten_dataset))

    def set_labels(self, labels):
        self.labels = labels
        self.label_key_map = {label: idx for idx, label in enumerate(labels)}

    def word_label_count(self, x, y):
        target_set = [x, y]
        counted_rows = map(lambda x_i: len(
            [True for word_label_set in x_i
             if np.array_equal(word_label_set, target_set)]),
                           self.dataset)
        return sum(counted_rows)

    def seq_label_pair_count(self, y, prev_y):
        counted_label_pair = map(lambda x_i: len([True for ind in
                                                  range(len(x_i) - 1)
                                                  if x_i[ind][1] == prev_y
                                                  and x_i[ind + 1][1] == y]),
                                 self.dataset)
        return sum(counted_label_pair)

    def p_x_when_y(self, x, y):
        # require cashe
        all_y_count = sum(map(functools.partial(self.word_label_count, y=y),
                              self.words))
        target_x_y_count = self.word_label_count(x, y)

        if all_y_count == 0:
            return 0

        return target_x_y_count / all_y_count

    def q_y_when_prev_y(self, y, prev_y):
        # require cashe
        all_seq_y_count = sum(map(functools.partial(self.seq_label_pair_count,
                                                    prev_y=prev_y), self.labels))
        target_seq_y_count = self.seq_label_pair_count(y, prev_y)

        if all_seq_y_count == 0:
            return 0

        return target_seq_y_count / all_seq_y_count

    def p_x_y_when_prev_x_y(self, x, y, prev_y):
        return self.p_x_when_y(x, y) * self.q_y_when_prev_y(y, prev_y)

    def fit(self):
        pass

    # TODO test
    def forword_hmm_viterbi(self, x_for_predict):
        # array for vitervia table (index of x, current key of label,
        # key of prev label and max prob)
        self.vitervi_max_previous_ys = [[() * len(self.labels)] for _
                                        in range(len(x_for_predict))]

        # p(x1, y1) = p(x1, y1| x0, y0) and p(y1) = p(y1|y0) with dummy element (x0, y0)
        for label_index in range(len(self.labels)):
            self.vitervi_previous_ys[0][label_index] = \
                (-100, np.log(self.p_x_when_y(x_for_predict[0], self.labels[label_index])))

        for j in range(1, len(x_for_predict)):
            for y, index in enumerate(self.labels):
                calc_prob_label = lambda label_ind, max_prob: (label_ind,
                                                               np.log(self.p_x_y_when_prev_x_y(x_for_predict[j],
                                                                                               y, self.labels[
                                                                                                   label_ind])) + max_prob)

                probs_each_prev_ys = [calc_prob_label(label_prov[0], label_prov[1])
                                      for label_prov
                                      in self.vitervi_max_previous_ys[j - 1]]

                self.vitervi_max_previous_ys[j][index] = max(probs_each_prev_ys,
                                                             key=lambda l_p: l_p[1])

    #test
    def backword_hmm_viterbi(self, x):
        predicted_y_indeces = [-1 for _ in range(len(x))]
        predicted_y_indeces[-1] = max(self.vitervi_max_previous_ys[len(x) - 1],
                                      key=lambda l_p: l_p[1])[1]
        for j in range(len(x) - 2, -1, -1):
            predicted_y_indeces[j] = max(
                self.vitervi_max_previous_ys[j + 1][predicted_y_indeces[j + 1]],
                key=lambda l_p: l_p[1])[1]
