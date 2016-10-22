import unittest

import numpy as np
from nlp_ml.ch5.hmm import *


class Ch5TestCase(unittest.TestCase):
    def test_word_label_count(self):
        dataset = np.array([np.array([("I", "pronoun"), ("am", "verb"), ("a", "article"), ("student", "noun")]),
                            np.array([("You", "pronoun"), ("are", "verb"), ("a", "article"), ("teacher", "noun"),
                                      ("Big", "propernoun")])])
        hmm = HmmWordLabel()
        hmm.set_dataset(dataset)
        a_article_count = hmm.word_label_count("a", "article")

        self.assertEqual(a_article_count, 2)

    def test_seq_label_pair_count(self):
        dataset = np.array([np.array([("I", "pronoun"), ("am", "verb"), ("a", "article"), ("student", "noun")]),
                            np.array([("You", "pronoun"), ("are", "verb"), ("a", "article"), ("teacher", "noun"),
                                      ("Big", "propernoun")])])
        hmm = HmmWordLabel()
        hmm.set_dataset(dataset)
        verb_article_count = hmm.seq_label_pair_count("article", "verb")
        self.assertEqual(verb_article_count, 2)

    def test_p_x_when_y(self):
        dataset = np.array([np.array([("I", "pronoun"), ("am", "verb"), ("a", "article"), ("student", "noun")]),
                            np.array([("You", "pronoun"), ("are", "verb"), ("a", "article"), ("teacher", "noun"),
                                      ("Big", "propernoun")])])
        hmm = HmmWordLabel()
        hmm.set_dataset(dataset)
        p_are_on_verb = hmm.p_x_when_y("am", "verb")
        expected = 1 / 2

        self.assertEqual(p_are_on_verb, expected)

    def test_q_y_when_prev_y(self):
        dataset = np.array([np.array([("I", "pronoun"), ("am", "verb"),
                                      ("a", "article"), ("student", "noun")]),
                            np.array([("You", "pronoun"), ("are", "verb"), ("a", "article"), ("teacher", "noun"),
                                      ("Big", "propernoun")])])

        hmm = HmmWordLabel()
        hmm.set_dataset(dataset)
        labels = ["verb", "noun", "adjective", "determiner", "adverb", "pronoun",
                  "preposition", "conjunction", "interjection", "propernoun",
                  "article"]

        hmm.set_labels(labels)
        article_on_verb = hmm.q_y_when_prev_y("article", "verb")
        expected = 1

        self.assertEqual(article_on_verb, expected)


if __name__ == '__main__':
    unittest.main()
