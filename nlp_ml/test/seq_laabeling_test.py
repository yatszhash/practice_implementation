import itertools
import unittest

import nltk
from nlp_ml.seq_labeling.hmm import *


class Ch5TestCase(unittest.TestCase):
    def setUp(self):
        self.sentence5 = Ch5TestCase.transform_corpus_to_sentence(5)

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

    def test_p_x_y_when_prev_x_y(self):
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

        teacher_noun_on_a_article = hmm.p_x_y_when_prev_x_y("teacher",
                                                            "noun", "article")
        expected = 0.5

        self.assertEqual(teacher_noun_on_a_article, expected)

    def test_forword_hmm_viterbi(self):
        pass

    # load brown corpus as sentence list
    @classmethod
    def transform_corpus_to_sentence(cls, sentence_num):
        corpus = nltk.corpus.brown.tagged_words(tagset="universal")
        all_end_indices = filter(lambda w: w[1][0] == ".", enumerate(corpus))
        end_indices = list(itertools.islice(all_end_indices, sentence_num))

        return [list(itertools.islice(corpus, end_indices[i - 1][0] + 1, end_index[0] + 1))
                if i > 0 else list(itertools.islice(corpus, 0, end_index[0] + 1))
                for i, end_index in enumerate(end_indices)]

    def test_transform_corpus_to_sentence(self):
        actual = Ch5TestCase.transform_corpus_to_sentence(1)
        expected = [[('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'),
                     ('Grand', 'ADJ'), ('Jury', 'NOUN'), ('said', 'VERB'),
                     ('Friday', 'NOUN'), ('an', 'DET'), ('investigation', 'NOUN'),
                     ('of', 'ADP'), ("Atlanta's", 'NOUN'), ('recent', 'ADJ'),
                     ('primary', 'NOUN'), ('election', 'NOUN'), ('produced', 'VERB'),
                     ('``', '.'), ('no', 'DET'), ('evidence', 'NOUN'), ("''", '.'), ('that', 'ADP'),
                     ('any', 'DET'), ('irregularities', 'NOUN'), ('took', 'VERB'),
                     ('place', 'NOUN'), ('.', '.')]]

        self.assertListEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
