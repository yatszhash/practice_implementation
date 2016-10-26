import itertools

import nltk


def transform_corpus_to_sentence(self, sentence_num):
    corpus = nltk.corpus.brown.tagged_words(tagset="universal")
    all_end_indices = filter(lambda w: w[0] == ".", enumerate(corpus))
    end_indices = itertools.islice(all_end_indices, 10)

    return [list(corpus.isslice(corpus, end_indices[i - 1] + 1, end_index + 1))
            if i > 0 else list(corpus.isslice(corpus, 0, end_index + 1))
            for end_index, i in enumerate(end_indices)]


if __name__ == "__main__":
    sentences = transform_corpus_to_sentence(10)
    sentences.re
