import numpy as np
from scipy.sparse import lil_matrix


class DoubleArrayTrie:
    NOT_FOUND = 0  # to use sparse matrix, use 0 as not found symbol
    INITIAL_SIZE = 1000000
    HEAD_NODE_BASE = 1
    HEAD_NODE_ID = 1
    END_LABEL = -1

    def __init__(self):
        self.base_array = lil_matrix((1, DoubleArrayTrie.INITIAL_SIZE),
                                     dtype=np.int64)
        self.check_array = lil_matrix((1, DoubleArrayTrie.INITIAL_SIZE),
                                      dtype=np.int64)

        self.is_dics = lil_matrix((1, DoubleArrayTrie.INITIAL_SIZE),
                                  dtype=np.int64)

    def get_base(self, n):
        return self.base_array[0, n]

    def get_check(self, m):
        return self.check_array[0, m]

    def move_child(self, n, ch):
        k = ord(ch)
        b = self.get_base(n)
        if b == DoubleArrayTrie.END_LABEL:
            return DoubleArrayTrie.NOT_FOUND
        m = b + k
        if self.get_check(m) == n:
            return m
        return DoubleArrayTrie.NOT_FOUND

    def common_prefix_search(self, target_str):
        n = DoubleArrayTrie.HEAD_NODE_ID  # head

        result = []

        for ch in target_str:
            n = self.move_child(n, ch)
            if n == DoubleArrayTrie.NOT_FOUND:
                break
            if self.is_word_id(n):
                result.append(n)

        return result

    def is_word_id(self, n):
        return self.is_dics[0, n]

    def add_all(self, words):
        sorted_words = sorted(words)
        self.add_nodes(1, sorted_words)

    def add_nodes(self, n, words):

        if len(words) == 0:
            return False

        head_chars, classified_words, dict_flags = self.classify_with_head(words)

        vfunc = np.vectorize(ord)
        ks = vfunc(head_chars)
        k_min = np.min(ks)

        is_added = False
        for i in range(k_min + 1, self.check_array.shape[1]):
            if self.check_array[0, i] == DoubleArrayTrie.NOT_FOUND:
                if all([self.check_array[0, i + k_j - k_min]
                                == DoubleArrayTrie.NOT_FOUND for k_j in ks]):
                    self.base_array[0, n] = i - k_min
                    for k_j in ks:
                        self.check_array[0, i + k_j - k_min] = n
                    is_added = True

                    for j in range(len(ks)):
                        self.is_dics[0, i + ks[j] - k_min] = int(dict_flags[j])
                        self.add_nodes(i + ks[j] - k_min, classified_words[j])
                    break

        return is_added

    def classify_with_head(self, words):
        head_chars = sorted(set(map(lambda x: x[0], words)))

        classified_words = []
        dict_flags = []

        word_idx = 0
        for ch in head_chars:
            is_same_ch = False
            same_ch_words = []
            each_dict_flag = False

            classified_words.append(same_ch_words)

            while word_idx < len(words):
                word = words[word_idx]
                if word[0] == ch and not is_same_ch:
                    is_same_ch = True
                elif word[0] != ch and is_same_ch:
                    break
                if is_same_ch and len(word) > 1:
                    same_ch_words.append(word[1:])
                if is_same_ch and len(word) == 1:
                    each_dict_flag = True
                word_idx += 1
            dict_flags.append(each_dict_flag)

        return head_chars, classified_words, dict_flags
