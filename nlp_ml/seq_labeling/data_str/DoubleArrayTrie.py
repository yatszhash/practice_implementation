import numpy as np
from scipy.sparse import lil_matrix

from seq_labeling.data_str.DoubleTrieFullException import DoubleTrieFullException


class DoubleArrayTrie:
    NOT_FOUND = 0  # to use sparse matrix, use 0 as not found symbol
    INITIAL_SIZE = 1000000
    HEAD_NODE_BASE = 1
    HEAD_NODE_ID = 1
    END_LABEL = -1
    CHAR_MAX_ORDINAL = 65536

    def __init__(self):
        self.base_array = lil_matrix((1, DoubleArrayTrie.INITIAL_SIZE),
                                     dtype=np.int64)
        self.check_array = lil_matrix((1, DoubleArrayTrie.INITIAL_SIZE),
                                      dtype=np.int64)

        self.is_word_ends = lil_matrix((1, DoubleArrayTrie.INITIAL_SIZE),
                                       dtype=np.int64)

    def get_word(self, node_num):
        if not self.is_word_ends[0, node_num]:
            return None

        m = node_num
        nodes = []

        while True:
            parent = self.check_array[0, m]

            nodes.append(m - self.base_array[0, parent])

            if parent == 1:
                break

            m = parent

        word = ""

        while nodes:
            word += chr(nodes.pop())

        return word

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
        return self.is_word_ends[0, n]

    def add_all_statically(self, words):
        sorted_words = sorted(words)
        self.add_nodes_statically(1, sorted_words)

    def add_nodes_statically(self, n, words):

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
                        self.is_word_ends[0, i + ks[j] - k_min] = int(dict_flags[j])
                        self.add_nodes_statically(i + ks[j] - k_min, classified_words[j])
                    break

        return is_added

    def add_dynamically(self, word):
        self.add_node_dynamically(1, word)

    def add_node_dynamically(self, n, word):
        if word == "":
            return

        m = self.base_array[0, n] + ord(word[0])

        next_parent_node = None
        # not registered yet
        if self.check_array[0, m] == DoubleArrayTrie.NOT_FOUND:
            self.check_array[0, m] = n
            next_parent_node = m

        # already registered same node
        elif self.check_array[0, m] == n:
            next_parent_node = m

        # already registered but conflict
        elif self.check_array[0, m] != n:
            conflict_n = self.check_array[0, m]
            n_children = self.compute_all_child_nodes(n)
            conflict_n_children = self.compute_all_child_nodes(conflict_n)

            if len(n_children) < len(conflict_n_children):
                next_parent_node = self.transfer_node(n, word[0])

            else:
                self.transfer_node(conflict_n)
                self.check_array[0, m] = n
                next_parent_node = m

        if len(word) == 1:
            self.is_word_ends[0, next_parent_node] = True
            return

        self.add_node_dynamically(next_parent_node, word[1:])

        return

    def transfer_node(self, n, add_node_char=None):
        children = self.compute_all_child_nodes(n)
        ks = [child - self.base_array[0, n] for child in children]

        old_start_node = min(children)
        max_node = max(children)
        k_min = min(ks)
        k_max = min(ks)

        add_k = -1
        if add_node_char:
            add_k = ord(add_node_char)

        for new_start_node in range(max_node + 1,
                                    self.check_array.shape[1] - k_max):

            if all([self.check_array[0, new_start_node + k - k_min]
                            == DoubleArrayTrie.NOT_FOUND for k in ks]):

                if add_k > 0 \
                        and self.check_array[0, new_start_node + add_k - k_min] \
                                != DoubleArrayTrie.NOT_FOUND:
                    continue

                self.base_array[0, n] = new_start_node - k_min
                self.transfer_children(k_max, k_min,
                                       new_start_node, old_start_node, n)

                if add_k > 0:
                    new_added_node = new_start_node + add_k - k_min
                    self.check_array[0, new_added_node] = n
                    return new_added_node

                return None

        raise DoubleTrieFullException()

    def transfer_children(self, k_max, k_min, new_start_node,
                          old_start_node, n):
        # add children in new places
        for k_offset in range(0, k_max - k_min + 1):
            new_node = new_start_node + k_offset
            old_node = old_start_node + k_offset
            self.check_array[0, new_node] = n
            self.base_array[0, new_node] = self.base_array[0, old_node]
            self.is_word_ends[0, new_node] = self.is_word_ends[0, old_node]

            self.check_array[0, old_node] = DoubleArrayTrie.NOT_FOUND
            self.base_array[0, old_node] = DoubleArrayTrie.NOT_FOUND
            self.is_word_ends[0, old_node] = False

    def compute_all_child_nodes(self, n):
        return [self.base_array[0, n] + k
                for k in range(1, DoubleArrayTrie.CHAR_MAX_ORDINAL)
                if self.check_array[0, self.base_array[0, n] + k] == n]

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
