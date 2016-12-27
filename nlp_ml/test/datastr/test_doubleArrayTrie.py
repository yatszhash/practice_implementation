import unittest
from collections import OrderedDict
from unittest import TestCase

from seq_labeling.data_str.DoubleArrayTrie import DoubleArrayTrie


class TestDoubleArrayTrie(TestCase):
    def setUp(self):
        self.sut = DoubleArrayTrie()

        # words are 102: "ab", 103: "ac", 99: "b", 101: 'd', 104: "da"
        self.sut._base_array[0, DoubleArrayTrie.HEAD_NODE_ID] \
            = DoubleArrayTrie.HEAD_NODE_BASE  # head node

        self.sut._check_array[0, 98] = DoubleArrayTrie.HEAD_NODE_ID  # a path
        self.sut._check_array[0, 99] = DoubleArrayTrie.HEAD_NODE_ID  # b path
        self.sut._check_array[0, 101] = DoubleArrayTrie.HEAD_NODE_ID  # d path

        self.sut._base_array[0, 98] = 4  # node 97, a
        self.sut._check_array[0, 102] = 98  # b path
        self.sut._check_array[0, 103] = 98  # c path

        self.sut._base_array[0, 102] = 8
        self.sut._base_array[0, 103] = 9
        self.sut._is_word_ends[0, 102] = 1
        self.sut._is_word_ends[0, 103] = 1

        self.sut._base_array[0, 99] = 2  # node 98, b
        self.sut._is_word_ends[0, 99] = 1

        self.sut._base_array[0, 101] = 7  # node 98, d
        self.sut._is_word_ends[0, 101] = 1
        self.sut._check_array[0, 104] = 101  # a path

        self.sut._base_array[0, 104] = 10
        self.sut._is_word_ends[0, 104] = 1

    def test_move_child_found(self):
        input_n = 98
        input_ch = 'c'
        expected = 103

        actual = self.sut.move_child(input_n, input_ch)

        self.assertEquals(actual, expected)

    def test_move_child_not_found(self):
        input_n = 98
        input_ch = 'd'
        expected = 0

        actual = self.sut.move_child(input_n, input_ch)

        self.assertEquals(actual, expected)

    def test_common_prefix_search_found(self):
        target_str = "da"

        expected = [101, 104]

        actual = self.sut.common_prefix_search(target_str)

        self.assertListEqual(actual, expected)

    def test_common_prefix_search_not_found(self):
        target_str = "af"

        expected = []

        actual = self.sut.common_prefix_search(target_str)

        self.assertListEqual(actual, expected)

    def test_classify_with_head(self):
        words = [
            "bird",
            "bison",
            "cat"
        ]

        actual_head_chars, actual_classfied_words \
            = self.sut.classify_with_head(words)

        expected_head_chars = ["b", "c"]
        expected_classified_words = [["ird", "ison"], ["at"]]

        self.assertListEqual(actual_head_chars, expected_head_chars)
        self.assertListEqual(actual_classfied_words, expected_classified_words)


class TestDoubleArrayTrieGenerate(TestCase):
    def setUp(self):
        self.sut = DoubleArrayTrie()

    def test_add_all(self):
        words = [
            "bird",
            "bison",
            "cat"
        ]

        self.sut.add_all_statically(words)

        self.assertTrue(self.sut.common_prefix_search("bird"))
        self.assertTrue(self.sut.common_prefix_search("bison"))
        self.assertTrue(self.sut.common_prefix_search("cat"))
        self.assertFalse(self.sut.common_prefix_search("bir"))

    def test_add_all2(self):
        words = {
            "う": 1,
            "ぎ": 1,
            "し": 1,
            "ひ": 1,
            "うし": 2,
            "ぎじ": 2,
            "しゃ": 2,
            "うろん": 2,
            "ひょう": 2,
            "ぎじゅつ": 3,
            "ひょうか": 3,
            "ひょうろん": 3
        }

        self.sut.add_all_statically(words.keys())

        for word, size in words.items():
            result = self.sut.common_prefix_search(word)
            self.assertEquals(len(result), size)

    def test_get_word(self):
        words = {
            "う": 1,
            "ぎ": 1,
            "し": 1,
            "ひ": 1,
            "うし": 2,
            "ぎじ": 2,
            "しゃ": 2,
            "うろん": 2,
            "ひょう": 2,
            "ぎじゅつ": 3,
            "ひょうか": 3,
            "ひょうろん": 3
        }

        self.sut.add_all_statically(words.keys())

        node_num = 12437
        actual = self.sut.get_word(node_num)

        self.assertEquals(actual, "ひょうろん")

    def test_add_dynamicaly(self):
        words = OrderedDict(
            (("う", 1),
             ("ぎ", 1),
             ("し", 1),
             ("ひ", 1),
             ("うし", 2),
             ("ぎじ", 2),
             ("しゃ", 2),
             ("うろん", 2),
             ("ひょう", 2),
             ("ぎじゅつ", 3),
             ("ひょうか", 3),
             ("ひょうろん", 3),
             ("うろんな", 3),
             ("ひょうろう", 3),
             ("ひょうろんか", 4))
        )
        words_list = list(words.keys())
        self.sut.add_all_statically(words_list[:-3])

        result = self.sut.common_prefix_search("ひょうろん")
        self.assertEquals(len(result), 3)

        new_word1 = words_list[-3]
        self.sut.add_dynamically(new_word1)

        result = self.sut.common_prefix_search("ひょうろん")
        self.assertEquals(len(result), 3)

        new_word2 = words_list[-2]
        self.sut.add_dynamically(new_word2)

        result = self.sut.common_prefix_search("ひょうろん")
        self.assertEquals(len(result), 3)

        new_word3 = words_list[-1]
        self.sut.add_dynamically(new_word3)

        result = self.sut.common_prefix_search("ひょうろん")
        self.assertEquals(len(result), 3)

        for word, size in words.items():
            result = self.sut.common_prefix_search(word)
            print(word)
            self.assertEquals(len(result), size)

    def test_transfer_node_into_empty(self):
        sut = DoubleArrayTrie()
        sut._base_array[0, 1] = 1

    def test_transfer_node(self):
        sut = DoubleArrayTrie()
        sut._base_array[0, 1] = 1

        sut._check_array[0, 2] = 1
        sut._check_array[0, 10] = 1
        sut._check_array[0, 100] = 1

        sut._base_array[0, 2] = 1
        sut._check_array[0, 3] = 2  # k = 2
        sut._check_array[0, 4] = 2  # k = 3
        sut._check_array[0, 9] = 2  # k = 8

        sut._is_word_ends[0, 3] = True
        sut._is_word_ends[0, 4] = False
        sut._is_word_ends[0, 9] = True

        # transfer \#2 node's children
        sut._current_transfer_parent = 2
        sut._current_transfer_children = sut.compute_all_child_nodes(2)
        sut._current_transfer_ks = [child - sut._base_array[0, 2] for child
                                    in sut._current_transfer_children]

        old_start_node = 3
        new_start_node = 13
        sut._transfer_node(old_start_node, new_start_node)

        self.assertEquals(sut._base_array[0, 2], 11)

        # check new nodes
        self.assertEquals(sut._check_array[0, 13], 2)
        self.assertEquals(sut._check_array[0, 14], 2)
        self.assertEquals(sut._check_array[0, 19], 2)

        self.assertTrue(sut.is_word_id(13))
        self.assertFalse(sut.is_word_id(14))
        self.assertTrue(sut.is_word_id(19))

        # check old nodes
        self.assertEquals(sut._check_array[0, 3], DoubleArrayTrie.NOT_FOUND)
        self.assertEquals(sut._check_array[0, 4], DoubleArrayTrie.NOT_FOUND)
        self.assertEquals(sut._check_array[0, 9], DoubleArrayTrie.NOT_FOUND)

        self.assertFalse(sut.is_word_id(3))
        self.assertFalse(sut.is_word_id(4))
        self.assertFalse(sut.is_word_id(9))

if __name__ == '__main__':
    unittest.main()
