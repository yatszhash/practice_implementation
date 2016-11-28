import unittest
from unittest import TestCase

from seq_labeling.data_str.DoubleArrayTrie import DoubleArrayTrie


class TestDoubleArrayTrie(TestCase):
    def setUp(self):
        self.sut = DoubleArrayTrie()

        # words are 102: "ab", 103: "ac", 99: "b", 101: 'd', 104: "da"
        self.sut.base_array[0, DoubleArrayTrie.HEAD_NODE_ID] \
            = DoubleArrayTrie.HEAD_NODE_BASE  # head node

        self.sut.check_array[0, 98] = DoubleArrayTrie.HEAD_NODE_ID  # a path
        self.sut.check_array[0, 99] = DoubleArrayTrie.HEAD_NODE_ID  # b path
        self.sut.check_array[0, 101] = DoubleArrayTrie.HEAD_NODE_ID  # d path

        self.sut.base_array[0, 98] = 4  # node 97, a
        self.sut.check_array[0, 102] = 98  # b path
        self.sut.check_array[0, 103] = 98  # c path

        self.sut.base_array[0, 102] = 8
        self.sut.base_array[0, 103] = 9
        self.sut.is_dics[0, 102] = 1
        self.sut.is_dics[0, 103] = 1

        self.sut.base_array[0, 99] = 2  # node 98, b
        self.sut.is_dics[0, 99] = 1

        self.sut.base_array[0, 101] = 7  # node 98, d
        self.sut.is_dics[0, 101] = 1
        self.sut.check_array[0, 104] = 101  # a path

        self.sut.base_array[0, 104] = 10
        self.sut.is_dics[0, 104] = 1

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

        self.sut.add_all(words)

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

        self.sut.add_all(words.keys())

        for word, size in words.items():
            result = self.sut.common_prefix_search(word)
            self.assertEquals(len(result), size)


if __name__ == '__main__':
    unittest.main()
