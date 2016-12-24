import unittest
from unittest import TestCase

from seq_labeling.svm.L1RegSVM import L1RegSVM


class TestL1RegSVM(TestCase):
    def test_learn_all_with_FOBOS(self):
        pass

    def test_update_w_with_fobos(self):
        # TODO implement test
        # clf = L1RegSVM(0.5, 1.0)
        # w = np.array([1, 1])
        # x_k = np.array([2, 3])
        # y = np.array([1])
        # actual = clf.update_w_with_fobos(w, x_k, y)
        pass

    def test_clip_v_plus_larger_than_c(self):
        v = 100
        c = 1

        expected = 99
        self.assertEqual(L1RegSVM.clip(v, c), expected)

    def test_clip_v_minus_larger_than_c(self):
        v = -100
        c = 1

        expected = -99

        self.assertEqual(L1RegSVM.clip(v, c), expected)

    def test_clip_v_plus_smaller_than_c(self):
        v = 1
        c = 100

        expected = 0

        self.assertEqual(L1RegSVM.clip(v, c), expected)

    def test_clip_v_minus_smaller_than_c(self):
        v = -1
        c = 100

        expected = 0

        self.assertEqual(L1RegSVM.clip(v, c), expected)


if __name__ == "__main__":
    unittest.main()
