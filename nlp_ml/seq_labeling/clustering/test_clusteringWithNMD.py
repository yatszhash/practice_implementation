from unittest import TestCase
from unittest import mock

import numpy as np

from seq_labeling.clustering.ClusteringWithNMD import ClusteringWithNMD


def mock_random_rand(r_num, c_num):
    return np.ones((r_num, c_num))


class TestClusteringWithNMD(TestCase):
    def test_fit(self):
        self.fail()

    @mock.patch('numpy.random.rand', mock_random_rand)
    def test__init_avgs_stds(self):
        sut = ClusteringWithNMD()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])
        sut._dimension = 4
        sut._dataset_size = 3

        sut._num_clusters = 2

        expected = np.array(
            [[2 * np.sqrt(2.0 / 3) + 2, 2 * np.sqrt(2.0 / 3) + 5, 2 * np.sqrt(2.0 / 3) + 8, 2 * np.sqrt(2.0 / 3) + 11],
             [2 * np.sqrt(2.0 / 3) + 2, 2 * np.sqrt(2.0 / 3) + 5, 2 * np.sqrt(2.0 / 3) + 8, 2 * np.sqrt(2.0 / 3) + 11],
             ])

        sut._init_avgs()

        np.testing.assert_array_almost_equal(sut._current_cluster_avgs, expected)
        # sut._compute_std()
        # print(sut._current_cluster_stds)
        #
        # #todo more correct
        # self.assertEqual(sut._current_cluster_avgs.shape, (2, 4))

    def test__compute_std(self):
        expected = np.sqrt(7 / 6)

        sut = ClusteringWithNMD()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2
        sut._current_cluster_avgs = np.array([[2, 5, 8, 11],
                                              [1, 4, 7, 10]])

        sut._init_std()
        np.testing.assert_array_almost_equal(
            sut._current_std, expected)

    def test__update_all_posterior_probs(self):
        sut = ClusteringWithNMD()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2
        sut._current_cluster_avgs = np.array([[2, 5, 8, 11],
                                              [1, 4, 7, 10]])

        sut._current_cluster_probs = np.array([[1 / 3, 2 / 3]])
        sut._current_std = 2

        sut._update_all_posterior_probs()

        expected = np.array(
            [[np.exp(-1 / 2) / (2 + np.exp(- 1 / 2)), 2 / (2 + np.exp(- 1 / 2))],
             [1 / (1 + 2 * np.exp(- 1 / 2)), 2 * np.exp(- 1 / 2) / (1 + 2 * np.exp(-1 / 2))],
             [np.exp(- 1 / 2) / (np.exp(-1 / 2) + 2 * np.exp(-2)),
              2 * np.exp(-2) / (np.exp(-1 / 2) + 2 * np.exp(-2))]])

        np.testing.assert_array_almost_equal(sut._posterior_probs_on_param,
                                             expected)

    def test__compute_new_cluster_avgs(self):
        sut = ClusteringWithNMD()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2
        sut._current_cluster_avgs = np.array([[2, 5, 8, 11],
                                              [1, 4, 7, 10]])

        sut._current_cluster_probs = np.array([[1 / 3, 2 / 3]])
        sut._current_std = 2
        sut._posterior_probs_on_param = np.array([
            [1 / 2, 1 / 2],
            [1 / 4, 3 / 4],
            [2 / 5, 3 / 5]
        ])

        expected = np.array([
            [-1 / 40, -1 / 40, -1 / 40, -1 / 40],
            [39 / 80, 39 / 80, 39 / 80, 39 / 80]
        ])

        actual = sut._comute_new_cluster_avgs()

        np.testing.assert_array_almost_equal(actual, expected)

    def test__compute_new_cluster_prob(self):
        sut = ClusteringWithNMD()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2

        sut._current_std = 2
        sut._posterior_probs_on_param = np.array([
            [1 / 2, 1 / 2],
            [1 / 4, 3 / 4],
            [2 / 5, 3 / 5]
        ])

        expected = np.array([
            [23 / 60, 37 / 60]
        ])

        actual = sut._compute_new_cluster_prob()

        np.testing.assert_array_almost_equal(actual, expected)

    def test__update_all_joint_distribution(self):
        sut = ClusteringWithNMD()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2
        sut._new_cluster_avgs = np.array([[2, 5, 8, 11],
                                          [1, 4, 7, 10]])

        sut._new_cluster_probs = np.array([[1 / 3, 2 / 3]])
        sut._new_std = 2

        expected = np.array([
            [1 / (192 * np.pi ** 2) * np.exp(- 1 / 2), 1 / 96 / np.pi ** 2],
            [1 / (192 * np.pi ** 2), 1 / 96 * np.exp(-1 / 2) / np.pi ** 2],
            [1 / (192 * np.pi ** 2) * np.exp(- 1 / 2),
             1 / 96 * np.exp(- 2) / np.pi ** 2]
        ])

        sut._update_all_joint_distribution()

        np.testing.assert_array_almost_equal(
            sut._joint_distribution, expected)

    def test__compute_logliklihood(self):
        sut = ClusteringWithNMD()

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2

        sut._joint_distribution = np.array([
            [1 / 4, 1 / 4],
            [1 / 3, 1 / 3],
            [1 / 5, 1 / 5]
        ])

        expected = - 2.01490302054

        actual = sut._compute_logliklihood()

        np.testing.assert_almost_equal(actual, expected)
