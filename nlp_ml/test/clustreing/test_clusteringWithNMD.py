from unittest import TestCase
from unittest import mock

import numpy as np

from seq_labeling.clustering.ClusteringWithGMM import ClusteringWithGMM


def mock_random_rand(r_num, c_num):
    return np.ones((r_num, c_num))

class TestClusteringWithNMD(TestCase):
    '''
    TODO not fixed test
    '''
    def test_fit(self):
        self.fail()

    @mock.patch('numpy.random.rand', mock_random_rand)
    def test__init_avgs(self):
        sut = ClusteringWithGMM()
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

        sut = ClusteringWithGMM()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2
        sut._current_cluster_avgs = np.array([[2, 5, 8, 11],
                                              [1, 4, 7, 10]])

        sut._init_covariances()
        np.testing.assert_array_almost_equal(
            sut._current_stds, expected)

    def test__update_all_posterior_probs(self):
        sut = ClusteringWithGMM()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2
        sut._current_cluster_avgs = np.array([[2, 5, 8, 11],
                                              [1, 4, 7, 10]])

        sut._current_cluster_probs = np.array([[1 / 3, 2 / 3]])
        sut._current_covariances = np.array([4, 4])

        sut._update_all_posterior_probs()

        expected = np.array(
            [[np.exp(-1 / 2) / (2 + np.exp(- 1 / 2)), 2 / (2 + np.exp(- 1 / 2))],
             [1 / (1 + 2 * np.exp(- 1 / 2)), 2 * np.exp(- 1 / 2) / (1 + 2 * np.exp(-1 / 2))],
             [np.exp(- 1 / 2) / (np.exp(-1 / 2) + 2 * np.exp(-2)),
              2 * np.exp(-2) / (np.exp(-1 / 2) + 2 * np.exp(-2))]])

        np.testing.assert_array_almost_equal(sut._posterior_probs,
                                             expected)

    def test__compute_new_avgs(self):
        sut = ClusteringWithGMM()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2

        sut._current_covariances = np.array([4, 4])
        sut._posterior_probs = np.array([
            [1 / 2, 1 / 2],
            [1 / 4, 3 / 4],
            [2 / 5, 3 / 5]
        ])

        expected = np.array([
            [44 / 23, 113 / 23, 182 / 23, 251 / 23],
            [76 / 37, 187 / 37, 298 / 37, 409 / 37]
        ])

        actual = sut._compute_new_avgs()

        np.testing.assert_array_almost_equal(actual, expected)

    def test__compute_new_cluster_prob(self):
        sut = ClusteringWithGMM()
        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2

        sut._current_covariances = np.array([4, 4])
        sut._posterior_probs = np.array([
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
        sut = ClusteringWithGMM()
        sut._train_X = np.array([[1, 4, 7],
                                 [2, 5, 8],
                                 [3, 6, 9],
                                 [4, 7, 10]])

        sut._dimension = 3
        sut._dataset_size = 4
        sut._num_clusters = 2
        sut._new_avgs = np.array([[2, 5, 8],
                                  [1, 4, 7]])

        sut._new_cluster_probs = np.array([[1 / 3, 2 / 3]])
        sut._new_covariances = np.array([[[-1, -2, -3],
                                          [-6, -4, -5],
                                          [3, 2, 1]],

                                         [[2, 0, 3],
                                          [3, 0, 5],
                                          [-2, -2, -1]]])

        expected = np.array([
            [(1 / 12) ** (1 / 2) * np.exp(- 13 / 12), 1 / 2 ** (1 / 2)],
            [(1 / 12) ** (1 / 2), 1 / (2) ** (1 / 2) * np.exp(1 / 2)],
            [(1 / 12) ** (1 / 2) * np.exp(- 13 / 12), 1 / 2 ** (1 / 2) * np.exp(2)],
            [(1 / 12) ** (1 / 2) * np.exp(- 13 / 3), 1 / 2 ** (1 / 2) * np.exp(3)]
        ]) / (2 * np.pi) ** (1 / 3)

        sut._update_all_joint_distribution()

        np.testing.assert_array_almost_equal(
            sut._joint_distribution, expected)

    def test__compute_logliklihood(self):
        sut = ClusteringWithGMM()

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2

        sut._joint_distribution = np.array([
            [1 / 4, 1 / 4],
            [1 / 3, 1 / 3],
            [1 / 5, 1 / 5]
        ])

        expected = - 2.01490302054

        actual = sut._compute_log_liklihood()

        np.testing.assert_almost_equal(actual, expected)

    def test_compute_new_covariances(self):
        sut = ClusteringWithGMM()

        sut._train_X = np.array([[1, 4, 7, 10],
                                 [2, 5, 8, 11],
                                 [3, 6, 9, 12]])

        sut._dimension = 4
        sut._dataset_size = 3
        sut._num_clusters = 2

        sut._posterior_probs = np.array([
            [1 / 2, 1 / 2],
            [1 / 4, 3 / 4],
            [2 / 5, 3 / 5]
        ])

        sut._new_avgs = np.array([[2, 5, 8, 11],
                                  [1, 3, 5, 7]])

        expected = np.array([[[18 / 23, 18 / 23, 18 / 23, 18 / 23],
                              [18 / 23, 18 / 23, 18 / 23, 18 / 23],
                              [18 / 23, 18 / 23, 18 / 23, 18 / 23],
                              [18 / 23, 18 / 23, 18 / 23, 18 / 23]],

                             [[63 / 37, 102 / 37, 141 / 37, 180 / 37],
                              [102 / 37, 178 / 37, 254 / 37, 330 / 37],
                              [141 / 37, 254 / 37, 367 / 37, 480 / 37],
                              [180 / 37, 330 / 37, 480 / 37, 630 / 37]]])

        actual = sut._fixed_compute_new_covariances()

        np.testing.assert_array_almost_equal(actual, expected)
