import logging
import sys

import numpy as np
from numpy.linalg import pinv
from sklearn.cluster import KMeans


class ClusteringWithGMM:
    '''
        clustering with normal mixture distribution
        EM algorithm is used.
    '''

    def __init__(self):
        self._max_iter = 400
        self._num_iter = 0

        self._train_X = None

        self._dataset_size = None
        self._dimension = None

        self._num_clusters = None
        self._current_cluster_avgs = None
        self._current_covariances = None
        self._current_cluster_probs = None
        self._random_state = None

        self._new_avgs = None
        self._new_cluster_probs = None
        self._new_covariances = None

        self._posterior_probs = None

        self._joint_distribution = None
        self._previous_log_likelihood = -sys.maxsize
        self._log_liklihood_history = None

        self._mixed_dist = None

        self._stop_threshold = 1e-4

    def fit(self, X, num_clusters=10, random_state=None):
        self._num_iter = 0
        self._num_clusters = num_clusters
        self._random_state = random_state
        self._dimension = X.shape[1]
        self._dataset_size = X.shape[0]
        self._previous_log_likelihood = -sys.maxsize
        self._log_liklihood_history = np.ones(self._max_iter)
        self._log_liklihood_history[0] = self._previous_log_likelihood

        self._train_X = X

        self._init_params()

        for _ in range(self._max_iter):
            isLearningEnd = self._each_step()

            if isLearningEnd:
                logging.debug("log liklihood history : \n {}".format(
                    self._log_liklihood_history
                ))
                break

    def predict(self, X):
        result = np.apply_along_axis(self._predict_each_x, 1, X)
        return result

    def _predict_each_x(self, x):
        probs = self._compute_posterior_probs_on_new_data(x)

        cluster_num = np.argmax(a=probs, axis=0)

        return cluster_num

    def get_learning_result(self):
        return np.copy(self._posterior_probs)

    # TODO whiy log liklihood increases
    def _each_step(self):
        self._num_iter += 1
        logging.info("iteration : {}".format(self._num_iter))
        self._update_all_posterior_probs()
        self._compute_new_params()

        self._update_all_mixed_dist()
        self._update_joint_distribution()

        log_liklihood = self._compute_log_liklihood()
        self._log_liklihood_history[self._num_iter] = log_liklihood
        logging.info("current log liklihood : {}".format(log_liklihood))
        logging.debug("currrent params\n"
                      "prior cluster prob : {0}\n"
                      "cluster avgs : {1}\n"
                      "covarinaces : {2}\n\n"
                      .format(self._new_cluster_probs, self._new_avgs,
                              self._new_covariances))

        isFinished = False

        if log_liklihood < self._previous_log_likelihood:
            logging.error("log liklihood has increased. "
                          "Something strange in implementation\n"
                          "previous  log liklihood: {0}\n"
                          "current log liklihood : {1}\n"
                          .format(self._previous_log_likelihood, log_liklihood))
            # raise IllegalLearningException()

        elif log_liklihood - self._previous_log_likelihood \
                <= self._stop_threshold:
            isFinished = True

        self._update_params()

        if np.isnan(self._current_cluster_avgs).any() or \
                np.isnan(self._current_covariances).any() or \
                np.isnan(self._current_cluster_probs).any():
            logging.warning("some paramaters include Nan. "
                            "The learning was failed and terminated forcibly")
            isFinished = True

        self._previous_log_likelihood = log_liklihood
        return isFinished

    def _init_params(self):
        np.random.seed(self._random_state)
        #
        # self._init_avgs()
        # self._init_covariances()
        # self._init_cluster_probs()

        self._init_with_kmeans()

        self._update_all_mixed_dist()
        self._update_joint_distribution()

    def _init_with_kmeans(self):
        estimator = KMeans(n_clusters=self._num_clusters)
        estimator.fit(self._train_X)

        init_Y = estimator.predict(self._train_X)

        c_indice = np.arange(self._num_clusters)
        c_avg = lambda c_idx: np.mean(self._train_X[init_Y == c_idx], axis=0)
        self._current_cluster_avgs = np.array([c_avg(c_idx)
                                               for c_idx in c_indice])

        self._new_avgs = self._current_cluster_avgs

        c_cov = lambda c_idx: np.cov(self._train_X[init_Y == c_idx].T)
        self._current_covariances = np.array([c_cov(c_idx) for c_idx in c_indice])
        self._new_covariances = self._current_covariances

        self._current_cluster_probs = np.unique(init_Y, return_counts=True)[1] \
                                      / self._dataset_size
        self._new_cluster_probs = self._current_cluster_probs

    def _init_avgs(self):
        self._current_cluster_avgs = np.random.rand(self._num_clusters,
                                                    self._dimension)

        self._new_avgs = self._current_cluster_avgs

    def _init_cluster_probs(self):
        self._current_cluster_probs = \
            np.random.dirichlet(np.ones(self._num_clusters), size=1) \
                .reshape(1, self._num_clusters)
        self._new_cluster_probs = self._current_cluster_avgs

    def _init_covariances(self):
        # self._current_covariances = self._compute_new_covariances()
        self._current_covariances = np.random.rand(self._num_clusters,
                                                   self._dimension, self._dimension) / 4
        self._new_covariances = self._current_covariances

    def _compute_log_liklihood(self):
        return np.log(self._joint_distribution.sum(axis=1)) \
            .sum(axis=0)


    def _update_all_posterior_probs(self):
        self._posterior_probs = np.array([
                                             self._mixed_dist[x_idx, :] / self._joint_distribution[x_idx]
                                             for x_idx in np.arange(self._dataset_size)
                                             ])

    def _compute_posterior_probs_on_new_data(self, x):

        return np.array([self._compute_mixture_distribution(x, c_idx)
                         for c_idx in np.arange(self._num_clusters)])

    '''
    # def _update_all_posterior_probs(self):
    #     self._posterior_probs = np.zeros((self._dataset_size,
    #                                       self._num_clusters))
    #     x_indices = np.arange(self._dataset_size)
    #     for i in x_indices:
    #         self._posterior_probs[i, :] \
    #             = self._compute_posterior_prob_on_each_x(i)

    # def _compute_posterior_prob_on_each_x(self, x_idx):
    #     vfunc = np.vectorize(lambda c_idx:
    #                          self._compute_posterior_prob(self._train_X[x_idx, :], c_idx))
    #     posterior_probs = np.apply_along_axis(vfunc, 1,
    #                                           np.arange(self._num_clusters)
    #                                           .reshape(1, self._num_clusters))
    #
    #     summed = posterior_probs.sum()
    #
    #     return posterior_probs / summed

    # def _compute_posterior_prob(self, x, c_idx):
    #     diff = (x - self._current_cluster_avgs[c_idx, :]).reshape(self._dimension, 1)
    #     cov = self._current_covariances[c_idx, :, :]
    #     co_inv = pinv(cov)
    #     first = np.dot(diff.T, co_inv)
    #     second = np.dot(first, diff)
    #     in_exp = np.exp(- 1 / 2 * second)
    #
    #     det = (2 * np.pi) ** (1 / self._dimension) * np.linalg.det(cov) ** (1 / 2)
    #
    #     result = self._new_cluster_probs[0, c_idx] / det * in_exp
    #
    #     return self._current_cluster_probs[0, c_idx] \
    #            * np.exp(
    #         - np.linalg.norm(x - self._current_cluster_avgs[c_idx, :]) ** 2 \
    #         / (2 * self._current_covariances[c_idx]))

    # def _update_all_joint_distribution(self):
    #     self._joint_distribution = np.zeros((self._dataset_size,
    #                                          self._num_clusters))
    #     for x_idx in np.arange(self._dataset_size):
    #         self._joint_distribution[x_idx, :] \
    #             = self._compute_joint_distribution_on_each_x(x_idx)
    #
    # def _compute_joint_distribution_on_each_x(self, x_idx):
    #     vfunc = lambda c_idx: self._compute_joint_distribution(
    #         self._train_X[x_idx, :], c_idx )
    #
    #     result =  np.array([vfunc(c_idx) for c_idx in np.arange(self._num_clusters)])
    #
    #     return result.ravel().ravel()
    #
    #
    # def _compute_joint_distribution(self, x, c_idx):
    #     diff = (x - self._new_avgs[c_idx, :]).reshape(self._dimension, 1)
    #     cov = self._new_covariances[c_idx, :, :]
    #     co_inv = pinv(cov)
    #     first = np.dot(diff.T, co_inv)
    #     second = np.dot(first, diff)
    #     in_exp = np.exp(- 1 / 2 * second)
    #
    #     det = (2 * np.pi) ** (1 / self._dimension) * np.linalg.det(cov) ** (1 /2)
    #
    #     result = self._new_cluster_probs[0, c_idx]  / det * in_exp
    #
    #     return result
    '''

    def _update_joint_distribution(self):
        each_p_x = lambda x_idx: \
            np.dot(self._new_cluster_probs.reshape(1, self._num_clusters),
                   self._mixed_dist[x_idx, :].T)
        p_x_0 = np.dot(self._new_cluster_probs.reshape(1, self._num_clusters),
                       self._mixed_dist[0, :].T)

        all_p_x = np.apply_along_axis(each_p_x, 1,
                                      np.arange(self._dataset_size).reshape(self._dataset_size, 1))

        self._joint_distribution = all_p_x

    def _update_all_mixed_dist(self):
        self._mixed_dist = np.zeros((self._dataset_size,
                                             self._num_clusters))
        for x_idx in np.arange(self._dataset_size):
            self._mixed_dist[x_idx, :] \
                = self._compute_mixture_distribution_on_each_x(x_idx)

    def _compute_mixture_distribution_on_each_x(self, x_idx):
        vfunc = lambda c_idx: self._compute_mixture_distribution(
            self._train_X[x_idx, :], c_idx)

        result = np.array([vfunc(c_idx) for c_idx in np.arange(self._num_clusters)])

        return result.ravel().ravel()

    def _compute_mixture_distribution(self, x, c_idx):
        diff = (x - self._new_avgs[c_idx, :]).reshape(self._dimension, 1)
        cov = self._new_covariances[c_idx, :, :]
        co_inv = pinv(cov)
        first = np.dot(diff.T, co_inv)
        second = np.dot(first, diff)
        in_exp = np.exp(- 1 / 2 * second)

        det = (2 * np.pi) ** (1 / self._dimension) * np.linalg.det(cov) ** (1 / 2)

        result = 1 / det * in_exp

        return result

    def _compute_new_params(self):
        self._new_avgs = self._compute_new_avgs()
        self._new_covariances = self._compute_new_covariances()
        self._new_cluster_probs = self._compute_new_cluster_prob()

    def _update_params(self):
        self._current_cluster_avgs = self._new_avgs
        self._current_covariances = self._new_covariances
        self._current_cluster_probs = self._new_cluster_probs

    def _compute_new_avgs(self):
        return np.array([self._compute_cluster_cluster_avgs(c_idx) / \
                         self._posterior_probs[:, c_idx].sum()
                         for c_idx in np.arange(self._num_clusters)])

    def _compute_cluster_cluster_avgs(self, c_idx):
        vfunc = lambda x_idx: \
            self._compute_cluster_avgs_on_each_data(c_idx, x_idx)

        return np.array(
            [vfunc(x_idx) for x_idx in np.arange(self._dataset_size)]) \
            .sum(axis=0)

    def _compute_cluster_avgs_on_each_data(self, c_idx, x_idx):
        return self._posterior_probs[x_idx, c_idx] \
               * self._train_X[x_idx, :]

    '''
    # def _compute_new_covariances(self):
    #     return np.array([self._compute_new_cluster_covariance(c_idx)
    #                      for c_idx in np.arange(self._num_clusters)])
    #
    # def _compute_new_cluster_covariance(self, c_idx):
    #     diff_sum_func = lambda x_idx: \
    #         self._posterior_probs[x_idx, c_idx] * \
    #         np.dot(self._train_X[x_idx, :] - self._new_avgs[c_idx, :],
    #                (self._train_X[x_idx, :] - self._new_avgs[c_idx, :]).T)
    #
    #     marginal = self._posterior_probs.sum(axis=0)[c_idx]
    #
    #     diff_sum = np.array([diff_sum_func(x_idx) / marginal
    #                          for x_idx in np.arange(self._dataset_size)]).sum(axis=0)
    #
    #     return diff_sum
    '''

    def _compute_new_covariances(self):
        return np.array([self._compute_new_cluster_covariance(c_idx)
                         for c_idx in np.arange(self._num_clusters)])

    def _compute_new_cluster_covariance(self, c_idx):
        diff = lambda x_idx: (self._train_X[x_idx, :] - self._new_avgs[c_idx, :]) \
            .reshape(self._dimension, 1)

        diff_sum_func = lambda x_idx: \
            self._posterior_probs[x_idx, c_idx] * \
            np.dot(diff(x_idx), diff(x_idx).T)

        marginal = self._posterior_probs.sum(axis=0)[c_idx]

        diff_sum = np.array([diff_sum_func(x_idx)
                             for x_idx in np.arange(self._dataset_size)]).sum(axis=0)

        return diff_sum / marginal


    def _compute_new_cluster_prob(self):
        return np.apply_along_axis(
            lambda col: col.sum() / self._dataset_size,
            0,
            self._posterior_probs).reshape(1, self._num_clusters)