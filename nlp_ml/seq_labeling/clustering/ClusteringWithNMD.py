import numpy as np


class ClusteringWithNMD:
    '''
        clustering with normal mixture distribution
        EM algorithm is used.
    '''

    def __init__(self):
        self.max_iter = 400

        self._train_X = None
        self._train_Y = None

        self._dataset_size = None
        self._dimension = None

        self._num_clusters = None
        self._current_cluster_avgs = None
        self._current_std = None
        self._current_cluster_probs = None
        self._random_state = None

        self._posterior_probs_on_param = None

    def fit(self, X, Y, num_clusters=10, random_state=None):
        self._num_clusters = num_clusters
        self._random_state = random_state

        self._init_params()

        self._posterior_probs_on_param = np.zeros((self._dataset_size,
                                                   self._num_clusters))

    def _init_params(self):
        np.random.seed(self._random_state)

        self._init_avgs()
        self._init_std()

        self._init_cluster_probs()

    def _init_avgs(self):
        x_avgs = np.average(self._train_X, axis=0)
        x_stds = np.std(self._train_X, axis=0)

        rands = (np.random.rand(self._num_clusters,
                                self._dimension) - 0.5) * 2

        self._current_cluster_avgs = 2 * rands * x_stds[None, :] + x_avgs

    def _init_cluster_probs(self):
        self._current_cluster_probs = np.ones((self._num_clusters, 1)) \
                                      * 1 / self._num_clusters

    def _init_std(self):
        self._current_std = self._compute_new_std()

    def _update_all_posterior_probs(self):
        # self._posterior_probs_on_param =
        self._posterior_probs_on_param = np.zeros((self._dataset_size,
                                                   self._num_clusters))
        x_indices = np.arange(self._dataset_size)
        for i in x_indices:
            self._posterior_probs_on_param[i, :] \
                = self._update_posterior_prob_on_each_x(i)

    def _update_posterior_prob_on_each_x(self, x_idx):
        vfunc = np.vectorize(lambda c_idx:
                             self._compute_posterior_prob(self._train_X[x_idx, :], c_idx))
        posterior_probs = np.apply_along_axis(vfunc, 1,
                                              np.arange(self._num_clusters)
                                              .reshape(1, self._num_clusters))
        summed = posterior_probs.sum()

        return posterior_probs / summed

    def _compute_posterior_prob(self, x, c_idx):
        return self._current_cluster_probs[0, c_idx] \
               * np.exp(
            - np.linalg.norm(x - self._current_cluster_avgs[c_idx, :]) ** 2 \
            / (2 * self._current_std ** 2))

    def _update_params(self):
        new_cluster_avgs = self._comute_new_cluster_avgs()
        new_std = self._compute_new_std()
        new_cluster_probs = self._compute_new_cluster_prob()

        self._current_cluster_avgs = new_cluster_avgs
        self._current_std = new_std
        self._current_cluster_probs = new_cluster_probs

    def _comute_new_cluster_avgs(self):
        return np.array([self._compute_cluster_avgs_each_cluster(c_idx)
                         for c_idx in np.arange(self._num_clusters)])

    def _compute_cluster_avgs_each_cluster(self, c_idx):
        vfunc = lambda x_idx: \
            self._compute_cluster_avgs_on_each_data(c_idx, x_idx)

        return np.array(
            [vfunc(x_idx) for x_idx in np.arange(self._dataset_size)]).sum(axis=0)

    def _compute_cluster_avgs_on_each_data(self, c_idx, x_idx):
        return self._posterior_probs_on_param[x_idx, c_idx] \
               * ((self._train_X[x_idx, :] - self._current_cluster_avgs[c_idx, :]) \
                  / self._current_std ** 2)

    def _compute_new_std(self):
        diff_sum = lambda mc: np.sum(
            np.linalg.norm(self._train_X - mc) ** 2, axis=0)

        each_cluster_diff_sum = \
            np.apply_along_axis(diff_sum, axis=1,
                                arr=self._current_cluster_avgs)

        all_diff_sum = np.sum(each_cluster_diff_sum, axis=0)

        return np.sqrt(all_diff_sum / (self._dimension
                                       * self._dataset_size * self._num_clusters))

    def _compute_new_cluster_prob(self):
        return np.apply_along_axis(
            lambda col: col.sum() / self._dataset_size,
            1,
            self._posterior_probs_on_param)
