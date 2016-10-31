import itertools

import numpy as np

'''
class CrfLenearChain:
    """
    this implementation doesn't work.
    it's made only to improve my understanding.
    only skelton
    """

    def __init__(self):

        # fields for training (fit)
        self.training_X = None
        self.z_coef = None
        self.labels = None
        self.w = None

        self.learning_rate = 1.0

        # fielda for transform
        self.max_prev_y_table = None

        self.feature_vector_table = None

        self.alphas = None

        # When y is the head and tail, these are not used.
        self.y_labels = None

    def get_p_y_on_x(self, weights, X, y_vec):
        self.z_coef = self.calc_z_coef(weights, X, y_vec)
        return \
            (1.0 / self.z_coef) * np.exp(weights, self.extract_feature_vec(X, y_vec))

    def extract_feature_vec(self, X, y_vec):
        feature_elements = np.array([
                                        self.extract_feature_element(y_vec[i], y_vec[i + 1])
                                        for i in range(len(y_vec) - 1)])

        return feature_elements.sum()

    def extract_feature_element(self, X, current_y, prev_y):
        # TODO method for extract
        return -1

    def calc_z_coef(self, weights, X, y_vec):
        # TODO calc with alpha
        return sum([np.log()])

    def get_grad_element(self, X, y):
        return

    def get_dot_all_label_probs_feature_vec(self, X, y_vec):
        get_element = lambda current_y, prev_y, t: \
            self.get_p_prev_y_y_on_X(X, current_y, prev_y, t) \
            * self.extract_feature_element(X, current_y, prev_y)
        get_t_th_summed = lambda t: \
            np.array([get_element(X, current_y, prev_y, t)
                      for current_y, prev_y in itertools.product(self.labels,
                                                                 self.labels)]).sum()
        T = len(X)

        # remove the begin and end of sentence
        begin = np.array([get_element(X, current_y, "B", 0)
                          for current_y in self.labels]).sum()
        summed = np.array([get_t_th_summed(t) for t in range(1, T)]).sum()
        end = np.array([get_element(X, "E", prev_y, T + 1)
                        for prev_y in self.labels]).sum()
        return summed + begin + end

    # depend on location of index
    def get_p_prev_y_y_on_X(self, weights, X, current_y, prev_y, t):
        return (1.0 / self.calc_z_coef(weights, X)) \
               * self.new_feature_vector(weights, X, current_y, prev_y, t) \
               * self.get_alpha(prev_y, t - 1) \
               * self.get_beta(current_y, t - 1)

    def new_feature_vector(self, weights, X, current_y, prev_y, t):
        pass
        return None

    # Table and list implementation
    def get_alpha(self, weights, X, current_y, t):
        # alpha("B", 0) = 0
        if t == 0:
            return self.new_feature_vector(weights, X, current_y, "B", t)

        return np.array([self.new_feature_vector(weights, X, current_y, prev_y, t)
                         for prev_y in self.labels]).sum()

    def get_beta(self, weights, X, current_y, t):

        if t == len(X):
            return self.new_feature_vector(weights, X, "E", current_y)
        return np.array([self.new_feature_vector(weights, X, next_y, current_y)
                         for next_y in self.labels]).sum()

    def gradient_decent(self, n_iter, initial_w):
        w_history = np.zeros((n_iter + 1, 1))
        w_history[0] = initial_w

        for i in range(1, n_iter + 1):
            prev_w = w_history[i - 1]
            w_history[i] = self.update_weights(prev_w)

        last_w = w_history[n_iter]
        return last_w, w_history

    def update_weights(self, weights, ):
        grad = 0
        return weights - self.learning_rate * grad
'''