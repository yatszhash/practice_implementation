import numpy as np

ALMOST_ZERO = 1.0e-15


class L1RegSVM:
    def __init__(self, eta, c):
        self.eta = eta
        self.fobos_eta = eta
        self.c = c
        self.w = None
        self.initial_w = None

    def loss(self, x, y, w):
        return np.maximum(1 - y * np.dot(x, w), 0)

    # y is 1 or -1
    def hingeloss(self, dataset_x, dataset_y, w):
        size = dataset_x.shape[0]

        return np.apply_along_axis(
            lambda i: self.loss(dataset_x[i, :], dataset_y[i], w)
        ).sum()

    def object_func(self, dataset_x, dataset_y, w, c):
        # remove regulation for dummy element
        w_wo_dummy = w
        w_wo_dummy[0, 1] = 0

        object = self.hingeloss(dataset_x, dataset_y, w) \
                 + c * np.linalg.norm(w_wo_dummy, ord=1)

        return object

    def grad(self, x_k, y, w):
        # Does this -x_k * y really need minus?
        x_k = x_k.reshape(len(x_k), 1)
        y = y.reshape(len(y), 1)

        loss_term = self.get_loss_term(w, x_k, y)
        reg_term = np.array([self.c * np.sign(w_k) if w_k > ALMOST_ZERO else 0
                             for idx, w_k in enumerate(w)])
        reg_term = reg_term.reshape(len(reg_term), 1)

        return loss_term + reg_term

    def get_loss_term(self, w, x_k, y):
        loss_term = -y * x_k if np.dot(np.dot(w.T, x_k), y) < 1.0 else 0
        return loss_term

    def learn_each(self, x_k, y, w):
        return w - self.eta * self.grad(x_k, y, w)

    def learn_all(self, dataset_x, dataset_y, initial_w):
        w = initial_w

        for i in range(dataset_y.shape[0]):
            w = self.learn_each(dataset_x[i, :], dataset_y[i, :], w)

        return w

    def learn_all_with_FOBOS(self, dataset_x, dataset_y, initial_w):
        w = initial_w

        for i in range(dataset_y.shape[0]):
            w = self.update_w_with_fobos(dataset_x[i, :], dataset_y[i, :], w)

        return w

    def fit(self, train_X, train_Y, method="FOBOS"):
        X = np.c_[np.ones((train_X.shape[0], 1)), train_X]
        X = train_X
        if self.initial_w is None:
            self.initial_w = np.zeros((X.shape[1], 1))

        Y = train_Y
        if not np.setdiff1d(np.unique(Y), np.array([0, 1])).size:
            Y = self.convert_from_binary(Y)
        if method == "FOBOS":
            self.w = self.learn_all_with_FOBOS(X, Y, self.initial_w)
        else:
            self.w = self.learn_all(X, Y, self.initial_w)

    def update_w_with_fobos(self, x_k, y, w):
        x_k = x_k.reshape(len(x_k), 1)
        y = y.reshape(len(y), 1)
        w_with_loss = w - self.eta * self.get_loss_term(w, x_k, y)
        vfunc = np.vectorize(self.clip)
        new_w = vfunc(w_with_loss, self.fobos_eta * self.c)

        return new_w

    def predict(self, test_X):
        X = np.c_[np.ones((test_X.shape[0], 1)), test_X]
        X = test_X
        return np.apply_along_axis(lambda x: self.classify(x), 1, X)

    def classify(self, x):
        if np.dot(x, self.w) > 0.0:
            return 1
        return 0

    @staticmethod
    def clip(v, c):
        return np.sign(v) * max(np.abs(v) - c, 0)

    @staticmethod
    def convert_from_binary(Y):
        Y_placed = np.copy(Y)
        np.place(Y_placed, Y_placed == 0, -1)
        return Y_placed
