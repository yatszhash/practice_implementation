import logging

from sklearn import cross_validation
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

logging.basicConfig(filename="../../../nmd_iris.log", level=logging.DEBUG)
logging.info("=========Started=======")
iris = datasets.load_iris()

X = iris['data']

ss = StandardScaler()

X = ss.fit_transform(X)

Y = iris['target']

random_state = 10

train_X, test_X, train_Y, test_Y = \
    cross_validation.train_test_split \
        (X, Y, test_size=0.2, random_state=random_state)

import numpy as np

n_classes = len(np.unique(train_Y))

cov_type = 'tied'

init_params = 'random'

max_iter = 10000

import sys

sys.path.append("../../")

import seq_labeling.clustering.ClusteringWithGMM as GMMCluster

import importlib

importlib.reload(GMMCluster)

clf = GMMCluster.ClusteringWithGMM()

clf.fit(train_X, num_clusters=n_classes, random_state=None)

Y_train_pred = clf.predict(train_X)

print(clf._posterior_probs)

import itertools

exchange_label = lambda y, labels: labels[y]

labels = np.unique(Y_train_pred.ravel())

for element in itertools.permutations(labels, 3):
    print(element)

train_accuracy = 0
actual_label = None
for element in itertools.permutations(labels, 3):
    current_ex_lab = lambda y: exchange_label(y, element)
    vfunc = np.vectorize(current_ex_lab)
    Y_train_pred_labeled = vfunc(Y_train_pred.ravel())
    temp_train_accuracy = np.mean(Y_train_pred_labeled == train_Y.ravel()) * 100
    if temp_train_accuracy > train_accuracy:
        train_accuracy = temp_train_accuracy
        actual_label = element

print("train accuracy : {}".format(train_accuracy))

Y_test_pred = clf.predict(test_X)

vfunc = np.vectorize(lambda y: exchange_label(y, actual_label))

Y_test_pred_label = vfunc(Y_test_pred.ravel())

test_accuracy = np.mean(Y_test_pred_label == test_Y.ravel()) * 100

print("test accuracy : {}".format(test_accuracy))

logging.info("============Finish======")
