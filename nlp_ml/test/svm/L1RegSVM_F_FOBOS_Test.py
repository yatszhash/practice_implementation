# coding: utf-8

# In[119]:


# In[120]:


# In[121]:

from sklearn import cross_validation
from  sklearn import datasets
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # In[123]:

    iris = datasets.load_iris()

    # In[124]:

    type(iris)

    # In[125]:

    iris.keys()

    # In[126]:

    iris_X = iris.data[:100]
    # Don't forget standarization or normalization
    ss = StandardScaler()
    iris_X = ss.fit_transform(iris_X)
    # iris_X = np.repeat(iris_X, 10, axis=0)
    # In[127]:

    iris_X[:10]

    # In[128]:

    iris_Y = iris.target[:100]
    # iris_Y = np.repeat(iris_Y, 10, axis=0)

    # In[129]:

    iris_Y = iris_Y.reshape(len(iris_Y), 1)

    # In[130]:

    train_x, test_x, train_y, test_y = \
        cross_validation.train_test_split(iris_X, iris_Y, test_size=0.2,
                                          random_state=20)

    # In[131]:

    import sys

    # In[132]:

    sys.path.append("../")

    # In[133]:

    import seq_labeling.svm.L1RegSVM as svm

    # In[134]:

    import importlib

    # In[135]:

    importlib.reload(svm)

    # In[136]:

    clf = svm.L1RegSVM(eta=0.1, c=1.0)

    # In[137]:

    train_x[:10], train_y[:10]

    # In[138]:

    clf.fit(train_x, train_y, method="F-FOBOS")

    # In[139]:

    print(clf.predict(train_x))
    print(train_y.reshape((1, len(train_y))))

    # In[140]:

    result = clf.predict(test_x)

    # In[141]:
    print("w: {}".format(clf.w))
    print(result)

    import sklearn.metrics as metrics

    report = metrics.classification_report(test_y, result.reshape(len(result), 1))
    print(report)
