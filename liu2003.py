# Most of the code is created from sklearn IncrementalPCA as reference
# For reference: https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/decomposition/incremental_pca.py
import numpy as np
from scipy import linalg
from sklearn.utils import check_array, gen_batches
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import time
from sklearn.decomposition import PCA


class IncrementalDecayPCA:
    """ Incremental Decay PCA is proposed by Liu et al. in the research titled: "Eigenspace updating for
    non-stationary process and its application to face recognition"

    Parameters
    ----------
    n_components : int or None, (default=None)
        Number of components to retain. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    whiten : bool, optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.

    alpha : int, (default = 0.9)
        Alpha is the decay parameter that controls the decay of previous data. The slower the change in the data,
        the higher the alpha value to be chosen.

    References
    ----------
    Xiaoming Liu, Tsuhan Chen, and Susan M. Thornton. Eigenspace updating for non-stationary process and its
    application to face recognition. Pattern Recognition, 36(9):1945 - 1959, 2003.
    Kernel and Subspace Methods for Computer Vision.
    """

    def __init__(self, n_components=None, alpha=0.9, batch_size=None, copy=True):
        self.n_components = n_components
        self.alpha = alpha
        self.batch_size = batch_size
        self.copy = copy

    def fit(self, X, y=None):
        """Completely fit the model with X, using minibatches of size batch_size.
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples and
                n_features is the number of features.
            y : Ignored
            Returns
            -------
            self : object
                Returns the instance itself.
        """
        self.components_ = None
        self.n_samples_seen_ = 0
        self.n_samples_seen_repeat_ = 0
        self.mean_ = .0
        self.var_ = .0
        self.first_data_ = None
        self.evals, self.evecs = None, None
        self.Q = None
        self.counter = 1

        X = check_array(X, copy=self.copy, dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in gen_batches(n_samples, self.batch_size_,
                                 min_batch_size=self.n_components or 0):
            self.partial_fit(X[batch], check_input=False)

        return self

    def partial_fit(self, X, y=None, check_input=True):
        """Incremental fit with X. All of X is processed as a single batch. In this particular algorithm,
        there is only single data point incrementation
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples and
                n_features is the number of features.
            check_input : bool
                Run check_array on X.
            y : Ignored
            Returns
            -------
            self : object
                Returns the instance itself.
        """
        if check_input:
            X = check_array(X, copy=self.copy, dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        # Error checks
        if not hasattr(self, 'components_'):
            self.components_ = None

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif not 1 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d, need "
                             "more rows than columns for IncrementalPCA "
                             "processing" % (self.n_components, n_features))
        elif not self.n_components <= n_samples:
            raise ValueError("n_components=%r must be less or equal to "
                             "the batch number of samples "
                             "%d." % (self.n_components, n_samples))
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (self.components_.shape[0] !=
                                               self.n_components_):
            raise ValueError("Number of input features has changed from %i "
                             "to %i between calls to partial_fit! Try "
                             "setting n_components to a fixed value." %
                             (self.components_.shape[0], self.n_components_))

        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self.n_samples_seen_repeat_ = 0
            self.mean_ = .0
            self.var_ = .0

        # Update statistics -- Algorithm begins from here
        new_sample_count = np.sum(~np.isnan(X), axis=0)
        updated_sample_count = self.n_samples_seen_repeat_ + new_sample_count
        updated_mean = np.add((self.alpha * self.mean_), (1 - self.alpha) * X)
        updated_variance = None

        print('Samples fitted: %d'%self.counter)

        evals, evecs = self.evals, self.evecs

        Q = self.Q

        if self.n_samples_seen_ < 2:
            if self.first_data_ is None:
                self.first_data_ = X
            else:
                B_n = np.vstack((np.sqrt(self.alpha) * np.subtract(self.first_data_, updated_mean),
                                np.subtract(X, updated_mean)))
                B_n = B_n.T

                A_n = np.dot(B_n.T, B_n)
                evals, A_evecs = linalg.eigh(A_n)
                A_evecs = normalize(A_evecs, norm='l2', axis=0)

                evals = np.flip(evals)
                A_evecs = np.fliplr(A_evecs)

                evals = evals[np.where(evals > 1)]
                Q = evals.size
                A_evecs = A_evecs[:, :Q]

                evecs = np.dot(B_n, A_evecs)
                evecs = normalize(evecs, norm='l2', axis=0)
        else:
            sqrt_alpha_evals = np.sqrt(np.multiply(self.alpha, evals)).reshape((1, Q))

            sqrt_1_minus_alpha_x = np.multiply(np.sqrt(1-self.alpha), np.subtract(X, updated_mean)).T

            B_n = evecs * sqrt_alpha_evals
            B_n = np.hstack((B_n, sqrt_1_minus_alpha_x))
            B_n = np.nan_to_num(B_n)

            A_n = np.dot(B_n.T, B_n)
            A_n = np.nan_to_num(A_n)
            evals, A_evecs = linalg.eigh(A_n)

            A_evecs = normalize(A_evecs, norm='l2', axis=0)

            evals = np.flip(evals)
            A_evecs = np.fliplr(A_evecs)

            if np.any(np.where(evals > 1)):
                evals = evals[np.where(evals > 1)]

            Q = evals.size
            if Q > min(n_features, 100) :
                Q = min(n_features, 100)

            evals = evals[:Q]
            A_evecs = A_evecs[:, :Q]

            evecs = np.dot(B_n, A_evecs)
            evecs = normalize(evecs, norm='l2', axis=0)

        # print('%d) evals evecs, X, B_n, Q'%self.counter)
        # print(evals)
        # print(evecs)
        # print(X)
        # print(B_n)
        # print(Q)
        # print('End of values:::::::')

        self.n_samples_seen_repeat_ = updated_sample_count
        self.n_samples_seen_ = updated_sample_count[0]
        self.mean_ = updated_mean
        self.var_ = updated_variance

        self.evecs = np.nan_to_num(evecs)
        self.evals = np.nan_to_num(evals)
        self.Q = Q

        self.counter = self.counter + 1

        return self

    def transform(self, X, y=None):
        """Transforms the data in X using the covariances previously computed
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples and
                n_features is the number of features.
            y : Ignored
            Returns
            -------
            X_transformed : Numpy ndarray
                Returns the transformed numpy array of the given data X in lower dimensions.
        """
        evecs = self.evecs
        X_transformed = np.dot(X, evecs)
        print('Shape of X')
        print(X.shape)
        print('Shape of X_transformed')
        print(X_transformed.shape)

        return X_transformed


def run_on_data(path):
    start_time = time.time()
    X = np.load(os.path.join(path, 'X.npy'))
    y = np.load(os.path.join(path, 'Y.npy'))

    n_splits = 5
    n_repeats = 10
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    accuracy_values = []

    count_val = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit and predict using LDA
        decaypca = IncrementalDecayPCA(batch_size=1)
        print("Fitting")
        decaypca.fit(X_train, y_train)

        print("Transforming")
        X_train_transformed = decaypca.transform(X_train, y_train)
        X_test_transformed = decaypca.transform(X_test, y_test)

        k_value = 5
        classifier = neighbors.KNeighborsClassifier(k_value)
        classifier.fit(X_train_transformed, y_train)

        y_predicted = classifier.predict(X_test_transformed)

        accuracy = accuracy_score(y_test, y_predicted)
        print("Accuracy:", accuracy)

        accuracy_values.append(accuracy)
        count_val = count_val + 1

    print('Accuracy Values:')
    print(accuracy_values)

    average_accuracy = np.average(accuracy_values)
    print('Average accuracy:')
    print(average_accuracy)

    f = open(os.path.join(path, 'decay_average_accuracy_ar.txt'), 'w')
    f.write('Average Accuracy = ' + str(average_accuracy) + '\n')
    f.close()
    print("--- Total time taken %s seconds ---" % (time.time() - start_time))


def ar_data_test_pca():
    start_time = time.time()
    path = "../Datasets/AR/"
    X = np.load(os.path.join(path, 'X.npy'))
    y = np.load(os.path.join(path, 'Y.npy'))

    n_splits = 4
    n_repeats = 40
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    accuracy_values = []

    count_val = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pca = PCA(n_components=100, whiten=True)
        print("Fitting")
        pca.fit(X=X_train, y=y_train)

        print("Transforming")
        X_train_transformed = pca.transform(X_train)
        X_test_transformed = pca.transform(X_test)

        k_value = 5
        classifier = neighbors.KNeighborsClassifier(k_value)
        classifier.fit(X_train_transformed, y_train)

        y_predicted = classifier.predict(X_test_transformed)

        accuracy = accuracy_score(y_test, y_predicted)
        print("Accuracy:", accuracy)

        accuracy_values.append(accuracy)
        count_val = count_val + 1

    print('Accuracy Values:')
    print(accuracy_values)

    average_accuracy = np.average(accuracy_values)
    print('Average accuracy:')
    print(average_accuracy)
    print("--- Total time taken %s seconds ---" % (time.time() - start_time))


def iris_data():
    # Load the dataset
    data = datasets.load_iris()
    #
    X = data.data
    y = data.target
    start_time = time.time()
    path = "../Datasets/AR/"
    # X = np.load(os.path.join(path, 'X.npy'))
    # y = np.load(os.path.join(path, 'Y.npy'))

    n_splits = 4
    n_repeats = 40
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    accuracy_values = []

    count_val = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit and predict using LDA
        decaypca = IncrementalDecayPCA(batch_size=1)
        print("Fitting")
        decaypca.fit(X_train, y_train)

        print("Transforming")
        X_train_transformed = decaypca.transform(X_train, y_train)
        X_test_transformed = decaypca.transform(X_test, y_test)

        k_value = 5
        classifier = neighbors.KNeighborsClassifier(k_value)
        classifier.fit(X_train_transformed, y_train)

        y_predicted = classifier.predict(X_test_transformed)

        accuracy = accuracy_score(y_test, y_predicted)
        print("Accuracy:", accuracy)

        accuracy_values.append(accuracy)
        count_val = count_val + 1

    print('Accuracy Values:')
    print(accuracy_values)

    average_accuracy = np.average(accuracy_values)
    print('Average accuracy:')
    print(average_accuracy)

    f = open(os.path.join(path, 'average_accuracy_iris.txt'), 'w')
    f.write('Average Accuracy on IRIS data= ' + str(average_accuracy) + '\n')
    f.close()
    print("--- Total time taken %s seconds ---" % (time.time() - start_time))
    # pca = PCA()
    # pca.plot_in_2d(X_test, y_pred, title="LDA", accuracy=accuracy)


if __name__ == "__main__":
    ar_path = "../Datasets/AR/"
    cacd_path = "../Datasets/CACD/"
    run_on_data(ar_path)
    # run_on_data(cacd_path)
    # ar_data_test_pca()
    # iris_data()
