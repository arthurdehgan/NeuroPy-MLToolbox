"""Functions and classes to make classification easier. It adds the ability to classify and get
AUC anc accuracy in one run of the clasification. It also contains the cross validation object
StratifiedShuffleGroupSplit

Author: Arthur Dehgan"""
from itertools import permutations, product
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from numpy.random import permutation
from joblib import Parallel, delayed
from .stats import compute_pval


def _cross_val(train_index, test_index, estimator, X, y):
    """fit and predict using the given data."""
    clf = clone(estimator)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred, y_test


def cross_val_score(estimator, cv, X, y, groups=None, n_jobs=1):
    """Computes all crossval on the chosen estimator, cross-val and dataset.

    it can be used instead of sklearn.model_selection.cross_val_score if you want both roc_auc and
    acc in one go."""
    clf = clone(estimator)
    crossv = clone(cv, safe=False)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_cross_val)(train_index, test_index, clf, X, y)
        for train_index, test_index in crossv.split(X=X, y=y, groups=groups)
    )

    AUC = not X.shape[1] > 1 and cv.n_groups > 1
    accuracy, auc_list = [], []
    for test in results:
        y_pred = test[0]
        y_test = test[1]
        acc = accuracy_score(y_test, y_pred)
        if AUC:
            auc = roc_auc_score(y_test, y_pred)
        else:
            auc = 0
        accuracy.append(acc)
        auc_list.append(auc)
    return accuracy, auc_list


def _permutations(iterable, size, limit=None):
    """Combinations Generator"""
    i = 0
    for elem in permutations(iterable, size):
        i += 1
        if limit is not None and i == limit:
            break
        yield elem


def permutation_test(estimator, cv, X, y, groups=None, n_perm=0, n_jobs=1):
    """Will do compute permutations aucs and accs."""
    acc_pscores, auc_pscores = [], []
    for _ in range(n_perm):
        perm_index = permutation(len(y))
        clf = clone(estimator)
        y_perm = y[perm_index]
        groups_perm = groups[perm_index]
        perm_acc, perm_auc = cross_val_score(clf, cv, X, y_perm, groups_perm, n_jobs)
        acc_pscores.append(np.mean(perm_acc))
        auc_pscores.append(np.mean(perm_auc))

    return acc_pscores, auc_pscores


def classification(estimator, cv, X, y, groups=None, perm=None, n_jobs=1):
    """Do a classification.

    Parameters
    ----------
    estimator : sklearn Estimator
        The estimator that will fit and be tested.
    cv : sklearn CrossValidator
        The cross-validation method that will be used to test the estimator.
    X : array
        The Data, must be of shape (n_samples x n_features).
    y : list or array
        The labels used for training and testing.
    groups : list or array, optional
        The groups for groups based cross-validations
    perm : int, optional
        The number of permutations that will be done to assert significance of the result.
        None means no permutations will be computed
    n_jobs : int, optional (default=1)
        The number of threads to use for the cross-validations. higher means faster. setting
        to -1 will use all available threads - Warning: may slow down computer. Set to -2 to
        keep a thread available for display and other tasks on the computer.

    Returns
    -------
    save : dictionnary
    The dictionnary contains all the information about the classification and the testing :
        acc_score: the mean score across all cross-validations using the
        accuracy scoring method
        auc_score: the mean score across all cross-validations using the
        roc_auc scoring method
        acc: the list of all cross-validations accuracy scores
        auc: the list of all cross-validations roc_auc scores
        if permutation is not None it also countains:
        auc_pvalue: the pvalue using roc_auc as a scoring method
        acc_pvalue: the pvalue using accuracy as a scoring method
        auc_pscores: a list of all permutation auc scores
        acc_pscores: a list of all permutation accuracy scores

    """
    y = np.asarray(y)
    X = np.asarray(X)
    if len(X) != len(y):
        raise ValueError(
            "Dimension mismatch for X and y : {}, {}".format(len(X), len(y))
        )
    if groups is not None:
        try:
            if len(y) != len(groups):
                raise ValueError("dimension mismatch for groups and y")
        except TypeError:
            print(
                "Error in classification: y or",
                "groups is not a list or similar structure",
            )
            exit()
    clf = clone(estimator)
    accuracies, aucs = cross_val_score(clf, cv, X, y, groups, n_jobs)
    acc_score = np.mean(accuracies)
    auc_score = np.mean(aucs)
    save = {
        "acc_score": [acc_score],
        "auc_score": [auc_score],
        "acc": accuracies,
        "auc": aucs,
        "n_splits": cv.get_n_splits(X, y, groups),
    }
    if perm is not None:
        acc_pscores, auc_pscores = permutation_test(clf, cv, X, y, groups, perm, n_jobs)
        acc_pvalue = compute_pval(acc_score, acc_pscores)
        auc_pvalue = compute_pval(auc_score, auc_pscores)

        save.update(
            {
                "auc_pvalue": auc_pvalue,
                "acc_pvalue": acc_pvalue,
                "auc_pscores": auc_pscores,
                "acc_pscores": acc_pscores,
            }
        )

    return save


class StratifiedShuffleGroupSplit(BaseEstimator):
    def __init__(self, n_groups, n_iter=None):
        """Pre-initialization."""
        self.n_groups = n_groups
        self.n_iter = n_iter
        self.counter = 0
        self.labels_list = []
        self.n_each = None
        self.n_labs = None
        self.labels_list = None
        self.lpgos = None
        self.indexes = None

    def _init_atributes(self, y, groups):
        """Initialization."""
        if len(y) != len(groups):
            raise Exception("Error: y and groups need to have the same length")
        if y is None:
            raise Exception("Error: y cannot be None")
        if groups is None:
            raise Exception("Error: this function requires a groups parameter")
        if self.labels_list is None:
            self.labels_list = list(set(y))
        if self.n_labs is None:
            self.n_labs = len(self.labels_list)
        assert (
            self.n_groups % self.n_labs == 0
        ), "Error: The number of groups to leave out must be a multiple of the number of classes"
        if self.n_each is None:
            self.n_each = int(self.n_groups / self.n_labs)
        if self.lpgos is None:
            lpgos, indexes = [], []
            for label in self.labels_list:
                index = np.where(y == label)[0]
                indexes.append(index)
                lpgos.append(LeavePGroupsOut(self.n_each))
            self.lpgos = lpgos
            self.indexes = np.array(indexes)

    def split(self, X, y, groups):
        """generator for splits of the data.

        Parameters
        ----------
        X : array
            The data, of shape (n_trials x n_features)
        y : list
            The labels list
        groups : list
            The groups list

        Yields
        ------
        n_splits : int
            The number of splits.
        """
        self._init_atributes(y, groups)
        y = np.asarray(y)
        groups = np.asarray(groups)
        iterators = []
        for lpgo, index in zip(self.lpgos, self.indexes):
            iterators.append(lpgo.split(index, y[index], groups[index]))
        for ite in product(*iterators):
            if self.counter == self.n_iter:
                break
            self.counter += 1
            train_idx = np.concatenate(
                [index[it[0]] for it, index in zip(ite, self.indexes)]
            )
            test_idx = np.concatenate(
                [index[it[1]] for it, index in zip(ite, self.indexes)]
            )
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups):
        """Gives the number of splits.

        Parameters
        ----------
        X : placeholder for compatibility
        y : list
            The labels list
        groups : list
            The groups list

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        self._init_atributes(y, groups)
        if self.n_iter is not None:
            return self.n_iter
        groups = np.asarray(groups)
        n = 1
        for index, lpgo in zip(self.indexes, self.lpgos):
            n *= lpgo.get_n_splits(None, None, groups[index])
        return n
