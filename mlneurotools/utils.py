"""collection of useful tools that help with setting up a pipeline

Author: Arthur Dehgan"""
import time
import functools
import numpy as np
from scipy.io import loadmat
from .stats import rm_outliers


def compute_relatives(cond1, cond2, **kwargs):
    """Computes the relative changes.

    Parameters
    ----------
    cond1, cond2 : array
        Arrays of shape (n_subject x n_eletrodes) or (n_trials x n_electrodes). The arrays of data
        for the conditions.

    Returns
    -------
    values : list
        The calculated relative changes

    """
    cond1 = np.asarray(cond1).mean(axis=0)
    cond2 = np.asarray(cond2).mean(axis=0)
    values = (cond1 - cond2) / cond2
    return values


def proper_loadmat(file_path):
    """Loads using scipy.io.loadmat, and cleans some of the metadata"""
    data = loadmat(file_path)
    clean_data = {}
    for key, value in data.items():
        if not key.startswith("__"):
            clean_data[key] = value.squeeze().tolist()
    return clean_data


def timer(func):
    """Decorator to compute time spend for the wrapped function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        val = func(*args, **kwargs)
        time_diff = elapsed_time(start_time, time.perf_counter())
        print('"{}" executed in {}'.format(func.__name__, time_diff))
        return val

    return wrapper


def create_groups(y):
    """Generate groups from labels of shape (subject x labels)."""
    k = 0
    y = np.asarray(list(map(np.ravel, y)))
    y = np.asarray(list(map(np.asarray, y)))
    groups = []
    for sub in y:
        for _ in range(len(sub.ravel())):
            groups.append(k)
        k += 1
    groups = np.asarray(groups).ravel()
    y = np.concatenate([lab.ravel() for lab in y], axis=0).ravel()
    return y, groups


def prepare_data(
    data, labels=None, n_trials=None, rm_outl=None, random_state=0, zscore=False
):
    """prepares the data to be used in your ml pipeline.

    The function can return the given data after removing outliers, randomly selecting trials
    to balance subjects, and zscoring. It can also generate a labels and groups list.

    Parameters
    ----------
    data : list of arrays
        The data to prepare. Each entry of the list must be an array
        of shape (n_trials, n_elecs, n_samples)

    labels : list, optional
        The labels of the groups, the list must be of the same length as the data list, and
        indicate the label of each array. You need to set labels if n_trials is not set.

    n_trials : int, optional
        The number of trials to pick at random for each array in the data list. You need to set
        n_trials if labels is not set.

    rm_outl : int, optional
        The number of standard deviation away from the mean you want to keep. For example if
        rm_outl=3, then all the subjects that have a mean that is strictly superior or inferior to
        3 times the std + the mean of all subjectswill be deleted. If rm_outl is None, no outlier
        removal will be done.

    random_state : int, optional
        The random_state for the random selection of trials. Not used if n_trials is None. You
        need to change random_state if you want to bootstrap and repeat the random selection
        multiple times or it will select the same subsets of trials.

    zscore : bool, optional, default=False
        Will zscore the data for each group if set to True.

    """
    final_data = None
    if rm_outl is not None:
        data = np.asarray([rm_outliers(sub, rm_outl) for sub in data])

    sizes = [len(sub) for sub in data]
    if n_trials is not None:
        n_sub_min = min(sizes)
        if n_trials > n_sub_min:
            print(
                "can't take {} trials, taking the minimum amout {} instead".format(
                    n_trials, n_sub_min
                )
            )
            n_trials = n_sub_min

        labels = np.asarray([[lab] * n_trials for lab in labels])
    elif labels is not None:
        labels = np.asarray([[labels[i]] * size for i, size in enumerate(sizes)])
    else:
        raise Exception(
            "Error: either specify a number of trials and the "
            + "labels will be generated or give the original labels"
        )
    labels, groups = create_groups(labels)

    for submat in data:
        if submat.shape[0] == 1:
            submat = submat.ravel()
        if n_trials is not None:
            index = np.random.RandomState(random_state).choice(
                range(len(submat)), n_trials, replace=False
            )
            prep_submat = submat[index]
        else:
            prep_submat = submat

        if zscore:
            prep_submat = zscore(prep_submat)

        final_data = (
            prep_submat
            if final_data is None
            else np.concatenate((prep_submat, final_data))
        )
    return np.asarray(final_data), labels, groups


def elapsed_time(t0, t1, formating=True):
    """Time lapsed between t0 and t1.

    Returns the time (from time.time()) between t0 and t1 in a
    more readable fashion.

    Parameters
    ----------
    t0: float
        time.time() initial measure of time
        (eg. at the begining of the script)
    t1: float
        time.time() time at the end of the script
        or the execution of a function.

    """
    lapsed = abs(t1 - t0)
    if formating:
        m, h, j = 60, 3600, 24 * 3600
        nbj = lapsed // j
        nbh = (lapsed - j * nbj) // h
        nbm = (lapsed - j * nbj - h * nbh) // m
        nbs = lapsed - j * nbj - h * nbh - m * nbm
        if lapsed > j:
            formated_time = "{:.0f}j, {:.0f}h:{:.0f}m:{:.0f}s".format(
                nbj, nbh, nbm, nbs
            )
        elif lapsed > h:
            formated_time = "{:.0f}h:{:.0f}m:{:.0f}s".format(nbh, nbm, nbs)
        elif lapsed > m:
            formated_time = "{:.0f}m:{:.0f}s".format(nbm, nbs)
        else:
            formated_time = "{:.4f}s".format(nbs)
        return formated_time
    return lapsed
