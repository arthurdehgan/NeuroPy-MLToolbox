"""Function do perform ttest indep with permutations

Author: Arthur Dehgan"""
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
from scipy.special import comb
from itertools import combinations
from joblib import Parallel, delayed
from sys import maxsize


def relative_perm(
    cond1,
    cond2,
    n_perm=0,
    correction="maxstat",
    method="indep",
    alpha=0.05,
    two_tailed=False,
    n_jobs=1,
):
    """compute relative changes (cond1 - cond2)/cond2
       with permuattions and correction.

    Parameters:
        cond1, cond2: numpy arrays of shape n_subject x n_eletrodes
                      or n_trials x n_electrodes. arrays of data for
                      the independant conditions.

        n_perm: int, number of permutations to do.
                If n_perm = 0 then exaustive permutations will be done.
                It will take exponential time with data size.

        correction: string, None, the choice of correction to compute
                    pvalues. If None, no correction will be done
                    Options are 'maxstat', 'fdr', 'bonferroni', None

        method : 'indep' | 'negcorr'
                Necessary only for fdr correction.
                Implements Benjamini/Hochberg method if 'indep' or
                Benjamini/Yekutieli if 'negcorr'.

        alpha: float, error rate

        two_tailed: bool, set to True if you want two-tailed ttest.

        n_jobs: int, Number of cores used to computer permutations in
                parallel (-1 uses all cores and will be faster)

    Returns:
        values: list, the calculated relative changes

        pval: pvalues after permutation test and correction if selected
    """
    _check_correction(correction)

    values = compute_relatives(cond1, cond2)

    perm_t = perm_test(cond1, cond2, n_perm, compute_relatives, n_jobs=n_jobs)

    pval = compute_pvalues(values, perm_t, two_tailed, correction=correction)

    if correction in ["bonferroni", "fdr"]:
        pval = pvalues_correction(pval, correction, method)

    return values, pval


def ttest_perm(
    cond1,
    cond2,
    n_perm=0,
    correction="maxstat",
    method="indep",
    alpha=0.05,
    equal_var=False,
    two_tailed=False,
    paired=False,
    n_jobs=1,
):
    """ttest indep with permuattions and maxstat correction

    Parameters:
        cond1, cond2: numpy arrays of shape n_subject x n_eletrodes
                      or n_trials x n_electrodes. arrays of data for
                      the independant conditions.

        n_perm: int, number of permutations to do.
                If n_perm = 0 then exaustive permutations will be done.
                It will take exponential time with data size.

        correction: string, None, the choice of correction to compute
                    pvalues. If None, no correction will be done
                    Options are 'maxstat', 'fdr', 'bonferroni', None

        method : 'indep' | 'negcorr'
                Necessary only for fdr correction.
                Implements Benjamini/Hochberg method if 'indep' or
                Benjamini/Yekutieli if 'negcorr'.

        alpha: float, error rate

        equal_var: bool, see scipy.stats.ttest_ind.

        two_tailed: bool, set to True if you want two-tailed ttest.

        n_jobs: int, Number of cores used to computer permutations in
                parallel (-1 uses all cores and will be faster)

    Returns:
        tval: list, the calculated t-statistics

        pval: pvalues after permutation test and correction if selected
    """
    _check_correction(correction)

    if paired:
        tval, _ = ttest_rel(cond1, cond2)
    else:
        tval, _ = ttest_ind(cond1, cond2, equal_var=equal_var)

    perm_t = perm_test(
        cond1, cond2, n_perm, _ttest_perm, equal_var, paired=paired, n_jobs=n_jobs
    )

    pval = compute_pvalues(tval, perm_t, two_tailed, correction=correction)

    if correction in ["bonferroni", "fdr"]:
        pval = pvalues_correction(pval, correction, method)

    return tval, pval


def perm_test(cond1, cond2, n_perm, function, equal_var=False, paired=False, n_jobs=1):
    """permutation ttest.

    Parameters:
        cond1, cond2: numpy arrays of shape n_subject x n_eletrodes
                      or n_trials x n_electrodes. arrays of data for
                      the independant conditions.

        n_perm: int, number of permutations to do, the more the better.

        function: func, the function to execute in parallel on the data.

        equal_var: bool, see scipy.stats.ttest_ind.

        n_jobs: int, Number of cores used to computer permutations in
                parallel (-1 uses all cores and will be faster)

    Returns:
        perm_t: list of permutation t-statistics
    """
    full_mat = np.concatenate((cond1, cond2), axis=0)
    n_samples = len(full_mat)
    perm_t = []
    n_comb = comb(n_samples, len(cond1))
    if np.isinf(n_comb):
        n_comb = maxsize
    else:
        n_comb = int(n_comb)

    if n_perm >= n_comb - 1:
        # print("All permutations will be done. n_perm={}".format(n_comb - 1))
        if n_perm == 0:
            print(
                "size of the dataset does not allow {}"
                + "permutations, instead".format(n_perm)
            )

        n_perm = n_comb
        print("All {} permutations will be done".format(n_perm))
    if n_perm > 9999:
        print("Warning: permutation number is very high : {}".format(n_perm))
        print("it might take a while to compute ttest on all permutations")

    perms_index = _combinations(range(n_samples), len(cond1), n_perm)
    perm_t = Parallel(n_jobs=n_jobs)(
        delayed(function)(full_mat, index, equal_var=equal_var) for index in perms_index
    )

    return perm_t[1:]  # the first perm is not a permutation


def compute_pvalues(tval, perm_t, two_tailed, correction):
    """computes pvalues.

    Parameters:
        tstat: computed t-statistics

        perm_t: list of permutation t-statistics

        two_tailed: bool, if you want two-tailed ttest.

        correction: string, None, the choice of correction to compute
                    pvalues. If None, no correction will be done
                    Options are 'maxstat', 'fdr', 'bonferroni', None

    Returns:
        pvalues: list of pvalues after permutation test
    """
    scaling = len(perm_t)
    perm_t = np.array(perm_t)
    pvalues = []
    tval_tocompare = tval
    if two_tailed:
        perm_t = abs(perm_t)
        tval_tocompare = abs(tval)

    if correction == "maxstat":
        perm_t = np.asarray(perm_t).max(axis=1)
        perm_t = np.array([perm_t for _ in range(len(tval))]).T

    for i, tstat in enumerate(tval_tocompare):
        p_final = 0
        compare_list = perm_t[:, i]
        for t_perm in compare_list:
            if tstat <= t_perm:
                p_final += 1 / scaling
        pvalues.append(p_final)

    pvalues = np.asarray(pvalues, dtype=np.float32)

    return pvalues


def pvalues_correction(pvalues, correction, method):
    """computes corrected pvalues from pvalues.

    Parameters:
        pvalues: list, list of pvalues.

        correction: string, None, the choice of correction to compute
                    pvalues. If None, no correction will be done
                    Options are 'maxstat', 'fdr', 'bonferroni', None

        method : 'indep' | 'negcorr'
                Necessary only for fdr correction.
                Implements Benjamini/Hochberg method if 'indep' or
                Benjamini/Yekutieli if 'negcorr'.

    Returns:
        pvalues: list of corrected pvalues
    """
    if correction == "bonferroni":
        pvalues *= float(np.array(pvalues).size)

    elif correction == "fdr":
        n_obs = len(pvalues)
        index_sorted_pvalues = np.argsort(pvalues)
        sorted_pvalues = pvalues[index_sorted_pvalues]
        sorted_index = index_sorted_pvalues.argsort()
        ecdf = (np.arange(n_obs) + 1) / float(n_obs)

        if method == "negcorr":
            cm = np.sum(1. / (np.arange(n_obs) + 1))
            ecdf /= cm
        elif method == "indep":
            pass
        else:
            raise ValueError(method, " is not a valid method option")

        raw_corrected_pvalues = sorted_pvalues / ecdf
        corrected_pvalues = np.minimum.accumulate(raw_corrected_pvalues[::-1])[::-1]
        pvalues = corrected_pvalues[sorted_index].reshape(n_obs)

    pvalues[pvalues > 1.0] = 1.0

    return pvalues


def compute_relatives(cond1, cond2, **kwargs):
    """Computes the relative changes.

    Parameters:
        cond1, cond2: numpy arrays of shape n_subject x n_eletrodes
                      or n_trials x n_electrodes. arrays of data for
                      the independant conditions.

    Returns:
        values: list, the calculated relative changes
    """
    cond1 = np.asarray(cond1).mean(axis=0)
    cond2 = np.asarray(cond2).mean(axis=0)
    values = (cond1 - cond2) / cond2
    return values


def _generate_conds(data, index):
    """

    Parameters:
        data: numpy array of the concatenated condition data.

        index: the permutation index to apply.

    Returns:
        cond1, cond2: numpy arrays of permutated values.
    """
    index = list(index)
    index_comp = list(set(range(len(data))) - set(index))
    perm_mat = np.vstack((data[index], data[index_comp]))
    cond1, cond2 = perm_mat[: len(index)], perm_mat[len(index) :]
    return cond1, cond2


def _combinations(iterable, r, limit=None):
    """combinations generator"""
    i = 0
    for e in combinations(iterable, r):
        yield e
        i += 1
        if limit is not None and i == limit:
            break


def _relative_perm(data, index, **kwargs):
    """Compute realtives changes after on a selectes permutation"""
    cond1, cond2 = _generate_conds(data, index)
    return compute_relatives(cond1, cond2, kwargs)


def _ttest_perm(data, index, equal_var, paired):
    """ttest with the permutation index"""
    cond1, cond2 = _generate_conds(data, index)
    if paired:
        return ttest_rel(cond1, cond2)[0]
    else:
        return ttest_ind(cond1, cond2, equal_var=equal_var)[0]


def _check_correction(correction):
    """Checks if correction is a correct option"""
    if correction not in ["maxstat", "bonferroni", "fdr", None]:
        raise ValueError(correction, "is not a valid correction option")


if __name__ == "__main__":
    cond1 = np.random.randn(10, 19)
    cond2 = np.random.randn(10, 19)
    tval, pval = ttest_perm(cond1, cond2, n_perm=100)
    tval4, pval4 = ttest_perm(cond1, cond2, n_perm=100, correction="maxstat")
    tval2, pval2 = ttest_perm(cond1, cond2, n_perm=100, correction="bonferroni")
    tval3, pval3 = ttest_perm(cond1, cond2, n_perm=100, correction="fdr")
    val, pval4 = relative_perm(cond1, cond2, n_perm=10)
    print(pval, pval2, pval4, pval3)
