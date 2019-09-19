"""Collection of stat functions: ttest, binom pvalues or score thresholds etc

Author: Arthur Dehgan"""
from itertools import combinations
import numpy as np
from scipy.stats import ttest_ind, ttest_rel, binom, zscore
from scipy.special import comb
from joblib import Parallel, delayed
from sys import maxsize


def compute_pval(score, perm_scores):
    """computes pvalue of an item in a distribution)"""
    n_perm = len(perm_scores)
    pvalue = (np.sum(perm_scores >= score) + 1.0) / (n_perm + 1)
    return pvalue


def is_signif(pvalue, p=0.05):
    """Tell if condition with classifier is significative.

    Returns a boolean : True if the condition is significativeat given p
    """
    answer = False
    if pvalue <= p:
        answer = True
    return answer


def rm_outliers(data, rm_outl=2):
    zs_dat = zscore(data)
    to_keep = np.where(abs(zs_dat) < rm_outl)[0]
    return data[to_keep]


def binom_pval(score, test_set, n_classes=2):
    """Computes the pvalue of your score according to the binomial law.

    Parameters
    ----------
    score : float
        The score you want to test for significance.
    test_set : array or int
        The array used for testing, or the total number of trials in the test set.
    n_classes : int, optional (default=2)
        The number of different classes in your classification problem.
    """
    if not isinstance(test_set, int):
        test_set = len(test_set)
    return binom.sf(score * test_set, test_set, 1 / n_classes)


def binomial_chance_level(test_set, p=0.05, n_classes=2):
    """Computes the chance level according to the binomial law

    Parameters
    ----------
    test_set : array or int
        The array used for testing, or the total number of trials in the test set.
    p : float, optional (default=0.05)
        The p value you want to reach.
    n_classes : int, optional (default=2)
        The number of different classes in your classification problem.

    Returns
    -------
    score : float
        The score threshold. Any score higher or equal than this score is significant with the
        pvalue that was given in input.
    """
    if not isinstance(test_set, int):
        test_set = len(test_set)
    return binom.isf(p, test_set, 1 / n_classes) / test_set


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

    Parameters
    ----------
    cond1, cond2 : arrays
        Must be of shape (n_subject x n_eletrodes) or n_trials x n_electrodes. arrays of data
        for the independant conditions.
    n_perm : int, optionnal (default=0)
        The number of permutations to do, if n_perm = 0 then exaustive permutations will be done.
        It will take exponential time with data size.
    correction : string, optionnal (default=None)
        The choice of correction to compute pvalues, if None, no correction will be done.
        Options are 'maxstat', 'fdr', 'bonferroni', None
    method : string, optionnal (default='indep')
        Necessary only for fdr correction. Implements Benjamini/Hochberg method if 'indep' or
        Benjamini/Yekutieli if 'negcorr'.
    alpha : float, optionnal (default=0.05)
        The error rate
    equal_var : bool, optionnal (default=False)
        If the variance of the two distributions are the same. See scipy.stats.ttest_ind for more info.
    two_tailed : bool, optionnal (default=False)
        Set to True for a two-tailed ttest.
    paired : bool, optionnal (default=False)
        Set if the condition 1 and 2 are paired condition or independent.
    n_jobs : int, optionnal (default=1)
        Number of cores used to computer permutations in parallel (-1 uses all cores and will be faster)

    Returns
    -------
        tval : list
            The calculated t-statistics
        pval :
            The pvalues after permutation test and correction if selected

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

    Parameters
    ----------
    cond1, cond2 : arrays
        Must be of shape (n_subject x n_eletrodes) or n_trials x n_electrodes. arrays of data
        for the independant conditions.
    n_perm : int
        The number of permutations to do.
    function: function
        The function to execute in parallel on the data.
    equal_var : bool, optionnal (default=False)
        If the variance of the two distributions are the same. See scipy.stats.ttest_ind for more info.
    paired : bool, optionnal (default=False)
        Set if the condition 1 and 2 are paired condition or independent.
    n_jobs : int, optionnal (default=1)
        Number of cores used to computer permutations in parallel (-1 uses all cores and will be faster)

    Returns
    -------
    perm_t :
        The list of permutation t-statistics

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
        delayed(function)(full_mat, index, equal_var=equal_var, paired=paired)
        for index in perms_index
    )

    return perm_t[1:]  # the first perm is not a permutation


def compute_pvalues(tval, perm_t, two_tailed, correction):
    """computes pvalues.

    Parameters
    ----------
    tstat : list
        The computed t-statistics.
    perm_t : list
        The list of permutation t-statistics.
    two_tailed : bool
        Will activate or deactivate two-tailed ttest.
    correction : string
        The choice of correction to compute pvalues. If None, no correction will be done
        Options are 'maxstat', 'fdr', 'bonferroni', None

    Returns
    -------
    pvalues : list
        The list of pvalues after permutation test.

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

    Parameters
    ----------
    pvalues : list
        The list of pvalues.
    correction: string
        The choice of correction to compute pvalues. If None, no correction will be done
        Options are 'maxstat', 'fdr', 'bonferroni', None
    method : string
        Necessary only for fdr correction. Implements Benjamini/Hochberg method if 'indep' or
        Benjamini/Yekutieli if 'negcorr'.

    Returns
    -------
    pvalues : list
        The list of corrected pvalues

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
            cm = np.sum(1.0 / (np.arange(n_obs) + 1))
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


def _generate_conds(data, index):
    """

    Parameters
    ----------
    data : array
        Array of the concatenated condition data.
    index : list
        The permutation index to apply.

    Returns
    -------
    cond1, cond2 : array
        Generated array from cond1 and cond2, contains permuted values.

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
    # print(cond1, cond2, sep="\n")
    tval1, pval1 = ttest_perm(cond1, cond2, two_tailed=True, n_perm=1000)
    tval2, pval2 = ttest_perm(
        cond1, cond2, n_perm=1000, two_tailed=True, correction="bonferroni"
    )
    tval3, pval3 = ttest_perm(
        cond1, cond2, n_perm=1000, two_tailed=True, correction="fdr"
    )
    val, tval4, pval4 = relative_perm(cond1, cond2, n_perm=1000, two_tailed=True)
    print(pval1, pval2, pval3, pval4)
    # print(tval4, pval4, sep="\n")
