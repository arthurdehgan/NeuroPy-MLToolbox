""" collection of feature functions to extract signal from time series

Author: Arthur Dehgan"""
from itertools import product, combinations
import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr


def power_spectral_density(signal, window, overlap, fmin, fmax, fs, average=True):
    """Compute PSD using welch method

    Parameters
    ----------
    signal : array of shape (n_samples,)
        The signal to compute the PSD on.
    window : int
        Length of the segments.
    overlap : float
        proportion of the segment to overlap. (eg. 0.5 will make each window have a 50% overlap
        with the previous one)
    fmin : float
        The lower bound of the frequency band that will be extracted
    fmax : float
        The upper bound of the frequency band that will be extracted
    fs : float
        The sampling frequency of the signal.
    average : bool, optional (default=True)
        Will average all frequencies and will only return one average value of PSD in the
        frequency band if True. If False, the function will return the PSD for each frequency
        bin and the associates frequency values.

    Returns
    -------
    if average is True (default)
    psd : value
        The avegare PSD in the frequency band.
    if average is False:
    f, psd : list, list
        The frequencies and their corresponding amplitudes in the signal.

    """
    f, psd = welch(
        signal,
        fs=fs,
        window="hanning",
        nperseg=window,
        noverlap=int(overlap * window),
        nfft=None,
    )
    psd = psd[(f >= fmin) * (f <= fmax)]
    if average:
        return np.mean(psd)
    return f, psd


def _compute_rsa(trial1, trial2, stat):
    """Computes the Representational Similarity Analysis (RSA) between two time
        frequency maps.

    Parameters
    ----------
    trial1, trial2 : arrays
        Must be of shape(n_freq x n_samples).
    stat : string
        The stat of correlation method used. Can be 'pearson' or 'spearman'.

    Returns
    -------
    rsa : array
        The Representational Similarity matrix of shape (n_samples x n_samples).
    """
    corr = pearsonr if stat == "pearson" else spearmanr

    assert (
        trial1.shape == trial2.shape
    ), "Error: shape of trial1 and trial2 must be the same"

    n_samples = trial1.shape[-1]

    rsa = np.zeros((n_samples, n_samples))
    for i, sample1 in enumerate(trial1.T):
        for j, sample2 in enumerate(trial2.T):
            rsa[i, j] = corr(sample1, sample2)[0]

    # return Fisher-z transform
    return np.arctanh(rsa)


def _btw_rsa_twoconds(cond1, cond2, stat):
    """Computes within rsa of the data.

    Parameters
    ----------
    cond1, cond2 : array
        the data of shape (n_freq x n_samples x n_trials).
        WARNING: ASSUMES cond1 AND cond2 HAVE THE SAME SIZE.
    stat : string
        The stat of correlation method used. Can be 'pearson' or 'spearman'.

    Returns
    -------
    rsa : array
        The Representational Similarity matrix of shape (n_samples x n_samples).
    """
    assert (
        cond1.shape == cond2.shape
    ), "Error: cond1 and cond2 should be of the same dimensions"
    n_samples, n_trials = cond1.shape[1:]
    rsa = np.zeros((n_samples, n_samples))
    counter = 0
    for trial1, trial2 in product(cond1.T, cond2.T):
        rsa += _compute_rsa(trial1.T, trial2.T, stat=stat)
        counter += 1
    return rsa / counter


def within_rsa(data, stat="pearson"):
    """Computes within rsa of the data.

    Parameters
    ----------
    data : array
        The data of shape (n_freq x n_samples x n_trials).
    stat : string
        The stat of correlation method used. Can be 'pearson' or 'spearman'.

    Returns
    -------
    rsa : array
        The Representational Similarity matrix of shape (n_samples x n_samples).
    """
    n_samples, n_trials = data.shape[1:]
    rsa = np.zeros((n_samples, n_samples))
    counter = 0
    for i, j in combinations(np.arange(n_trials), 2):
        rsa += _compute_rsa(data[:, :, i], data[:, :, j], stat=stat)
        counter += 1
    return rsa / counter


def btw_rsa(conds, stat, rep=1, n_jobs=-1):
    """

    Parameters
    ----------
    conds : list of arrays
        Each array must be of shape (n_freq x n_samples x n_trials).
    stat : string
        The stat of correlation method used. Can be 'pearson' or 'spearman'.
    rep : int
        The number of bootstrapping steps. Use if data is inbalanced.

    Returns
    -------
    rsa : array
        The Representational Similarity matrices of shape (n_channels x n_samples x n_samples).

    """
    assert len(cond1.shape) == 3, "Error: conds should have only 3 dimensions"
    n_freq, n_samples = conds[0].shape[:-1]

    # Do we need bootstrapping ?
    for _ in range(rep):
        # get the minimum number of stim in our conditions
        mini = min([cond.shape[-1] for cond in conds])

        # preallocation
        balanced_conds = [np.zeros((n_freq, n_samples, mini)) for _ in conds]

        # selecting random trials from each condition and concatenating in one matrix
        for i, cond in enumerate(conds):
            index = np.random.choice(np.arange(cond.shape[-1]), mini, replace=False)
            balanced_conds[i] = conds[i][:, :, index]

        rsa = np.zeros((n_samples, n_samples))
        counter = 0
        for i, j in combinations(np.arange(len(conds)), 2):
            rsa += _btw_rsa_twoconds(balanced_conds[i], balanced_conds[j], stat=stat)
            counter += 1
    return rsa / counter


if __name__ == "__main__":
    # PSD testing:
    print("testing PSD")
    X = np.random.rand(10000)
    psd = power_spectral_density(X, 1000, 0.5, 0, 120, 1000)
    f, psd = power_spectral_density(X, 1000, 0.5, 0, 120, 1000, average=False)

    # RSA testing:
    N_BOOT = 2
    N_FREQ = 5
    N_SAMPLES = 10
    N_TRIALS = 10
    N_TRIALS2 = 5
    N_TRIALS3 = 3
    cond1 = np.random.rand(N_FREQ, N_SAMPLES, N_TRIALS)
    cond2 = np.random.rand(N_FREQ, N_SAMPLES, N_TRIALS2)
    cond3 = np.random.rand(N_FREQ, N_SAMPLES, N_TRIALS2)
    conds = [cond1, cond2, cond3]
    for stat in ("pearson", "spearman"):
        print("testing", stat)
        btw_rsa(conds, stat)
        within_rsa(cond1, stat)
