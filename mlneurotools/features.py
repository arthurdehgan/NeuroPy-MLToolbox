""" collection of feature functions to extract signal from time series

Author: Arthur Dehgan"""
import numpy as np
from scipy.signal import welch


def power_spectral_density(signal, window, overlap, fmin, fmax, fs, average=True):
    """Compute PSD using welch method

    Parameters
    ----------
    signal : array
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
    if average:
        return np.mean(psd[(f >= fmin) * (f <= fmax)])
    return f, psd
