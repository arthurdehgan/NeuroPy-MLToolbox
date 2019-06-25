""" collection of feature functions to extract signal from time series

Author: Arthur Dehgan"""
import numpy as np
from scipy.signal import welch


def computePSD(signal, window, overlap, fmin, fmax, fs):
    """Compute PSD."""
    f, psd = welch(
        signal, fs=fs, window="hamming", nperseg=window, noverlap=overlap, nfft=None
    )
    psd = np.mean(psd[(f >= fmin) * (f <= fmax)])
    return psd
