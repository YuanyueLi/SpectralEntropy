import numpy as np
from typing import Union

from . import tools
import scipy.stats


def calculate_entropy(spectrum: Union[list, np.ndarray],
                      max_mz: float = None,
                      noise_removal: float = 0.01,
                      ms2_da: float = 0.05, ms2_ppm: float = None):
    """
    The spectrum will be cleaned with the procedures below. Then, the spectral entropy will be returned.

    1. Remove ions have m/z higher than a given m/z (defined as max_mz).
    2. Centroid peaks by merging peaks within a given m/z (defined as ms2_da or ms2_ppm).
    3. Remove ions have intensity lower than max intensity * fixed value (defined as noise_removal)

    :param spectrum: The input spectrum, need to be in 2-D list or 2-D numpy array
    :param max_mz: The ions with m/z higher than max_mz will be removed.
    :param noise_removal: The ions with intensity lower than max ion's intensity * noise_removal will be removed.
    :param ms2_da: The MS/MS tolerance in Da.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    If both ms2_da and ms2_ppm is given, ms2_da will be used.
    """
    spectrum = tools.clean_spectrum(spectrum,
                                    max_mz=max_mz, noise_removal=noise_removal, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
    return scipy.stats.entropy(spectrum[:, 1])


def calculate_entropy_similarity(spectrum_a: Union[list, np.ndarray], spectrum_b: Union[list, np.ndarray],
                                 ms2_da: float = None, ms2_ppm: float = None,
                                 need_clean_spectra: bool = True, noise_removal: float = 0.01):
    if need_clean_spectra:
        spectrum_a = tools.clean_spectrum(spectrum_a, noise_removal=noise_removal, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        spectrum_b = tools.clean_spectrum(spectrum_b, noise_removal=noise_removal, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    else:
        spectrum_a = tools.check_spectrum(spectrum_a)
        spectrum_a = tools.standardize_spectrum(spectrum_a)
        spectrum_b = tools.check_spectrum(spectrum_b)
        spectrum_b = tools.standardize_spectrum(spectrum_b)

    spec_matched = tools.match_peaks_in_spectra(spec_a=spectrum_a, spec_b=spectrum_b, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    return _entropy_similarity(spec_matched[:, 1], spec_matched[:, 2])


def _entropy_similarity(a, b):
    entropy_a, a = _get_entropy_and_weighted_intensity(a)
    entropy_b, b = _get_entropy_and_weighted_intensity(b)

    entropy_merged = scipy.stats.entropy(a + b)
    return 1 - (2 * entropy_merged - entropy_a - entropy_b) / np.log(4)


def _get_entropy_and_weighted_intensity(intensity):
    spectral_entropy = scipy.stats.entropy(intensity)
    if spectral_entropy < 3:
        weight = 0.25 + 0.25 * spectral_entropy
        weighted_intensity = np.power(intensity, weight)
        intensity_sum = np.sum(weighted_intensity)
        weighted_intensity /= intensity_sum
        spectral_entropy = scipy.stats.entropy(weighted_intensity)
        return spectral_entropy, weighted_intensity
    else:
        return spectral_entropy, intensity
