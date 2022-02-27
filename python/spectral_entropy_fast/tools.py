import numpy as np
import scipy.stats
from typing import Union
from .tools_fast import centroid_spectrum


def clean_spectrum(spectrum: np.ndarray,
                   max_mz: float = None,
                   noise_threshold: float = 0.01,
                   ms2_ppm: float = None, ms2_da: float = 0.05) -> np.ndarray:
    """
    Clean the spectrum with the following steps:
    1. Remove the peaks have m/z higher than the max_mz. This step can be used for
       remove precursor ions.
    2. Centroid the spectrum, merge the peaks within the +/- ms2_ppm or +/- ms2_da, sort the result spectrum by m/z.
    3. Remove the peaks with intensity less than the noise_threshold * maximum (intensity).
    4. Normalize the intensity to sum to 1.

    At least one of the ms2_ppm or ms2_da need be not None. If both ms2_da and ms2_ppm is given, ms2_da will be used.

    :param spectrum: The spectrum.
    :param max_mz: The maximum m/z to keep, if None, all the peaks will be kept.
    :param noise_threshold: The noise threshold, peaks have intensity lower than
                            noise_threshold * maximum (intensity) will be removed.
                            If None, all the peaks will be kept.
    :param ms2_ppm: The mass accuracy in ppm.
    :param ms2_da: The mass accuracy in Da.
    :return: The cleaned spectrum.
    """
    # Check the input.
    if ms2_ppm is None and ms2_da is None:
        raise RuntimeError("Either ms2_ppm or ms2_da should be given!")

    # Convert the spectrum to numpy array.
    spectrum = convert_spectrum_to_numpy_array(spectrum)

    # 1. Remove the peaks have m/z higher than the max_mz.
    if max_mz is not None:
        spectrum = spectrum[spectrum[:, 0] <= max_mz]

    # Sort spectrum by m/z.
    spectrum = spectrum[np.argsort(spectrum[:, 0])]
    # 2. Centroid the spectrum, merge the peaks within the +/- ms2_ppm or +/- ms2_da.
    spectrum = centroid_spectrum(spectrum, ms2_ppm=ms2_ppm, ms2_da=ms2_da)

    # 3. Remove the peaks with intensity less than the noise_threshold * maximum (intensity).
    if noise_threshold is not None and spectrum.shape[0] > 0:
        spectrum = spectrum[spectrum[:, 1] >= noise_threshold * np.max(spectrum[:, 1])]

    # 4. Normalize the intensity to sum to 1.
    spectrum_sum = np.sum(spectrum[:, 1])
    if spectrum_sum == 0:
        return spectrum
    else:
        spectrum[:, 1] /= spectrum_sum
        return spectrum


def convert_spectrum_to_numpy_array(spectrum: Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert the spectrum to numpy array.
    """
    spectrum = np.asarray(spectrum, dtype=np.float32, order="C")
    if spectrum.shape[0] == 0:
        return np.zeros(0, dtype=np.float32, order="C").reshape(-1, 2)
    if spectrum.ndim != 2:
        raise RuntimeError("Error in input spectrum format!")
    return spectrum
