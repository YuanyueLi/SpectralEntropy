import numpy as np
from typing import Union

try:
    from . import math_distance, ms_distance
    from .tools import clean_spectrum, match_peaks_in_spectra, normalize_distance
except:
    pass

methods_name = {
    "entropy": "Entropy distance",
    "unweighted_entropy": "Unweighted entropy distance",
    "euclidean": "Euclidean distance",
    "manhattan": "Manhattan distance",
    "chebyshev": "Chebyshev distance",
    "squared_euclidean": "Squared Euclidean distance",
    "fidelity": "Fidelity distance",
    "matusita": "Matusita distance",
    "squared_chord": "Squared-chord distance",
    "bhattacharya_1": "Bhattacharya 1 distance",
    "bhattacharya_2": "Bhattacharya 2 distance",
    "harmonic_mean": "Harmonic mean distance",
    "probabilistic_symmetric_chi_squared": "Probabilistic symmetric χ2 distance",
    "ruzicka": "Ruzicka distance",
    "roberts": "Roberts distance",
    "intersection": "Intersection distance",
    "motyka": "Motyka distance",
    "canberra": "Canberra distance",
    "baroni_urbani_buser": "Baroni-Urbani-Buser distance",
    "penrose_size": "Penrose size distance",
    "mean_character": "Mean character distance",
    "lorentzian": "Lorentzian distance",
    "penrose_shape": "Penrose shape distance",
    "clark": "Clark distance",
    "hellinger": "Hellinger distance",
    "whittaker_index_of_association": "Whittaker index of association distance",
    "symmetric_chi_squared": "Symmetric χ2 distance",
    "pearson_correlation": "Pearson/Spearman Correlation Coefficient",
    "improved_similarity": "Improved Similarity",
    "absolute_value": "Absolute Value Distance",
    "dot_product": "Dot-Product (cosine)",
    "dot_product_reverse": "Reverse dot-Product (cosine)",
    "spectral_contrast_angle": "Spectral Contrast Angle",
    "wave_hedges": "Wave Hedges distance",
    "jaccard": "Jaccard distance",
    "dice": "Dice distance",
    "inner_product": "Inner Product distance",
    "divergence": "Divergence distance",
    "avg_l": "Avg (L1, L∞) distance",
    "vicis_symmetric_chi_squared_3": "Vicis-Symmetric χ2 3 distance",
    "ms_for_id_v1": "MSforID distance version 1",
    "ms_for_id": "MSforID distance",
    "weighted_dot_product": "Weighted dot product distance",
}

methods_range = {
    "entropy": [0, np.log(4)],
    "unweighted_entropy": [0, np.log(4)],
    "absolute_value": [0, 2],
    "avg_l": [0, 1.5],
    "bhattacharya_1": [0, np.arccos(0) ** 2],
    "bhattacharya_2": [0, np.inf],
    "canberra": [0, np.inf],
    "clark": [0, np.inf],
    "divergence": [0, np.inf],
    "euclidean": [0, np.sqrt(2)],
    "hellinger": [0, np.inf],
    "improved_similarity": [0, np.inf],
    "lorentzian": [0, np.inf],
    "manhattan": [0, 2],
    "matusita": [0, np.sqrt(2)],
    "mean_character": [0, 2],
    "motyka": [-0.5, 0],
    "ms_for_id": [-np.inf, 0],
    "ms_for_id_v1": [0, np.inf],
    "pearson_correlation": [-1, 1],
    "penrose_shape": [0, np.sqrt(2)],
    "penrose_size": [0, np.inf],
    "probabilistic_symmetric_chi_squared": [0, 1],
    "similarity_index": [0, np.inf],
    "squared_chord": [0, 2],
    "squared_euclidean": [0, 2],
    "symmetric_chi_squared": [0, 0.5 * np.sqrt(2)],
    "vicis_symmetric_chi_squared_3": [0, 2],
    "wave_hedges": [0, np.inf],
    "whittaker_index_of_association": [0, np.inf]
}


def similarity(spectrum_query: Union[list, np.ndarray], spectrum_library: Union[list, np.ndarray], method: str,
               ms2_ppm: float = None, ms2_da: float = None,
               need_clean_spectra: bool = True, need_normalize_result: bool = True) -> float:
    """
    Calculate the similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param method: Supported methods:
            "entropy", "unweighted_entropy", "euclidean", "manhattan", "chebyshev", "squared_euclidean", "fidelity", \
            "matusita", "squared_chord", "bhattacharya_1", "bhattacharya_2", "harmonic_mean", \
            "probabilistic_symmetric_chi_squared", "ruzicka", "roberts", "intersection", \
            "motyka", "canberra", "baroni_urbani_buser", "penrose_size", "mean_character", "lorentzian",\
            "penrose_shape", "clark", "hellinger", "whittaker_index_of_association", "symmetric_chi_squared", \
            "pearson_correlation", "improved_similarity", "absolute_value", "dot_product", "dot_product_reverse", \
            "spectral_contrast_angle", "wave_hedges", "jaccard", "dice", "inner_product", "divergence", \
            "avg_l", "vicis_symmetric_chi_squared_3", "ms_for_id_v1", "ms_for_id", "weighted_dot_product"
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Similarity between two spectra
    """
    if need_normalize_result:
        return 1 - distance(spectrum_query=spectrum_query, spectrum_library=spectrum_library, method=method,
                            need_clean_spectra=need_clean_spectra, need_normalize_result=need_normalize_result,
                            ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    else:
        return 0 - distance(spectrum_query=spectrum_query, spectrum_library=spectrum_library, method=method,
                            need_clean_spectra=need_clean_spectra, need_normalize_result=need_normalize_result,
                            ms2_ppm=ms2_ppm, ms2_da=ms2_da)


def all_similarity(spectrum_query: Union[list, np.ndarray], spectrum_library: Union[list, np.ndarray],
                   ms2_ppm: float = None, ms2_da: float = None,
                   need_clean_spectra: bool = True, need_normalize_result: bool = True) -> dict:
    """
    Calculate all the similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: A dict contains all similarity.
    """
    all_similarity_score = all_distance(spectrum_query=spectrum_query, spectrum_library=spectrum_library,
                                        need_clean_spectra=need_clean_spectra,
                                        need_normalize_result=need_normalize_result,
                                        ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    for m in all_similarity_score:
        if need_normalize_result:
            all_similarity_score[m] = 1 - all_similarity_score[m]
        else:
            all_similarity_score[m] = 0 - all_similarity_score[m]
    return all_similarity_score


def multiple_similarity(spectrum_query: Union[list, np.ndarray], spectrum_library: Union[list, np.ndarray],
                        methods: list = None,
                        ms2_ppm: float = None, ms2_da: float = None,
                        need_clean_spectra: bool = True, need_normalize_result: bool = True) -> dict:
    """
    Calculate multiple similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param methods: A list of method names.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: A dict contains all similarity.
    """
    if methods:
        result = {}
        for m in methods:
            s = similarity(spectrum_query=spectrum_query, spectrum_library=spectrum_library, method=m,
                           need_clean_spectra=need_clean_spectra, need_normalize_result=need_normalize_result,
                           ms2_ppm=ms2_ppm, ms2_da=ms2_da)
            result[m] = float(s)
        return result
    else:
        return all_similarity(spectrum_query=spectrum_query, spectrum_library=spectrum_library,
                              need_clean_spectra=need_clean_spectra, need_normalize_result=need_normalize_result,
                              ms2_ppm=ms2_ppm, ms2_da=ms2_da)


def distance(spectrum_query: Union[list, np.ndarray], spectrum_library: Union[list, np.ndarray], method: str,
             ms2_ppm: float = None, ms2_da: float = None,
             need_clean_spectra: bool = True, need_normalize_result: bool = True) -> float:
    """
    Calculate the distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param method: Supported methods:
            "entropy", "unweighted_entropy", "euclidean", "manhattan", "chebyshev", "squared_euclidean", "fidelity", \
            "matusita", "squared_chord", "bhattacharya_1", "bhattacharya_2", "harmonic_mean", \
            "probabilistic_symmetric_chi_squared", "ruzicka", "roberts", "intersection", \
            "motyka", "canberra", "baroni_urbani_buser", "penrose_size", "mean_character", "lorentzian",\
            "penrose_shape", "clark", "hellinger", "whittaker_index_of_association", "symmetric_chi_squared", \
            "pearson_correlation", "improved_similarity", "absolute_value", "dot_product", "dot_product_reverse", \
            "spectral_contrast_angle", "wave_hedges", "jaccard", "dice", "inner_product", "divergence", \
            "avg_l", "vicis_symmetric_chi_squared_3", "ms_for_id_v1", "ms_for_id", "weighted_dot_product"
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """

    if ms2_ppm is None and ms2_da is None:
        raise ValueError("MS2 tolerance need to be defined!")

    spectrum_query = np.asarray(spectrum_query, dtype=np.float32)
    spectrum_library = np.asarray(spectrum_library, dtype=np.float32)
    if need_clean_spectra:
        spectrum_query = clean_spectrum(spectrum_query, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        spectrum_library = clean_spectrum(spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da)

    # Calculate similarity
    if spectrum_query.shape[0] > 0 and spectrum_library.shape[0] > 0:
        function_name = method + "_distance"
        if hasattr(math_distance, function_name):
            f = getattr(math_distance, function_name)
            spec_matched = match_peaks_in_spectra(spec_a=spectrum_query, spec_b=spectrum_library,
                                                  ms2_ppm=ms2_ppm, ms2_da=ms2_da)
            dist = f(spec_matched[:, 1], spec_matched[:, 2])

        elif hasattr(ms_distance, function_name):
            f = getattr(ms_distance, function_name)
            dist = f(spectrum_query, spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        else:
            raise RuntimeError("Method name: {} error!".format(method))

        # Normalize result
        if need_normalize_result:
            if method not in methods_range:
                dist_range = [0, 1]
            else:
                dist_range = methods_range[method]

            dist = normalize_distance(dist, dist_range)

        return dist

    else:
        if need_normalize_result:
            return 1
        else:
            return np.inf


def all_distance(spectrum_query: Union[list, np.ndarray], spectrum_library: Union[list, np.ndarray],
                 ms2_ppm: float = None, ms2_da: float = None,
                 need_clean_spectra: bool = True, need_normalize_result: bool = True) -> dict:
    """
    Calculate the distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra

    """

    if ms2_ppm is None and ms2_da is None:
        raise ValueError("MS2 tolerance need to be defined!")
    spectrum_query = np.asarray(spectrum_query, dtype=np.float32)
    spectrum_library = np.asarray(spectrum_library, dtype=np.float32)
    if need_clean_spectra:
        spectrum_query = clean_spectrum(spectrum_query, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        spectrum_library = clean_spectrum(spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da)

    # Calculate similarity
    result = {}
    if spectrum_query.shape[0] > 0 and spectrum_library.shape[0] > 0:
        spec_matched = match_peaks_in_spectra(spec_a=spectrum_query, spec_b=spectrum_library,
                                              ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        for method in methods_name:
            function_name = method + "_distance"
            if hasattr(math_distance, function_name):
                f = getattr(math_distance, function_name)
                dist = f(spec_matched[:, 1], spec_matched[:, 2])
            elif hasattr(ms_distance, function_name):
                f = getattr(ms_distance, function_name)
                dist = f(spectrum_query, spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
            else:
                raise RuntimeError("Method name: {} error!".format(method))

            # Normalize result
            if need_normalize_result:
                if method not in methods_range:
                    dist_range = [0, 1]
                else:
                    dist_range = methods_range[method]

                dist = normalize_distance(dist, dist_range)
            result[method] = dist

    else:
        for method in methods_name:
            if need_normalize_result:
                result[method] = 1
            else:
                result[method] = np.inf
    return result


def multiple_distance(spectrum_query: Union[list, np.ndarray], spectrum_library: Union[list, np.ndarray],
                      methods: list = None,
                      ms2_ppm: float = None, ms2_da: float = None,
                      need_clean_spectra: bool = True, need_normalize_result: bool = True) -> dict:
    """
    Calculate multiple distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param methods: A list of method names.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """
    if methods:
        result = {}
        for m in methods:
            dist = distance(spectrum_query=spectrum_query, spectrum_library=spectrum_library, method=m,
                            need_clean_spectra=need_clean_spectra, need_normalize_result=need_normalize_result,
                            ms2_ppm=ms2_ppm, ms2_da=ms2_da)
            result[m] = float(dist)
        return result
    else:
        return all_distance(spectrum_query=spectrum_query, spectrum_library=spectrum_library,
                            need_clean_spectra=need_clean_spectra, need_normalize_result=need_normalize_result,
                            ms2_ppm=ms2_ppm, ms2_da=ms2_da)
