from .spectral_similarity import \
    entropy_similarity, unweighted_entropy_similarity, spectral_entropy

from .tools import clean_spectrum

from .tools_fast import apply_weight_to_intensity, \
    spectral_entropy as spectral_entropy_fast, intensity_entropy as intensity_entropy_fast, \
    spectral_entropy_log2, merged_spectral_entropy_log2, centroid_spectrum
