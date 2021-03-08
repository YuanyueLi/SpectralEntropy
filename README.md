[![Python Package using Conda](https://github.com/hechth/Daphnis/actions/workflows/python-package-conda.yml/badge.svg?branch=master)](https://github.com/hechth/Daphnis/actions/workflows/python-package-conda.yml)

# Spectral similarity

The similarity score for MS/MS spectral comparison

You can find the detail reference here: [https://daphnis.readthedocs.io/en/master/](https://daphnis.readthedocs.io/en/master/)

# Requirement

Python 3.7, numpy>=1.17.4, scipy>=1.3.2

cython>=0.29.13 (Not required by recommended)

```bash
# The command below is not required but strongly recommended, as it will compile the cython code to run faster
python setup.py build_ext --inplace
```

# Example code

```python
import numpy as np
import spectral_similarity

spec_query = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]], dtype=np.float32)
spec_reference = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]], dtype=np.float32)

# Get entropy distance.
print('-' * 30)
similarity = spectral_similarity.similarity(spec_query, spec_reference, method="entropy",
                                            ms2_da=0.05)
print("Entropy similarity:{}.".format(similarity))
# The output should be: Entropy similarity:0.9082203404798316.

# Get dynamic weighted entropy distance.
print('-' * 30)
similarity = spectral_similarity.similarity(spec_query, spec_reference, method="unweighted_entropy",
                                            ms2_da=0.05)
print("Unweighted entropy similarity:{}.".format(similarity))
# The output should be: Unweighted entropy similarity:0.9826668790176113.

# Get all similarity.
print('-' * 30)
all_dist = spectral_similarity.all_similarity(spec_query, spec_reference, ms2_da=0.05)
for dist_name in all_dist:
    method_name = spectral_similarity.methods_name[dist_name]
    print("Method name: {}, similarity score:{}.".format(method_name, all_dist[dist_name]))

# A list of different spectral similarity will be shown.
```

# Supported similarity algorithm list:

    "entropy": Entropy distance
    "unweighted_entropy": Unweighted entropy distance
    "euclidean": Euclidean distance
    "manhattan": Manhattan distance
    "chebyshev": Chebyshev distance
    "squared_euclidean": Squared Euclidean distance
    "fidelity": Fidelity distance
    "matusita": Matusita distance
    "squared_chord": Squared-chord distance
    "bhattacharya_1": Bhattacharya 1 distance
    "bhattacharya_2": Bhattacharya 2 distance
    "harmonic_mean": Harmonic mean distance
    "probabilistic_symmetric_chi_squared": Probabilistic symmetric χ2 distance
    "ruzicka": Ruzicka distance
    "roberts": Roberts distance
    "intersection": Intersection distance
    "motyka": Motyka distance
    "canberra": Canberra distance
    "baroni_urbani_buser": Baroni-Urbani-Buser distance
    "penrose_size": Penrose size distance
    "mean_character": Mean character distance
    "lorentzian": Lorentzian distance
    "penrose_shape": Penrose shape distance
    "clark": Clark distance
    "hellinger": Hellinger distance
    "whittaker_index_of_association": Whittaker index of association distance
    "symmetric_chi_squared": Symmetric χ2 distance
    "pearson_correlation": Pearson/Spearman Correlation Coefficient
    "improved_similarity": Improved Similarity
    "absolute_value": Absolute Value Distance
    "dot_product": Dot-Product (cosine)
    "dot_product_reverse": Reverse dot-Product (cosine)
    "spectral_contrast_angle": Spectral Contrast Angle
    "wave_hedges": Wave Hedges distance
    "cosine": Cosine distance
    "jaccard": Jaccard distance
    "dice": Dice distance
    "inner_product": Inner Product distance
    "divergence": Divergence distance
    "avg_l": Avg (L1, L∞) distance
    "vicis_symmetric_chi_squared_3": Vicis-Symmetric χ2 3 distance
    "ms_for_id_v1": MSforID distance version 1
    "ms_for_id": MSforID distance
    "weighted_dot_product": Weighted dot product distance"
