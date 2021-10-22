[![DOI](https://zenodo.org/badge/232434019.svg)](https://zenodo.org/badge/latestdoi/232434019)
[![Python Package using Conda](https://github.com/YuanyueLi/SpectralEntropy/actions/workflows/python-package-conda.yml/badge.svg?branch=master)](https://github.com/YuanyueLi/SpectralEntropy/actions/workflows/python-package-conda.yml)
[![Python package](https://github.com/YuanyueLi/SpectralEntropy/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/YuanyueLi/SpectralEntropy/actions/workflows/python-package.yml)

# Search spectra with entropy similarity

To search spectral files with entropy similarity, you can download pre-compiled program from [https://github.com/YuanyueLi/EntropySearch/releases](https://github.com/YuanyueLi/EntropySearch/releases).

For advanced user who want to calculate spectral entropy / entropy similarity / other spectral similarity by themself, please use the Python code below.

A jupyter notebook example is provided here: [https://github.com/YuanyueLi/SpectralEntropy/blob/master/example.ipynb](https://github.com/YuanyueLi/SpectralEntropy/blob/master/example.ipynb)

The detailed reference for using the 43 different algorithm to calculate spectral similarity can be found here: [https://SpectralEntropy.readthedocs.io/en/master/](https://SpectralEntropy.readthedocs.io/en/master/) 

# Requirement

Python 3.7, numpy>=1.17.4, scipy>=1.3.2

cython>=0.29.13 (Not required but highly recommended)

```bash
# The command below is not required but strongly recommended, as it will compile the cython code to run faster
python setup.py build_ext --inplace
```

# Spectral entropy

To calculate spectral entropy, the spectrum need to be centroid first.
When you are focusing on fragment ion's information, the precursor ion may need to be removed from the spectrum before calculating spectral entropy.
If isotope peak exitsted on the MS/MS spectrum, the isotope peak should be removed fist as the isotope peak does not contain useful information for identifing molecule.

Calculate spectral entropy for **centroid** spectrum with python is very simple (just one line with scipy package).

```python
import numpy as np
import scipy.stats

spectrum = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]], dtype=np.float32)

entropy = scipy.stats.entropy(spectrum[:, 1])
print("Spectral entropy is {}.".format(entropy))
# The output should be: Spectral entropy is 0.3737888038158417.
print('-' * 30)
```

For **profile** spectrum which haven't been centroid, you can use a ```clean_spectrum``` to centroid the spectrum, for
example:

```python
import numpy as np
import scipy.stats
import spectral_entropy

spectrum = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]], dtype=np.float32)

spectrum = spectral_entropy.clean_spectrum(spectrum)
entropy = scipy.stats.entropy(spectrum[:, 1])
print("Spectral entropy is {}.".format(entropy))
# The output should be: Entropy similarity:0.2605222463607788.
print('-' * 30)
```


We provide a function  ```clean_spectrum``` to help you remove precursor ion, centroid spectrum and remove noise ions.
Please note that this function will not remove the isotope peak, you need to remove the isotope peak by yourself.
For example:

```python
import numpy as np
import spectral_entropy

spectrum = np.array([[41.04, 0.3716], [69.071, 7.917962], [69.071, 100.], [86.0969, 66.83]], dtype=np.float32)
clean_spectrum = spectral_entropy.clean_spectrum(spectrum,
                                                 max_mz=85,
                                                 noise_removal=0.01,
                                                 ms2_da=0.05)
print("Clean spectrum will be:{}".format(clean_spectrum))
# The output should be: Clean spectrum will be:[[69.071  1.   ]]
print('-' * 30)
```

# Entropy similarity

Before calculate entropy similarity, the spectrum need to be centroid first. Remove the noise ions is highly recommend.
Also, base on our test on NIST20 and Massbank.us database, remove ions have m/z higher than precursor ion's m/z - 1.6
will greatly improve the spectral identification performance.

We provide ```calculate_entropy_similarity``` function to calculate two spectral entropy.

```python
import numpy as np
import spectral_entropy

spec_query = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]], dtype=np.float32)
spec_reference = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]], dtype=np.float32)

# Calculate entropy similarity.
similarity = spectral_entropy.calculate_entropy_similarity(spec_query, spec_reference, ms2_da=0.05)
print("Entropy similarity:{}.".format(similarity))
# The output should be: Entropy similarity:0.8984397722577456.
print('-' * 30)
```

# Spectral similarity
We also provide 43 different spectral similarity algorithm for MS/MS spectral comparison

You can find the detail reference
here: [https://SpectralEntropy.readthedocs.io/en/master/](https://SpectralEntropy.readthedocs.io/en/master/)

# Example code

Before calculating spectral similarity, it's highly recommended to remove spectral noise. For example, peaks have
intensity less than 1% maximum intensity can be removed to improve identificaiton performance.

```python
import numpy as np
import spectral_entropy

spec_query = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]], dtype=np.float32)
spec_reference = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]], dtype=np.float32)

# Calculate entropy similarity.
similarity = spectral_entropy.similarity(spec_query, spec_reference, method="entropy",
                                         ms2_da=0.05)
print("Entropy similarity:{}.".format(similarity))
# The output should be: Entropy similarity:0.8984397722577456.
print('-' * 30)

# Calculate unweighted entropy similarity.
similarity = spectral_entropy.similarity(spec_query, spec_reference, method="unweighted_entropy",
                                         ms2_da=0.05)
print("Unweighted entropy similarity:{}.".format(similarity))
# The output should be: Unweighted entropy similarity:0.9826668790176113.
print('-' * 30)

# Calculate all similarity.
all_dist = spectral_entropy.all_similarity(spec_query, spec_reference, ms2_da=0.05)
for dist_name in all_dist:
    method_name = spectral_entropy.methods_name[dist_name]
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
