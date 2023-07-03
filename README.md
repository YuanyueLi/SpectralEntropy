# Spectral entropy

[![DOI](https://zenodo.org/badge/232434019.svg)](https://zenodo.org/badge/latestdoi/232434019)
[![Python Package using Conda](https://github.com/YuanyueLi/SpectralEntropy/actions/workflows/python-package-conda.yml/badge.svg?branch=master)](https://github.com/YuanyueLi/SpectralEntropy/actions/workflows/python-package-conda.yml)
[![Python package](https://github.com/YuanyueLi/SpectralEntropy/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/YuanyueLi/SpectralEntropy/actions/workflows/python-package.yml)

This repository contains the original source code for the paper:

> Li, Y., Kind, T., Folz, J. _et al._ Spectral entropy outperforms MS/MS dot product similarity for small-molecule compound identification. _Nat Methods_ **18**, 1524–1531 (2021). https://doi.org/10.1038/s41592-021-01331-z

If you find our work useful, please consider citing our paper.

We are constantly improving our code and adding new features. Currently, our package includes spectral entropy, entropy similarity, and many other functions. These are all integrated into the [MSEntropy package (https://github.com/YuanyueLi/MSEntropy)](https://github.com/YuanyueLi/MSEntropy).

We recommend using the [MSEntropy package](https://github.com/YuanyueLi/MSEntropy) to access the most recent version of our code. With the `MSEntropy` package, the method for calculating entropy similarity has been rewritten using the Flash entropy search algorithm. This has resulted in speed improvements without compromising accuracy.

The MSEntropy package supports multiple languages, including `Python`, `R`, `C/C++`, and `JavaScript`.

In addition, we provide a standalone GUI named [Entropy Search (https://github.com/YuanyueLi/EntropySearch)](https://github.com/YuanyueLi/EntropySearch) for comparing spectral files or searching a spectral file against a spectral library using entropy similarity. The GUI supports `.mgf`, `.msp`, `.mzML`, and `.lbm2` file formats.

------------------------------------------------------------------------

## Ways to Calculate Spectral Entropy and Entropy Similarity

There are several ways you can calculate spectral entropy and entropy similarity, either through our GUI or by integrating our package into your code.

### Using the GUI

Our GUI provides a user-friendly way to visualize and calculate entropy similarity:

- For a straightforward approach to real-time visualize and calculate entropy similarity for two MS/MS spectra, use the [MS Viewer web app](https://yuanyueli.github.io/MSViewer).

- To search one spectral file against another spectral file or a spectral library, use the [Entropy Search GUI](https://github.com/YuanyueLi/EntropySearch). The GUI supports `.mgf`, `.msp`, `.mzML`, and `.lbm2` file formats.

### Coding with Our Package

If you prefer to integrate our tools directly into your code, visit the [MSEntropy repository](https://github.com/YuanyueLi/MSEntropy) for the latest version of our code.

- To calculate spectral entropy or entropy similarity:

  - **Python** users: use the [`ms-entropy` package](https://pypi.org/project/ms-entropy/). Find the documentation [here](https://msentropy.readthedocs.io/).

  - **R** users: use the [`msentropy` package](https://cran.r-project.org/web/packages/msentropy/index.html). Documentation is available [here](https://cran.r-project.org/web/packages/msentropy/msentropy.pdf).

  - **C/C++** users: refer to the examples in the [languages/c folder of `MSEntropy` repository](https://github.com/YuanyueLi/MSEntropy/tree/main/languages/c).

  - **JavaScript** users: refer to the examples in the [languages/javascript folder of `MSEntropy` repository](https://github.com/YuanyueLi/MSEntropy/tree/main/languages/javascript).

- To use the Flash entropy search algorithm to search a spectral file against a large spectral library:

  Currently, the Flash entropy search algorithm is only available in **Python**. Use the [`ms-entropy` package](https://pypi.org/project/ms-entropy/). Find the documentation [here](https://msentropy.readthedocs.io/).

------------------------------------------------------------------------

## Brief Introduction

### For Python user

To calculate spectral entropy or entropy similarity in Python, use the `MSEntropy` package. You can download it from [PyPI (https://pypi.org/project/ms-entropy/)](https://pypi.org/project/ms-entropy/).

For detailed documentation of the `MSEntropy` package, refer to our [ReadTheDocs page https://msentropy.readthedocs.io/](https://msentropy.readthedocs.io/).

You can install the `MSEntropy` package using `pip`:

    pip install ms_entropy

### For R user

For R users, the `msentropy` package allows calculation of spectral entropy and entropy similarity. Download it from [CRAN](https://cran.r-project.org/web/packages/msentropy/index.html).

You can access the detailed documentation for the `msentropy` package [here](https://cran.r-project.org/web/packages/msentropy/msentropy.pdf).

You can install the `msentropy` package using `install.packages`:

    install.packages("msentropy")

### For C/C++ user

Source code and example code for calculating spectral entropy and entropy similarity in C/C++ can be found in the [languages/c folder of `MSEntropy` repository](https://github.com/YuanyueLi/MSEntropy/tree/main/languages/c).

### For JavaScript user

Source code and example code for calculating spectral entropy and entropy similarity in JavaScript can be found in the [languages/javascript folder of `MSEntropy` repository](https://github.com/YuanyueLi/MSEntropy/tree/main/languages/javascript).

Please note: If you encounter an entropy similarity score higher than 1 in your self-implemented code, it could be due to errors in merging peaks within MS2-tolerance. Use the code provided in our repository to avoid this issue.

------------------------------------------------------------------------

## Spectral Similarity

The code in this repository provides 43 different spectral similarity algorithms for MS/MS spectral comparison.

For detailed information about these algorithms, refer to our [documentation](https://SpectralEntropy.readthedocs.io/en/master/).

### Supported similarity algorithm list

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
