from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy as np

# from distutils.core import setup

setup(
    name="SpectralEntropy",
    version="1.0",
    packages=find_packages(exclude=['*tests*']),
    python_requires='>=3.7',
    url="https://github.com/YuanyueLi/SpectralEntropy",
    license="Apache Software License 2.0",
    install_requires=[
        "numpy>=1.17.4",
        "scipy>=1.3.2",
        "cython>=0.29.13",
        "pytest",
        "pytest-cov"
    ],
    ext_modules=cythonize("spectral_entropy/tools_fast.pyx",
                          annotate=True,
                          compiler_directives={
                              'language_level': "3",
                              'cdivision': True,
                              'boundscheck': False,  # turn off bounds-checking for entire function
                              'wraparound': False  # turn off negative index wrapping for entire function
                          }),
    test_suite="tests",
    include_dirs=[np.get_include()]
)

# python setup.py build_ext --inplace
