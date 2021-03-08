from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# from distutils.core import setup

setup(
    name='Daphnis',
    packages=find_packages(exclude=['*tests*']),
    package_data={"Daphnis": ["data/*.csv"]},
    ext_modules=cythonize("spectral_similarity/tools_fast.pyx",
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
