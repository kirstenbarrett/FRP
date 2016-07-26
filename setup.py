from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

exts= [ Extension("cyfrp", ["cyfrp.pyx"], include_dirs=[np.get_include()]) ]

setup ( ext_modules=cythonize(exts) )