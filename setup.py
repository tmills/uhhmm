from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [Extension("*", ["scripts/*.pyx"], 
                )]

setup(
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
