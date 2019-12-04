import numpy
from distutils.core import setup
from Cython.Build import cythonize

ext_options = {'compiler_directives': {'profile': True, 'language_level': '3'}, 'annotate': True}
setup(ext_modules=cythonize(['./cython/SLIM_BPR_Cython_Epoch.pyx', './cython/Compute_Similarity_Cython.pyx'],
                            **ext_options),
      include_dirs=[numpy.get_include()])
