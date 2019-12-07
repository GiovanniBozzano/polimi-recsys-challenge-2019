import numpy
from distutils.core import setup
from Cython.Build import cythonize

ext_options = {'compiler_directives': {'profile': True, 'language_level': '3'}, 'annotate': True}
setup(ext_modules=cythonize(['./recommenders/slim_bpr_epoch.pyx'],
                            **ext_options),
      script_args=['build'],
      options={'build': {'build_lib': './recommenders'}},
      include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize(['./lib/similarity/compute_similarity_cython.pyx'],
                            **ext_options),
      script_args=['build'],
      options={'build': {'build_lib': './lib/similarity'}},
      include_dirs=[numpy.get_include()])
