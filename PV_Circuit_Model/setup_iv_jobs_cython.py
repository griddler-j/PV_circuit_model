from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="iv_jobs_cython",
    sources=["iv_jobs_cython.pyx"],
    language="c++",
    include_dirs=[np.get_include()],
    extra_compile_args=["/O2"],       
)

setup(
    name="iv_jobs_cython",
    ext_modules=cythonize([ext]),
)

