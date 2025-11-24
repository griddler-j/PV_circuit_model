from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np  # <-- add this

ext = Extension(
    name="ivkernel",
    sources=["ivkernel_wrapper.pyx", "ivkernel.cpp"],
    language="c++",
    include_dirs=[np.get_include()],        # <-- add this line
    extra_compile_args=["/O2"],             # '/O2' for MSVC; '-O3' gets ignored
)

setup(
    name="ivkernel",
    ext_modules=cythonize([ext]),
)




