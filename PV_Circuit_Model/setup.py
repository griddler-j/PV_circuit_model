from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# compile_args = ["/O2"]
compile_args = ["/Zi", "/Od"]
link_args = ["/DEBUG"]

if sys.platform == "win32":
    # VS 2022: /openmp or /openmp:llvm (either should work)
    compile_args.append("/openmp")
else:
    compile_args += ["-O3", "-fopenmp"]
    link_args.append("-fopenmp")

ext = Extension(
    name="ivkernel",
    sources=["ivkernel_wrapper.pyx", "ivkernel.cpp"],
    language="c++",
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="ivkernel",
    ext_modules=cythonize([ext]),
)

