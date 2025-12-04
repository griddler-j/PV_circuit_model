from setuptools import setup, Extension
from pathlib import Path
import numpy as np
import sys, io
from contextlib import contextmanager

force_cython = False
silent_build = True

@contextmanager
def silent(build_silently=False):
    if not build_silently:
        yield
        return
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

with silent(silent_build):
    here = Path(__file__).resolve().parent

    # --------------------------------------------------------------------
    # Decide whether to use the generated C++ file or Cython .pyx
    # --------------------------------------------------------------------
    pyx_file = here / "IV_jobs.pyx"
    cpp_file = here / "IV_jobs.cpp"

    # --------------------------------------------------------------------
    # Compiler flags (OpenMP, optimization, etc.)
    # --------------------------------------------------------------------
    compile_args = []
    link_args = []

    if sys.platform == "win32":
        # MSVC
        compile_args += ["/O2", "/openmp"]
    else:
        # GCC / Clang
        compile_args += ["-O3", "-fopenmp"]
        link_args.append("-fopenmp")

    sources = [str(cpp_file)]

    if not cpp_file.exists() or force_cython:
        try:
            from Cython.Build import cythonize
        except ImportError as e:
            raise RuntimeError(
                "Cython is required to run this command.\n"
                "Install with `pip install cython`."
            ) from e
            
    try_paths = ["PV_Circuit_Model.IV_jobs_cython","IV_jobs_cython"]
    for try_path in try_paths:
        try:
            if not cpp_file.exists() or force_cython:
                # This will generate ivkernel_wrapper.cpp
                temp_ext = Extension(
                name=try_path,      # MUST match your real extension name
                sources=[str(pyx_file)],
                language="c++",
                )
                cythonize(
                    [temp_ext],
                    language_level="3",
                )

            ext = Extension(
                name=try_path,
                sources=sources,
                language="c++",
                include_dirs=[np.get_include()],
                extra_compile_args=compile_args,
                extra_link_args=link_args
            )
            ext_modules = [ext]

            setup(name="IV_jobs_cython",ext_modules=ext_modules)
            break
        except:
            pass