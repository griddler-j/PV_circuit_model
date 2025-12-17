# setup.py (top-level)
from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

ROOT = Path(__file__).resolve().parent
PKG_NAME = "PV_Circuit_Model"
PKG_DIR = ROOT / PKG_NAME

def rel(p: Path) -> str:
    # must be relative to setup.py directory, and use forward slashes
    return p.resolve().relative_to(ROOT).as_posix()

# ----------------------------------------------------------------------------
# Build helpers
# ----------------------------------------------------------------------------

def is_windows() -> bool:
    return sys.platform.startswith("win")


def want_openmp() -> bool:
    """
    Allow disabling OpenMP if needed (e.g., some macOS/clang setups).
    Set env PV_CIRCUIT_NO_OPENMP=1 to disable.
    """
    return os.environ.get("PV_CIRCUIT_NO_OPENMP", "").strip() not in {"1", "true", "True", "YES", "yes"}


def openmp_flags():
    link_args = []
    if is_windows():
        # MSVC
        compile_args = ["/O2"]
    else:
        compile_args = ["-O3"]
    if want_openmp():
        if is_windows():
            compile_args.append("/openmp")
        else:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")
    return (compile_args,link_args)

def cpp_std_flags():
    # Conservative C++ standard flag for non-MSVC. MSVC defaults are usually OK, but we set it too.
    if is_windows():
        return ["/std:c++17"]
    return ["-std=c++17"]


def has_cython() -> bool:
    try:
        import Cython  # noqa: F401
        return True
    except Exception:
        return False


def pick_source(pyx: Path, cpp: Path) -> str:
    """
    Prefer .pyx if Cython is available and the file exists.
    Otherwise build from the generated .cpp, if present.
    """
    if pyx.exists() and has_cython():
        return rel(pyx)
    if cpp.exists():
        return rel(cpp)
    raise FileNotFoundError(f"Neither {pyx} nor {cpp} exists. Install Cython or ship the generated .cpp.")

def maybe_cythonize(exts):
    """
    Cythonize only extensions that use .pyx sources.
    If building from pre-generated .cpp, do nothing.
    """
    if not has_cython():
        return exts

    any_pyx = any(any(str(s).endswith(".pyx") for s in ext.sources) for ext in exts)
    if not any_pyx:
        return exts

    from Cython.Build import cythonize
    return cythonize(
        exts,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "initializedcheck": False,
        },
    )

def strip_openmp_flags(ext: Extension) -> None:
    def keep(arg: str) -> bool:
        s = arg.lower()
        # remove common OpenMP flags
        if "openmp" in s:
            return False
        # remove common omp link libs
        if s in {"-lomp", "-lgomp", "-liomp5"}:
            return False
        return True

    ext.extra_compile_args = [a for a in (ext.extra_compile_args or []) if keep(a)]
    ext.extra_link_args = [a for a in (ext.extra_link_args or []) if keep(a)]


class build_ext(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        try:
            import numpy
        except ModuleNotFoundError:
            # During "get_requires_for_build_*" numpy may not exist yet.
            return

        inc = numpy.get_include()

        # 1) Add to the command's include_dirs
        if inc not in self.include_dirs:
            self.include_dirs.append(inc)

        # 2) ALSO add to each extension
        for ext in self.extensions:
            if inc not in ext.include_dirs:
                ext.include_dirs.append(inc)

    def build_extensions(self):
        try:
            super().build_extensions()
        except Exception as e:
            # Only fall back if we were attempting OpenMP
            msg = str(e).lower()
            ompish = any(k in msg for k in ["openmp", "omp", "libomp", "libgomp", "vcomp", "-fopenmp", "/openmp"])
            if want_openmp() and ompish:
                print("\n[PV_Circuit_Model] OpenMP build failed; retrying without OpenMP...\n")
                for ext in self.extensions:
                    strip_openmp_flags(ext)
                super().build_extensions()
            else:
                raise

# ----------------------------------------------------------------------------
# Extensions
# ----------------------------------------------------------------------------

cflags, lflags = openmp_flags()
cflags += cpp_std_flags()

# ivkernel: Cython wrapper + C++ kernel implementation
ivkernel_sources = [
    pick_source(PKG_DIR / "ivkernel_wrapper.pyx", PKG_DIR / "ivkernel_wrapper.cpp"),
    rel(PKG_DIR / "ivkernel.cpp"),
]

# IV_jobs: Cython module (wrapper-only; generated .cpp name is IV_jobs.cpp in your old script)
iv_jobs_sources = [
    pick_source(PKG_DIR / "IV_jobs.pyx", PKG_DIR / "IV_jobs.cpp"),
]

ext_modules = [
    Extension(
        name=f"{PKG_NAME}.ivkernel",
        sources=ivkernel_sources,
        include_dirs=[], 
        language="c++",
        extra_compile_args=cflags,
        extra_link_args=lflags,
    ),
    Extension(
        name=f"{PKG_NAME}.IV_jobs",
        sources=iv_jobs_sources,
        include_dirs=[],  
        language="c++",
        extra_compile_args=cflags,
        extra_link_args=lflags,
    ),
]

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

setup(
    name=PKG_NAME,
    packages=find_packages(),
    ext_modules=maybe_cythonize(ext_modules),
    cmdclass={"build_ext": build_ext},
)
