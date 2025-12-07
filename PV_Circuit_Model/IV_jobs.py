import warnings
import sys
from pathlib import Path

_FORCE_PYTHON = False

def _try_import_cython():
    """Try importing the fast extension."""
    try:
        import PV_Circuit_Model.IV_jobs_cython as iv_jobs_cython
        return iv_jobs_cython
    except Exception:
        return None

def _try_autobuild_extension(package_root):
    """
    Programmatically run: python setup.py build_ext --inplace
    """
    setup_file_names = ["setup_ivkernel.py","setup_IV_jobs.py"]
    for file in setup_file_names:
        setup_py = Path(package_root) / file
        if not setup_py.exists():
            warnings.warn("No setup.py found, cannot auto-build cython/c++.", RuntimeWarning)
            return False

        # Temporarily modify sys.argv as if running: setup.py build_ext --inplace
        old_argv = sys.argv
        sys.argv = [str(setup_py), "build_ext", "--inplace"]

        try:
            # Running setup() loads setup.py, runs build_ext, writes the .pyd
            with setup_py.open("r") as f:
                code = compile(f.read(), str(setup_py), "exec")
                exec(
                    code,
                    {
                        "__name__": "__main__",
                        "__file__": str(setup_py),
                    },
                )
        except Exception as e:
            warnings.warn(f"Programmatic build_ext failed: {e}", RuntimeWarning)
            return False

        finally:
            sys.argv = old_argv
    return True

IV_Job_Heap = None
iv_jobs_cython = None
if not _FORCE_PYTHON:
    # ---------------------------------------------------------------------
    # 1) If cython import fails, try auto-build
    # ---------------------------------------------------------------------
    iv_jobs_cython = _try_import_cython()
    if iv_jobs_cython is None:
        warnings.warn(
            "Building C++/Cython extensions (need to be done only once)...",
            RuntimeWarning
        )
        pkg_root = Path(__file__).resolve().parent
        if _try_autobuild_extension(pkg_root):
            # Try import again
            iv_jobs_cython = _try_import_cython()
            if iv_jobs_cython is not None:
                print("\n\n\nSucceeded building C++/Cython extensions!")

# ---------------------------------------------------------------------
# 2) If still missing, fall back to Python
# ---------------------------------------------------------------------
if iv_jobs_cython is not None:
    IV_Job_Heap = iv_jobs_cython.IV_Job_Heap
else:
    if not _FORCE_PYTHON:
        warnings.warn(
            "The optimized C++/Cython extensions could not be imported, "
            "even after attempting automatic build.\n"
            "Falling back to pure-Python implementation (much slower and has a tiny numerical difference).",
            RuntimeWarning
        )
    from PV_Circuit_Model.ivkernel_python import build_component_IV_python, calc_intrinsic_Si_I, IV_Job_Heap

    