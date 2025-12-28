import subprocess
from pathlib import Path
import sys
import types
import importlib

__version__ = "0.6.0"

def _get_git_info():
    try:
        root = Path(__file__).resolve().parent

        # Commit hash
        git_hash = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Commit date (ISO 8601)
        git_date = subprocess.check_output(
            ["git", "-C", str(root), "show", "-s", "--format=%cI", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Dirty state
        porcelain = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        dirty = (len(porcelain) > 0)

        return git_hash, git_date, dirty

    except Exception:
        return None, None, None

__git_hash__, __git_date__, __dirty__ = _get_git_info()


def _install_lazy_module_alias(old_name: str, new_name: str) -> None:
    """
    Make `import old_name` resolve to `new_name` without importing `new_name`
    during package initialization (avoids circular imports).
    """
    if old_name in sys.modules:
        return

    proxy = types.ModuleType(old_name)
    proxy.__package__ = old_name.rsplit(".", 1)[0]

    def _load():
        real = importlib.import_module(new_name)
        sys.modules[old_name] = real  # replace proxy for future imports
        return real

    def __getattr__(attr):
        return getattr(_load(), attr)

    def __dir__():
        return dir(_load())

    proxy.__getattr__ = __getattr__
    proxy.__dir__ = __dir__

    sys.modules[old_name] = proxy


# Old module -> new module
_MODULE_ALIASES = {
    __name__ + ".cell": __name__ + ".device",
    __name__ + ".module": __name__ + ".device",
    __name__ + ".multi_junction_cell": __name__ + ".device",
    __name__ + ".cell_analysis": __name__ + ".device_analysis",
}

for old_name, new_name in _MODULE_ALIASES.items():
    _install_lazy_module_alias(old_name, new_name)