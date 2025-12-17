import subprocess
from pathlib import Path

__version__ = "0.3.0"

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
