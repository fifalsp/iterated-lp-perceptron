"""Small helpers for reading and writing records."""
import json
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    """Write JSON through a temporary file so that a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def environment():
    """Versions of Python and of the numerical libraries used for a run."""
    import numpy
    import scipy
    return dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                platform=platform.platform())


def display_path(path):
    """Path relative to the repository if possible (keeps records machine independent)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_config(name):
    """Load configs/<name>.json from the repository."""
    return read_json(REPO_ROOT / 'configs' / (name + '.json'))
