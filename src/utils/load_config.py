"""
load_config.py
--------------
Loads config.toml from the repository root and resolves it for use in scripts.

    from load_config import cfg
    df = pd.read_csv(cfg.paths.inputs.training_csv)   # pathlib.Path
    cfg.data.features                                  # list
    cfg.periods.historical                             # range (inclusive end)

- Directory and file paths become absolute pathlib.Path objects.
- The data directory can be overridden with the environment variable FFGT_DATA_DIR.
- [periods] entries [first, last] become range(first, last + 1).
- Tables support attribute access and remain dicts, so `Cls(**model.params)` works.
"""

import os
import tomllib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_DIR / "config.toml"


class Config(dict):
    """A dict whose keys can also be read as attributes (cfg.paths == cfg["paths"])."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"config has no key '{key}'") from None


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    return obj


def load(path=CONFIG_FILE):
    with open(path, "rb") as f:
        cfg = _wrap(tomllib.load(f))

    paths = cfg.paths
    data_dir = Path(os.environ.get("FFGT_DATA_DIR", PROJECT_DIR / paths.data_dir))
    paths.data_dir = data_dir
    paths.results_dir = PROJECT_DIR / paths.results_dir
    paths.models_dir = PROJECT_DIR / paths.models_dir
    paths.manifest = PROJECT_DIR / paths.manifest
    for section, base in [("raw", data_dir), ("inputs", data_dir), ("generated", data_dir),
                          ("results", paths.results_dir)]:
        for key, rel in paths[section].items():
            paths[section][key] = base / rel

    for key, (first, last) in cfg.periods.items():
        cfg.periods[key] = range(first, last + 1)

    return cfg


cfg = load()
