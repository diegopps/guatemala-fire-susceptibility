"""
load_config.py
--------------
Loads config.toml from the repository root and resolves it for use in scripts.

    from load_config import cfg
    df = pd.read_csv(cfg.paths.inputs.training_csv)   # pathlib.Path
    cfg.data.features                                  # list
    cfg.periods.historical                             # range (inclusive end)

    settings = load_settings(path_to_other_settings_toml)   # same path rules

- Directory and file paths become absolute pathlib.Path objects: those of the [paths.*]
  tables and of any [<section>.paths.*] table of a section (same rules).
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


def _resolve_paths(paths, data_dir, results_dir):
    """Makes the entries of the raw/inputs/generated/results tables of `paths` absolute."""
    for section, base in [("raw", data_dir), ("inputs", data_dir), ("generated", data_dir),
                          ("results", results_dir)]:
        for key, rel in paths.get(section, {}).items():
            paths[section][key] = base / rel


def load(path=CONFIG_FILE):
    with open(path, "rb") as f:
        cfg = _wrap(tomllib.load(f))

    paths = cfg.paths
    data_dir = Path(os.environ.get("FFGT_DATA_DIR", PROJECT_DIR / paths.data_dir))
    paths.data_dir = data_dir
    paths.results_dir = PROJECT_DIR / paths.results_dir
    paths.models_dir = PROJECT_DIR / paths.models_dir
    paths.manifest = PROJECT_DIR / paths.manifest
    _resolve_paths(paths, data_dir, paths.results_dir)
    for name, table in cfg.items():
        if name != "paths" and isinstance(table, dict) and isinstance(table.get("paths"), dict):
            _resolve_paths(table.paths, data_dir, paths.results_dir)

    for key, (first, last) in cfg.periods.items():
        cfg.periods[key] = range(first, last + 1)

    return cfg


cfg = load()


def load_settings(path):
    """Loads another TOML settings file (e.g. one kept next to a script). Its
    [paths.raw|inputs|generated|results] tables are resolved like those of config.toml,
    against the same data and results directories."""
    with open(path, "rb") as f:
        settings = _wrap(tomllib.load(f))
    if isinstance(settings.get("paths"), dict):
        _resolve_paths(settings.paths, cfg.paths.data_dir, cfg.paths.results_dir)
    return settings
