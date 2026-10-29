"""
risk_thresholds.py
------------------
Fire-susceptibility classes (config [risk_classes]): equal-area classes of a per-cell
susceptibility value, as in Verde and Zêzere (2010). The breaks are percentiles
(risk_classes.quintiles) of the values of a reference surface; they are computed once (step
05, 05_risk_maps.py) and then applied unchanged to every other surface (other months,
periods, scenarios), whose class areas can therefore differ from the reference.

The classes rank the cells of one model against each other: the breaks are percentiles of
that model's own values, so models with different probability scales (samplers, class
weights) are each split into the same class areas.

quintile_breaks(values)     breaks (ascending) at the percentiles risk_classes.quintiles
classify(values, breaks)    class of each value, 1 (lowest) to len(breaks) + 1; a value equal
                            to a break falls in the lower class
load_breaks()               the frozen breaks of each final model (written by 05_risk_maps.py)

Reference: Verde, J. C. and Zêzere, J. L.: Assessment and validation of wildfire
susceptibility and hazard in Portugal, Nat. Hazards Earth Syst. Sci., 10, 485–497,
https://doi.org/10.5194/nhess-10-485-2010, 2010.
"""

import json

import numpy as np

from load_config import cfg

BREAKS_FILE = cfg.model_training.paths.results.risk_maps / "breaks.json"


def quintile_breaks(values):
    """Breaks at the percentiles risk_classes.quintiles of the values (linear interpolation)."""
    return [float(b) for b in np.percentile(np.asarray(values, dtype=np.float64),
                                            cfg.risk_classes.quintiles)]


def classify(values, breaks):
    """Class of each value: 1 + the number of breaks below it (int8)."""
    if any(a >= b for a, b in zip(breaks, breaks[1:])):
        raise ValueError(f"Breaks are not strictly ascending: {breaks}")
    return (np.searchsorted(breaks, values, side="left") + 1).astype(np.int8)


def load_breaks():
    """{model: breaks} of the final models, from 05_risk_maps.py."""
    if not BREAKS_FILE.exists():
        raise FileNotFoundError(f"{BREAKS_FILE} not found: run 05_risk_maps.py first.")
    return {m: d["breaks"] for m, d in json.loads(BREAKS_FILE.read_text())["models"].items()}
