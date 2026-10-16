"""
figures.py
----------
Shared style and helpers for the paper figures, following the Copernicus figure
guidelines: PDF with embedded fonts, one sans-serif font family (config [figures]),
panel labels (a), (b), ..., units written with exponents and a space before them
(mm month$^{-1}$), coordinates as "15° N", ranges with an en dash, sentence-case labels,
no titles inside the figure (captions are in the manuscript).

setup_style()                   rcParams for figures drawn at print size
VARIABLES                       publication name and unit of every predictor, in display order
CLIMATE_VARIABLES, STATIC_VARIABLES   the two predictor groups, in display order
label(var)                      axis or colour-bar label, e.g. "Mean temperature (°C)"
to_display_units(var, values, days)   predictor units -> display units
SCENARIOS                       label and colour of the climate scenarios (IPCC AR6 colours)
lighten(color, amount)          colour mixed with white
cm(x)                           centimetres -> inches
add_panel_label(ax, i)          "(a)", "(b)", ... at the top left of an axes
save_figure(fig, name, appendix=False)
                                writes results/mainfigs|appendix/<name>.pdf and checks its size
"""

import matplotlib as mpl
from matplotlib.colors import to_rgb

from load_config import cfg
from climate_derived import precip_mm

FIG = cfg.figures

# Publication name and unit of each predictor, in the order used in all figures:
# climate (temperature, moisture, energy), then terrain and distances.
VARIABLES = {
    "tas":        ("Mean temperature",             "°C"),
    "tasmax":     ("Maximum temperature",          "°C"),
    "tasmin":     ("Minimum temperature",          "°C"),
    "pr":         ("Precipitation",                "mm month$^{-1}$"),
    "PET":        ("Potential evapotranspiration", "mm month$^{-1}$"),
    "climbal":    ("Climatic water balance",       "mm month$^{-1}$"),
    "rsds":       ("Solar radiation",              "W m$^{-2}$"),
    "elev":       ("Elevation",                    "m"),
    "slope":      ("Slope",                        "°"),
    "aspect_sin": ("Aspect (sine)",                None),
    "aspect_cos": ("Aspect (cosine)",              None),
    "distRoads":  ("Distance to roads",            "m"),
    "distUrban":  ("Distance to urban areas",      "m"),
    "distWater":  ("Distance to water bodies",     "m"),
}
CLIMATE_VARIABLES = ["tas", "tasmax", "tasmin", "pr", "PET", "climbal", "rsds"]
STATIC_VARIABLES = [v for v in VARIABLES if v not in CLIMATE_VARIABLES]

SCENARIOS = {
    "historical": {"label": "Historical", "color": "#4d4d4d"},
    "ssp126":     {"label": cfg.scenarios.labels.ssp126, "color": "#173c66"},
    "ssp370":     {"label": cfg.scenarios.labels.ssp370, "color": "#e71d25"},
    "ssp585":     {"label": cfg.scenarios.labels.ssp585, "color": "#951b1e"},
}


def label(var):
    """Axis or colour-bar label of a predictor, e.g. 'Precipitation (mm month$^{-1}$)'."""
    name, unit = VARIABLES[var]
    return f"{name} ({unit})" if unit else name


def to_display_units(var, values, days=None):
    """Values in display units: temperatures K -> °C, pr kg m-2 s-1 -> mm month-1
    (needs the days of each value's month); all other predictors are unchanged."""
    if var in ("tas", "tasmax", "tasmin"):
        return values - 273.15
    if var == "pr":
        if days is None:
            raise ValueError("pr needs the number of days in each month.")
        return precip_mm(values, days)
    return values


def setup_style():
    """Matplotlib settings for figures drawn at print size, with embedded TrueType fonts."""
    size = FIG.font_size
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   [FIG.font],
        "mathtext.fontset":  "custom",          # units such as m$^{-2}$ in the same font
        "mathtext.rm":       FIG.font,
        "mathtext.it":       f"{FIG.font}:italic",
        "mathtext.bf":       f"{FIG.font}:bold",
        "mathtext.default":  "regular",
        "font.size":         size,
        "axes.titlesize":    size,
        "axes.labelsize":    size,
        "xtick.labelsize":   size - 1,
        "ytick.labelsize":   size - 1,
        "legend.fontsize":   size,
        "axes.linewidth":    0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "lines.linewidth":   1.0,
        "legend.frameon":    False,
        "savefig.dpi":       FIG.dpi,
        "pdf.fonttype":      42,                # embed TrueType, not Type 3
        "figure.dpi":        100,
    })


def lighten(color, amount):
    """Colour mixed with white; amount 0 = unchanged, 1 = white."""
    return tuple(1 - (1 - c) * (1 - amount) for c in to_rgb(color))


def cm(x):
    """Centimetres to inches (matplotlib figure sizes)."""
    return x / 2.54


def add_panel_label(ax, i, x=0.0, y=1.02):
    """Panel label '(a)', '(b)', ... above the top-left corner of the axes."""
    ax.text(x, y, f"({'abcdefghijklmnopqrstuvwxyz'[i]})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FIG.font_size + 1)


def save_figure(fig, name, appendix=False):
    """Writes a paper figure to results/mainfigs (or appendix) as <name>.<format>, and warns
    if the file exceeds the size limit for submission."""
    folder = cfg.paths.results.appendix_figures if appendix else cfg.paths.results.main_figures
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.{FIG.format}"
    fig.savefig(path, dpi=FIG.dpi)
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved: {path} ({size_mb:.2f} MB)")
    if size_mb > FIG.max_size_mb:
        print(f"  WARNING: {path.name} exceeds {FIG.max_size_mb} MB.")
    return path
