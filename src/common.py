"""Shared loading, splitting, and model fitting for the analysis scripts."""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path(__file__).resolve().parents[1] / "data" / "model" / "dataset.parquet"
FIGURES = Path(__file__).resolve().parents[1] / "report" / "figures"
SPLIT = "2024-10-01"

FEATURES = ["moneyness", "days_to_expiry", "is_call", "log_option_volume",
            "log_stock_price", "log_stock_volume", "stock_vol"]

INK = "#333333"


def plot_style():
    """Grayscale CMU Serif style used by all report figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["CMU Serif"],
                         "mathtext.fontset": "cm", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK,
                         "axes.unicode_minus": False})  # CMU lacks the glyph


def load_split():
    """Same time split as train.py."""
    df = pd.read_parquet(DATA)
    return df[df.date < SPLIT], df[df.date >= SPLIT]


def fit_gbdt(train, **kwargs):
    params = dict(max_iter=300, random_state=42)
    params.update(kwargs)
    gbdt = HistGradientBoostingRegressor(**params)
    gbdt.fit(train[FEATURES], train.rel_spread)
    return gbdt
