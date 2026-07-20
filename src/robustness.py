"""Robustness checks for the GBDT result.

Four checks from the milestone report: a split by ticker instead of time,
a small hyperparameter grid validated on Aug-Sep, sensitivity to the
per-day sampling rate, and partial dependence plots for the two features
whose effects looked nonlinear.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, r2_score

from common import DATA, FEATURES, FIGURES, fit_gbdt, load_split

SEED = 42


def score(name, gbdt, test):
    pred = gbdt.predict(test[FEATURES])
    print(f"{name:<44} R2 {r2_score(test.rel_spread, pred):5.3f}   "
          f"MAE {mean_absolute_error(test.rel_spread, pred):.4f}")


def ticker_split():
    # hold out 20% of underlyings entirely: does the model generalize to
    # stocks it has never seen, or is it memorizing per-name liquidity?
    df = pd.read_parquet(DATA)
    tickers = np.sort(df.ticker.unique())
    rng = np.random.default_rng(SEED)
    held_out = set(rng.choice(tickers, int(0.2 * len(tickers)), replace=False))
    test_mask = df.ticker.isin(held_out)
    train, test = df[~test_mask], df[test_mask]
    score(f"ticker split ({len(held_out)} names held out)",
          fit_gbdt(train), test)


def tuning(train, test):
    # validate on the last two train months so the test period stays untouched
    fit, val = train[train.date < "2024-08-01"], train[train.date >= "2024-08-01"]
    results = []
    for lr in [0.05, 0.1]:
        for leaves in [31, 63, 127]:
            gbdt = fit_gbdt(fit, learning_rate=lr, max_leaf_nodes=leaves,
                            max_iter=600, early_stopping=False)
            r2 = r2_score(val.rel_spread, gbdt.predict(val[FEATURES]))
            results.append((r2, lr, leaves))
            print(f"  lr {lr:<5} leaves {leaves:<4} val R2 {r2:.3f}")
    _, lr, leaves = max(results)
    print(f"best on validation: lr {lr}, leaves {leaves}")
    gbdt = fit_gbdt(train, learning_rate=lr, max_leaf_nodes=leaves,
                    max_iter=600, early_stopping=False)
    score("tuned GBDT (time split)", gbdt, test)


def sampling_sensitivity(train, test):
    # the dataset samples 5000 contracts/day; check the result is not an
    # artifact of that rate by refitting on thinner subsamples
    for per_day in [1000, 2500]:
        sub = (train.groupby("date", group_keys=False)
                    .apply(lambda d: d.sample(per_day, random_state=SEED)))
        score(f"trained on {per_day}/day subsample", fit_gbdt(sub), test)


def pdp_figure(gbdt, train):
    ink, line = "#333333", "#3B6EA5"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    panels = [("moneyness", "Moneyness", (0.5, 2.0)),
              ("stock_vol", "Realized volatility (annualized)", (0.05, 1.5))]
    for ax, (feat, label, (lo, hi)) in zip(axes, panels):
        grid = np.linspace(lo, hi, 60)
        pd_res = partial_dependence(gbdt, train[FEATURES], [feat],
                                    custom_values={feat: grid})
        ax.plot(grid, pd_res["average"][0], color=line, lw=2)
        ax.set_xlabel(label, color=ink)
        ax.set_ylabel("Predicted relative spread", color=ink)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("GBDT partial dependence", color=ink)
    fig.tight_layout()
    out = FIGURES / "fig4_pdp.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    train, test = load_split()

    print("baseline for reference:")
    score("GBDT (time split, milestone settings)", fit_gbdt(train), test)

    print("\n1. split by ticker instead of time:")
    ticker_split()

    print("\n2. hyperparameter grid (validated on Aug-Sep):")
    tuning(train, test)

    print("\n3. per-day sampling rate:")
    sampling_sensitivity(train, test)

    print("\n4. partial dependence plots:")
    pdp_figure(fit_gbdt(train), train)


if __name__ == "__main__":
    main()
