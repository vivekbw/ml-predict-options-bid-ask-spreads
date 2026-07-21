"""Robustness checks for the GBDT result.

Five checks: a split by ticker instead of time, a small hyperparameter
grid validated on Aug-Sep, sensitivity to the per-day sampling rate,
sensitivity to every randomization seed, and partial dependence plots
for the two features whose effects looked nonlinear.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, r2_score

from common import DATA, FEATURES, FIGURES, INK, fit_gbdt, load_split, plot_style

SEED = 42
SEEDS = [0, 7, 42, 123]

plot_style()


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


def mase(gbdt, test, naive_level):
    pred = gbdt.predict(test[FEATURES])
    naive = np.full(len(test), naive_level)
    return (mean_absolute_error(test.rel_spread, pred)
            / mean_absolute_error(test.rel_spread, naive))


def seed_grids(train, test):
    """MASE over a model seed x data seed grid, three settings."""
    grids = {}

    for per_day in [2500, 1000]:
        m = np.zeros((len(SEEDS), len(SEEDS)))
        for j, ds in enumerate(SEEDS):
            sub = (train.groupby("date", group_keys=False)
                        .apply(lambda d: d.sample(per_day, random_state=ds)))
            for i, ms in enumerate(SEEDS):
                m[i, j] = mase(fit_gbdt(sub, random_state=ms), test,
                               sub.rel_spread.mean())
        grids[f"{per_day}/day subsample, time split"] = ("sampling seed", m)

    df = pd.read_parquet(DATA)
    tickers = np.sort(df.ticker.unique())
    m = np.zeros((len(SEEDS), len(SEEDS)))
    for j, ds in enumerate(SEEDS):
        rng = np.random.default_rng(ds)
        held = set(rng.choice(tickers, int(0.2 * len(tickers)), replace=False))
        tr, te = df[~df.ticker.isin(held)], df[df.ticker.isin(held)]
        for i, ms in enumerate(SEEDS):
            m[i, j] = mase(fit_gbdt(tr, random_state=ms), te,
                           tr.rel_spread.mean())
    grids["ticker split, full sample"] = ("holdout seed", m)
    return grids


def seed_figure(grids, baseline):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    mats = [m for _, m in grids.values()]
    lo, hi = min(m.min() for m in mats), max(m.max() for m in mats)

    for ax, (title, (xlabel, m)) in zip(axes.flat, grids.items()):
        ax.imshow(m, cmap="Greys", vmin=lo - 0.005, vmax=hi + 0.005)
        for i in range(len(SEEDS)):
            for j in range(len(SEEDS)):
                dark = (m[i, j] - lo + 0.005) / (hi - lo + 0.01) > 0.6
                ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if dark else INK)
        ax.set_xticks(range(len(SEEDS)), SEEDS)
        ax.set_yticks(range(len(SEEDS)), SEEDS)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("model seed")
        ax.set_title(title, fontsize=11)

    # fourth panel: every run against the milestone baseline
    ax = axes.flat[3]
    for row, (title, (_, m)) in enumerate(grids.items()):
        vals = m.ravel()
        ax.scatter(vals, np.full(len(vals), row), s=28, color="#555555",
                   alpha=0.6, edgecolors="none")
    ax.axvline(baseline, color="#999999", ls="--", lw=1)
    ax.text(baseline, 2.55, " milestone baseline", fontsize=9, color="#777777")
    ax.set_yticks(range(len(grids)),
                  [t.split(",")[0] for t in grids], fontsize=9)
    ax.set_ylim(-0.5, 2.8)
    ax.set_xlabel("MASE")
    ax.set_title("all 48 runs", fontsize=11)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)

    fig.suptitle("MASE across randomization seeds")
    fig.tight_layout()
    out = FIGURES / "fig5_seed_mase.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def pdp_figure(gbdt, train):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    panels = [("moneyness", "Moneyness", (0.5, 2.0)),
              ("stock_vol", "Realized volatility (annualized)", (0.05, 1.5))]
    for ax, (feat, label, (lo, hi)) in zip(axes, panels):
        grid = np.linspace(lo, hi, 60)
        pd_res = partial_dependence(gbdt, train[FEATURES], [feat],
                                    custom_values={feat: grid})
        ax.plot(grid, pd_res["average"][0], color=INK, lw=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Predicted relative spread")
        ax.grid(alpha=0.25, lw=0.5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("GBDT partial dependence")
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

    print("\n4. randomization seeds:")
    baseline = mase(fit_gbdt(train), test, train.rel_spread.mean())
    grids = seed_grids(train, test)
    for title, (_, m) in grids.items():
        print(f"  {title:<32} MASE {m.min():.3f} to {m.max():.3f}")
    seed_figure(grids, baseline)

    print("\n5. partial dependence plots:")
    pdp_figure(fit_gbdt(train), train)


if __name__ == "__main__":
    main()
