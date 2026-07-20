"""Cluster the contracts whose spreads the model most understates.

Fits the GBDT on Jan-Sep, takes the test contracts with the largest
positive residuals (actual much wider than predicted), and runs k-means
on their characteristics. The cluster profiles are the evidence for the
information asymmetry vs dealer capacity question.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import FEATURES, fit_gbdt, load_split

TOP_PCT = 0.95
SEED = 42

CLUSTER_FEATURES = ["moneyness", "days_to_expiry", "log_option_volume",
                    "log_open_interest", "log_stock_price",
                    "log_stock_volume", "stock_vol"]


def main():
    train, test = load_split()
    gbdt = fit_gbdt(train)

    test = test.copy()
    test["pred"] = gbdt.predict(test[FEATURES])
    test["residual"] = test.rel_spread - test.pred

    cut = test.residual.quantile(TOP_PCT)
    flagged = test[test.residual > cut].copy()
    flagged["log_open_interest"] = np.log1p(flagged.open_interest)
    print(f"flagged {len(flagged):,} of {len(test):,} test contracts "
          f"(residual > {cut:.3f})\n")

    X = StandardScaler().fit_transform(flagged[CLUSTER_FEATURES])

    # pick k by silhouette on a subsample, k=2..6
    sample = np.random.default_rng(SEED).choice(len(X), 5000, replace=False)
    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        scores[k] = silhouette_score(X[sample], km.labels_[sample])
        print(f"k={k} silhouette {scores[k]:.3f}")
    k = max(scores, key=scores.get)
    print(f"\nusing k={k}\n")

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    flagged["cluster"] = km.labels_

    profile_cols = ["rel_spread", "pred", "residual", "moneyness",
                    "days_to_expiry", "opt_mid", "log_option_volume",
                    "open_interest", "log_stock_price", "log_stock_volume",
                    "stock_vol", "is_call"]
    prof = flagged.groupby("cluster")[profile_cols].median()
    prof["pct_zero_volume"] = flagged.groupby("cluster").log_option_volume \
                                     .apply(lambda s: (s == 0).mean())
    prof["n"] = flagged.groupby("cluster").size()
    prof = prof.sort_values("n", ascending=False)

    pd.set_option("display.width", 200)
    print("cluster medians (pct_zero_volume and is_call are shares):")
    print(prof.round(3).to_string())

    base = test[profile_cols].median()
    print("\nfull test set medians for comparison:")
    print(base.round(3).to_string())


if __name__ == "__main__":
    main()
