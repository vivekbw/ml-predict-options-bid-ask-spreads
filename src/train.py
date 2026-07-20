"""Fit the baselines and models, report out-of-sample results.

Train on Jan-Sep 2024, test on Oct-Dec 2024.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

DATA = Path(__file__).resolve().parents[1] / "data" / "model" / "dataset.parquet"
SPLIT = "2024-10-01" # Training Set: Jan - Sept, Holdout Set: Oct - Dec

FEATURES = ["moneyness", "days_to_expiry", "is_call", "log_option_volume",
            "log_stock_price", "log_stock_volume", "stock_vol"]


def evaluate(name, y_test, pred, naive_mae):
    mae = mean_absolute_error(y_test, pred)
    # MAPE blows up when the true spread is near zero, so MASE is the
    # headline metric: our MAE relative to the naive baseline's (< 1 is better)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    print(f"{name:<12} R2 {r2_score(y_test, pred):6.3f}   MAE {mae:.4f}   "
          f"MAPE {mape:6.1%}   MASE {mae / naive_mae:.3f}")
    return pred


def main():
    df = pd.read_parquet(DATA)

    # split by time, not randomly: the same contract shows up on many nearby
    # days, so a random split would leak near-duplicates into the test set
    train, test = df[df.date < SPLIT], df[df.date >= SPLIT]
    X_train, y_train = train[FEATURES], train.rel_spread
    X_test, y_test = test[FEATURES], test.rel_spread
    print(f"train {len(train):,} rows (to {SPLIT}), test {len(test):,} rows\n")

    # floor to beat: predict the training-set average spread for everything
    naive = np.full(len(y_test), y_train.mean())
    naive_mae = mean_absolute_error(y_test, naive)
    evaluate("naive mean", y_test, naive, naive_mae)

    # plain linear regression; standardizing first puts every feature on the
    # same scale so the coefficients are comparable to each other
    ols = make_pipeline(StandardScaler(), LinearRegression())
    ols.fit(X_train, y_train)
    evaluate("OLS", y_test, ols.predict(X_test), naive_mae)

    # linear again but with L1/L2 penalties that shrink weights and can drop
    # useless features; penalty strength and mix picked by cross-validation
    enet = make_pipeline(StandardScaler(),
                         ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=3))
    enet.fit(X_train, y_train)
    evaluate("elastic net", y_test, enet.predict(X_test), naive_mae)

    # boosted trees: hundreds of small trees, each correcting the last one's
    # errors, which lets it fit curves and interactions the linear models miss
    gbdt = HistGradientBoostingRegressor(max_iter=300, random_state=42)
    gbdt.fit(X_train, y_train)
    evaluate("GBDT", y_test, gbdt.predict(X_test), naive_mae)

    # sign and size of each feature's linear effect, biggest first
    coefs = ols.named_steps["linearregression"].coef_
    print("\nOLS coefficients (standardized features):")
    for f, c in sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1])):
        print(f"  {f:<18} {c:+.4f}")


if __name__ == "__main__":
    main()
