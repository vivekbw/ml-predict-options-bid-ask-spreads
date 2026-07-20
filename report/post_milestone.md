# Post-milestone results

Zoya V, Rithvik K, Vivek B

Two additions since the milestone report: the residual cluster analysis
and the robustness checks. Everything below is out of sample: models are
fit on January through September 2024 and evaluated on October through
December, except where a check deliberately changes the split. Scripts:
`src/residuals.py` and `src/robustness.py`.

## 1. Residual clusters: who has wider spreads than they should?

We flag the 5% of test contracts with the largest positive residuals
(actual spread far above predicted, 16,000 contracts) and run k-means on
their standardized characteristics. Silhouette scores pick k=2.

| Median | Cluster A (76%) | Cluster B (24%) | All test |
|---|---|---|---|
| Relative spread | 1.61 | 1.53 | 0.12 |
| Predicted spread | 0.66 | 0.55 | 0.16 |
| Moneyness | 0.85 | 0.82 | 1.12 |
| Days to expiry | 121 | 42 | 71 |
| Option mid | $2.41 | $0.58 | $8.95 |
| Open interest | 0 | 102 | 1 |
| Share with zero volume | 93% | 21% | 76% |
| log share volume | 13.7 | 15.1 | 14.3 |
| Realized vol | 0.33 | 0.37 | 0.33 |

Both clusters are dominated by out-of-the-money puts (median is_call is 0
in both, vs a call majority overall), but they look like the two theories:

- Cluster A is the dealer capacity story. These contracts essentially never
  trade (93% zero volume, zero open interest), are long dated, and sit on
  lower-volume underlyings. No order flow competes the spread down, and a
  dealer who fills one carries the inventory for months, so quotes are wide
  even though nothing suggests informed trading.
- Cluster B is consistent with information asymmetry. These are cheap,
  short-dated OTM puts on high-volume, higher-volatility names, and they
  do trade (79% traded that day, median open interest 102). This is the
  textbook vehicle for trading on bad news ahead of events, and dealers
  quoting them face exactly the adverse selection in Glosten and Milgrom:
  the spread stays wide despite active flow.

## 2. Robustness

| Check | R2 | MAE |
|---|---|---|
| GBDT, time split (baseline) | 0.61 | 0.164 |
| Split by ticker (717 names held out) | 0.62 | 0.173 |
| Tuned GBDT, time split | 0.61 | 0.162 |
| Trained on 2,500/day subsample | 0.61 | 0.164 |
| Trained on 1,000/day subsample | 0.61 | 0.164 |

The result is stable in every direction we pushed it. Holding out 20% of
underlyings entirely gives the same R2 as the time split, so the model is
learning how characteristics map to spreads, not memorizing per-name
liquidity. A 2x3 hyperparameter grid validated on Aug-Sep improves test
MAE by about 1% over the milestone settings, so the defaults were already
near the ceiling for these features. Cutting the training sample to 1,000
contracts per day costs 0.003 of R2, so the 5,000/day rate is not doing
any heavy lifting.

We also checked that nothing depends on the randomization seeds. Every
seed in the pipeline (the GBDT's internal seed, the per-day subsample
seed, and the ticker holdout seed) was varied over {0, 7, 42, 123} in a
4x4 grid for each of three settings, 48 fits in total. MASE lands in
0.505 to 0.507 for the 2,500/day subsample, 0.507 to 0.511 at 1,000/day,
and 0.519 to 0.528 across ticker splits, where the holdout seed changes
which underlyings are held out and so moves the number the most. Even
that widest band is under 0.01 of MASE, far smaller than the gap to the
next best model. The full grid is in `report/figures/fig5_seed_mase.png`;
within any column of the heatmaps the model seed changes MASE by at most
0.001, so the fit itself is essentially deterministic.

Partial dependence plots for moneyness and realized volatility are in
`report/figures/fig4_pdp.png`. Both effects the linear models missed are
visible directly: predicted spread rises steeply as contracts move out of
the money below moneyness 1 and flattens in the money, and realized
volatility has a strong negative effect that is entirely concentrated
below about 0.4, then flat. A single linear coefficient averages each of
these curves to roughly zero, which is why OLS saw nothing.

## Contributions

| Name | Contribution |
|---|---|
| Vivek B | Majority of the code: data pipeline, dataset construction, models, residual clustering, robustness checks, figures |
| Rithvik K | Research approach and design, milestone report, final white paper |
| Zoya V | Research approach and design, milestone report, final white paper |
