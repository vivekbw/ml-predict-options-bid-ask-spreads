# Milestone Report: Predicting Option Bid-Ask Spreads

Zoya V, Rithvik K, Vivek B

## 1. Updated problem statement

The bid-ask spread on an option is both the cost a trader pays to transact and
the compensation a market maker earns for providing liquidity. We ask how much
of the cross-sectional variation in option spreads can be explained by
observable contract and underlying stock characteristics, and which contracts
have spreads wider than those characteristics predict. We then examine the
contracts the model most understates to ask whether wide spreads look like
information asymmetry (Glosten and Milgrom, 1985) or limited dealer capacity
(Christoffersen et al., 2018).

One change from the proposal: the instructor-provided data turned out to be
end-of-day closing marks rather than 5-minute intraday quotes, one snapshot
per contract per trading day. We therefore predict the cross-section of daily
closing relative spreads. This removes two anticipated challenges (time-of-day
effects and stale intraday quotes) since closing marks are market maker quotes
that are live at the close whether or not the contract traded that day.

## 2. Final data

We combine two sources for calendar year 2024:

- Option close marks: one file per trading day covering all listed US equity
  options, about 1.46M contracts per day across roughly 6,100 underlyings
  (252 days, 381M contract-day rows in total).
- CRSP daily stock file for the underlyings: price, returns, share volume,
  shares outstanding, and share codes for 5,641 securities.

Target: relative spread = 2(ask - bid)/(ask + bid), which normalizes for
option price levels so expensive options are not mislabeled as liquid.

Features: moneyness (S/K for calls, K/S for puts, with S the underlying quote
midpoint), days to expiry, a call dummy, log(1 + option volume), log
underlying price, log share volume, and 21-day realized volatility from CRSP
returns (annualized).

Sample selection (contract-day counts):

| Filter | Rows remaining |
|---|---|
| All contract-days | 381,499,776 |
| Has a real quote (bid > 0, ask > bid) | 291,945,225 |
| Valid underlying quote | 285,522,061 |
| Option mid price >= $0.10 | 280,895,128 |
| 7 to 365 days to expiry | 233,052,649 |
| Matched to CRSP common stock (share codes 10-12) | 133,268,956 |

The quote filter removes deep out-of-the-money contracts no dealer quotes.
The minimum price filter removes penny options whose relative spread is
mechanically near its maximum of 2. The expiry filter drops contracts inside
one week of expiration and illiquid long-dated contracts. The CRSP match
restricts to common stocks, excluding ETFs, ADRs, and funds. There is no
missing data after these filters. Because realized volatility needs 21 prior
trading days, the first month of 2024 drops out, leaving 232 days.

From the eligible sample we draw 5,000 contracts per day at random (fixed
seed), giving a modeling dataset of 1.16M rows across 3,387 underlyings.

Descriptive statistics:

| Variable | Mean | SD | P25 | Median | P75 |
|---|---|---|---|---|---|
| Relative spread | 0.324 | 0.476 | 0.049 | 0.118 | 0.333 |
| Moneyness | 1.33 | 1.90 | 0.91 | 1.11 | 1.43 |
| Days to expiry | 107 | 91 | 30 | 74 | 175 |
| log(1 + option volume) | 0.59 | 1.32 | 0 | 0 | 0 |
| log stock price | 4.43 | 1.48 | 3.52 | 4.56 | 5.40 |
| log share volume | 14.2 | 1.6 | 13.2 | 14.2 | 15.2 |
| Realized vol (annualized) | 0.49 | 5.15 | 0.23 | 0.33 | 0.49 |

51% of contracts are calls. 76% of quoted contracts did not trade that day,
which is typical for options and is why volume enters as log(1 + volume).

## 3. Baseline model and out-of-sample results

We split by time: train on January through September (840K rows), test on
October through December (320K rows). A random split would leak information,
since the same contract appears on adjacent days.

| Model | Test R2 | MAE | MASE |
|---|---|---|---|
| Naive (train mean) | 0.00 | 0.326 | 1.000 |
| OLS | 0.06 | 0.307 | 0.942 |
| Elastic net (CV over alpha, l1 ratio) | 0.06 | 0.307 | 0.943 |
| Gradient-boosted trees | 0.62 | 0.163 | 0.501 |

All models run end to end: dataset construction, fitting, and out-of-sample
evaluation are reproducible from two scripts in the code archive.

## 4. Preliminary results and what surprised us

The headline: the cross-section of option spreads is explainable, but not
linearly. OLS on all seven features beats the naive mean by very little (R2 of
0.06), and the gradient-boosted trees explain 62% of out-of-sample variance
and cut mean absolute error in half relative to the naive baseline.

What surprised us:

1. Elastic net adds nothing over OLS. Regularization solves variable selection
   and collinearity, but with seven sensible features that is not the
   bottleneck. The bottleneck is functional form.
2. The moneyness effect is strongly nonlinear and dominates. Permutation
   importance for the GBDT puts moneyness first by a factor of five over days
   to expiry. Relative spread is lowest for in-the-money contracts (high
   premium in the denominator) and explodes as contracts move out of the
   money, a shape a single linear coefficient cannot capture.
3. Realized volatility gets a near-zero linear coefficient yet carries real
   importance in the GBDT, again suggesting its effect is nonlinear or
   interacts with moneyness.
4. MAPE is a poor headline metric here: spreads near zero make percentage
   errors explode (387% for OLS even as R2 improves). We will report MASE and
   MAE as primary metrics and flag this in the final report.

## 5. Remaining work (to July 21)

- Residual analysis: k-means on the contracts with the largest positive
  residuals, then characterize the clusters (volume, open interest, expiry,
  underlying type) to argue information asymmetry vs dealer capacity.
- Log-transform the target so the linear models compete on fairer ground, and
  winsorize realized volatility (its SD of 5.2 is driven by a few extreme
  small-cap movers).
- Pull December 2023 CRSP returns to recover January 2024, and re-pull CRSP
  with all securities to include names that delisted during 2024 (currently a
  mild survivorship bias, and delisting candidates are exactly where adverse
  selection should show up).
- Economic translation: convert predicted spreads into round-trip transaction
  costs for example strategies and flag the most expensive residual contracts.
- Robustness: alternative splits (by ticker), GBDT tuning, and partial
  dependence plots for the moneyness and volatility effects.

## References

Christoffersen, P., Goyenko, R., Jacobs, K., and Karoui, M. (2018).
Illiquidity premia in the equity options market. Review of Financial Studies,
31(3), 811-851.

Glosten, L. R., and Milgrom, P. R. (1985). Bid, ask and transaction prices in
a specialist market with heterogeneously informed traders. Journal of
Financial Economics, 14(1), 71-100.
