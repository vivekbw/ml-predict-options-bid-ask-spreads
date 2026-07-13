# Predicting Option Bid-Ask Spreads

We predict the cross-section of option relative bid-ask spreads,
`2 * (ask - bid) / (ask + bid)`, from contract characteristics (moneyness,
time to maturity, option type, volume) and underlying stock characteristics
(price, volume, volatility). We then look at the contracts our model most
understates and ask whether the wide spreads look like information asymmetry
or limited dealer capacity.

## Setup

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The provided data is one file per trading day of end-of-day option
marks for all listed US equity options: about 1.46M contract rows per day,
252 days of 2024, 59 columns per row. Raw it is roughly 150 GB uncompressed,
so none of it is committed to git.

To rebuild from scratch: drop the downloaded zips into `data/raw/` (the
`drive-download-*.zip` bundles and individual `tbloptionclosemarkhist` day
zips both work), then

```
python src/extract.py
```

This keeps the 19 columns we use (contract key, option and underlying quotes,
IVs, delta, volume, open interest) and writes one
`data/processed/options_YYYY-MM-DD.parquet` per day, about 42 MB each. It
skips days already converted, so it is safe to re-run. CRSP files for the
underlying stocks go in `data/crsp/`.

## Layout

```
data/raw/        instructor zips (not in git)
data/processed/  per-day parquet files (not in git)
data/crsp/       CRSP data for the underlyings (not in git)
src/             pipeline code
notebooks/       exploration
report/          proposal, milestone, final report
```

## Pipeline

1. `src/extract.py`: raw zips to per-day parquet
2. `src/build_dataset.py`: filters, features, CRSP merge, 5000 contracts per day sample
3. `src/train.py`: naive and OLS baselines, elastic net, gradient-boosted trees, out-of-sample eval
4. next: k-means on the contracts with the largest positive residuals
