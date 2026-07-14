"""Build the modeling dataset from the per-day option parquets and CRSP.

Filters each day down to contracts with real quotes on common stocks,
merges in stock-level features from CRSP, samples 5000 contracts per day,
and writes one combined parquet. Prints the filter counts for the report.
"""

import glob
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
CRSP_ZIP = next(DATA.glob("crsp/crsp_*.csv.zip"))
OUT = DATA / "model" / "dataset.parquet"

PER_DAY = 5000
SEED = 42


def load_crsp() -> pd.DataFrame:
    crsp = pd.read_csv(CRSP_ZIP, skiprows=1)
    crsp.columns = ["date", "permno", "tsymbol", "ticker", "shrcd",
                    "prc", "vol", "ret", "shrout"]
    crsp = crsp[crsp.shrcd.isin([10, 11, 12])].copy()  # common stock only
    crsp["ticker"] = crsp.ticker.fillna(crsp.tsymbol)
    crsp["ret"] = pd.to_numeric(crsp.ret, errors="coerce")

    # trailing 21-day realized vol, annualized
    crsp = crsp.sort_values(["permno", "date"])
    crsp["stock_vol"] = (crsp.groupby("permno").ret
                             .transform(lambda s: s.rolling(21).std())
                         * np.sqrt(252))
    crsp["log_stock_volume"] = np.log1p(crsp.vol)

    crsp = crsp.dropna(subset=["ticker", "stock_vol"])
    crsp = crsp.drop_duplicates(["ticker", "date"])
    return crsp[["ticker", "date", "stock_vol", "log_stock_volume"]]


def build_day(path: str, crsp: pd.DataFrame, counts: dict) -> pd.DataFrame:
    date = path.split("_")[-1].split(".")[0]
    df = pd.read_parquet(path)
    counts["all contracts"] += len(df)

    df = df[(df.bidPrc > 0) & (df.askPrc > df.bidPrc)]
    counts["has a real quote (bid > 0, ask > bid)"] += len(df)

    df = df[(df.uBid > 0) & (df.uAsk > df.uBid)]
    counts["valid underlying quote"] += len(df)

    df = df[(df.bidPrc + df.askPrc) / 2 >= 0.10]
    counts["option mid price >= $0.10"] += len(df)

    expiry = pd.to_datetime(dict(year=df.okey_yr, month=df.okey_mn, day=df.okey_dy))
    df["days_to_expiry"] = (expiry - pd.Timestamp(date)).dt.days
    df = df[df.days_to_expiry.between(7, 365)]
    counts["7 to 365 days to expiry"] += len(df)

    day_crsp = crsp[crsp.date == date]
    df = df.merge(day_crsp, left_on="okey_tk", right_on="ticker", how="inner")
    counts["matched to CRSP common stock"] += len(df)

    if len(df) > PER_DAY:
        df = df.sample(PER_DAY, random_state=SEED)

    out = pd.DataFrame({
        "date": date,
        "ticker": df.okey_tk,
        "rel_spread": 2 * (df.askPrc - df.bidPrc) / (df.askPrc + df.bidPrc),
        "moneyness": np.where(df.okey_cp == "Call",
                              (df.uBid + df.uAsk) / 2 / df.okey_xx,
                              df.okey_xx / ((df.uBid + df.uAsk) / 2)),
        "days_to_expiry": df.days_to_expiry,
        "is_call": (df.okey_cp == "Call").astype(int),
        "log_option_volume": np.log1p(df.prtVolume),
        "log_stock_price": np.log((df.uBid + df.uAsk) / 2),
        "log_stock_volume": df.log_stock_volume,
        "stock_vol": df.stock_vol,
    })
    return out


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    crsp = load_crsp()
    counts = {k: 0 for k in ["all contracts",
                             "has a real quote (bid > 0, ask > bid)",
                             "valid underlying quote",
                             "option mid price >= $0.10",
                             "7 to 365 days to expiry",
                             "matched to CRSP common stock"]}

    days = []
    for path in sorted(glob.glob(str(DATA / "processed" / "*.parquet"))):
        days.append(build_day(path, crsp, counts))
    data = pd.concat(days, ignore_index=True)
    data.to_parquet(OUT, index=False)

    print("sample selection:")
    for step, n in counts.items():
        print(f"  {step}: {n:,}")
    print(f"after sampling {PER_DAY}/day: {len(data):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
