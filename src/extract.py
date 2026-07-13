"""Convert the instructor-provided option zips into per-day parquet files.

Drop the downloaded zips into data/raw/ and run:
    python src/extract.py
Days that were already converted are skipped, so it's safe to re-run
as more zips finish downloading.
"""

import tempfile
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# the raw files have 59 columns, these are the ones we use
COLS = {
    "okey_tk": str,      # underlying ticker
    "okey_yr": int,      # expiry year
    "okey_mn": int,      # expiry month
    "okey_dy": int,      # expiry day
    "okey_xx": float,    # strike
    "okey_cp": str,      # Call / Put
    "tradingDate": str,
    "undSecType": str,
    "uBid": float,       # underlying bid
    "uAsk": float,       # underlying ask
    "bidPrc": float,     # option bid
    "askPrc": float,     # option ask
    "bidIV": float,
    "askIV": float,
    "srVol": float,      # vendor implied vol estimate
    "de": float,         # delta
    "openInterest": float,
    "prtCount": float,   # number of trades that day
    "prtVolume": float,  # contracts traded that day
}


def out_path(date: str) -> Path:
    return OUT_DIR / f"options_{date}.parquet"


def date_from_name(name: str) -> str:
    # e.g. "Copy of tbloptionclosemarkhist_EQT_v2.00_2024-01-02.zip" -> "2024-01-02"
    return Path(name).stem.split("_")[-1]


def convert_day(day_zip: Path) -> None:
    date = date_from_name(day_zip.name)
    df = pd.read_csv(day_zip, sep="\t", compression="zip",
                     usecols=list(COLS), dtype=COLS)
    df.to_parquet(out_path(date), index=False)
    print(f"{date}: {len(df):,} rows")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for z in sorted(RAW_DIR.glob("*.zip")):
        if "tbloptionclosemarkhist" in z.name:  # a single day, unbundled
            if out_path(date_from_name(z.name)).exists():
                continue
            convert_day(z)
        else:  # a drive-download bundle of day zips
            with zipfile.ZipFile(z) as bundle:
                for name in sorted(bundle.namelist()):
                    if not name.endswith(".zip"):
                        continue
                    if out_path(date_from_name(name)).exists():
                        continue
                    with tempfile.TemporaryDirectory() as tmp:
                        convert_day(Path(bundle.extract(name, tmp)))

    done = len(list(OUT_DIR.glob("options_*.parquet")))
    print(f"{done} days available in {OUT_DIR}")


if __name__ == "__main__":
    main()
