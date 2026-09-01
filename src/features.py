"""Load the credit-card default export and build the derived risk metrics.

The raw file is not committed. Download `Credit_card.xlsx` (the UCI "Default of
Credit Card Clients" dataset) and place it in the repository root — see README.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "Credit_card.xlsx"
SAMPLE_CSV = ROOT / "Feature_selected_credits_sample.csv"

TARGET = "default.payment.next.month"

# The source orders repayment columns newest -> oldest:
#   PAY_0 = September 2005 (most recent) ... PAY_6 = April 2005 (oldest).
PAY_COLUMNS_SOURCE_ORDER = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# Chronological order, oldest first. Any trend over time must use this, or the
# slope's sign is inverted.
PAY_COLUMNS_CHRONOLOGICAL = list(reversed(PAY_COLUMNS_SOURCE_ORDER))

BILL_COLUMNS_SOURCE_ORDER = [f"BILL_AMT{i}" for i in range(1, 7)]
BILL_COLUMNS_CHRONOLOGICAL = list(reversed(BILL_COLUMNS_SOURCE_ORDER))
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

# A repayment status of 2 or more means the account is two-plus months late.
DELAY_STREAK_THRESHOLD = 2


class MissingDataError(FileNotFoundError):
    """Raised when the raw export is not where the script expects it."""


def load_raw(path: pathlib.Path = RAW_XLSX) -> pd.DataFrame:
    """Read the raw export, failing with instructions if it is absent."""
    if not path.exists():
        raise MissingDataError(
            f"Could not find {path.name} in {path.parent}.\n"
            "This file is not committed. Download the 'Default of Credit Card\n"
            "Clients' dataset from\n"
            "  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients\n"
            f"and save it as: {path}"
        )
    return pd.read_excel(path)


def add_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Average delay, trend over time, and worst consecutive-delay streak."""
    df = df.copy()

    df["AVG_PAY_DELAY"] = df[PAY_COLUMNS_SOURCE_ORDER].mean(axis=1)

    # Slope of repayment status over time, fitted oldest -> newest so that a
    # POSITIVE value means delays are getting worse. Fitting in the source
    # column order inverts this and makes the correlation with default look
    # counterintuitive when it is simply backwards.
    chronological = df[PAY_COLUMNS_CHRONOLOGICAL].to_numpy(dtype=float)
    months = np.arange(chronological.shape[1])
    # polyfit over all rows at once: solve the same 1-D fit column-wise.
    slopes = np.polyfit(months, chronological.T, deg=1)[0]
    df["DELAY_TREND"] = slopes

    df["MAX_DELAY_STREAK"] = _max_delay_streak(chronological)
    return df


def _max_delay_streak(values: np.ndarray) -> np.ndarray:
    """Longest run of consecutive months at or beyond the delay threshold."""
    late = values >= DELAY_STREAK_THRESHOLD
    streak = np.zeros(len(values), dtype=int)
    best = np.zeros(len(values), dtype=int)
    for month in range(late.shape[1]):
        streak = np.where(late[:, month], streak + 1, 0)
        best = np.maximum(best, streak)
    return best


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Utilisation, payment-to-bill ratio, volatility and last-month shock."""
    df = df.copy()

    for i in range(1, 7):
        df[f"UTIL_{i}"] = df[f"BILL_AMT{i}"] / df["LIMIT_BAL"]
    df["AVG_UTIL"] = df[[f"UTIL_{i}" for i in range(1, 7)]].mean(axis=1)

    # A bill of zero (or a credit balance) makes payment/bill meaningless
    # rather than infinite, so those months are excluded from the average
    # instead of poisoning it with inf.
    for i in range(1, 7):
        bill = df[f"BILL_AMT{i}"]
        df[f"PAY_RATIO_{i}"] = np.where(
            bill > 0, df[f"PAY_AMT{i}"] / bill.replace(0, np.nan), np.nan
        )
    df["AVG_PAY_RATIO"] = df[[f"PAY_RATIO_{i}" for i in range(1, 7)]].mean(
        axis=1, skipna=True
    )

    df["BILL_VOLATILITY"] = df[BILL_COLUMNS_SOURCE_ORDER].std(axis=1)

    # Positive = the most recent bill ran above the preceding months' average.
    df["LAST_MONTH_SHOCK"] = (
        df["BILL_AMT1"] - df[BILL_COLUMNS_SOURCE_ORDER[1:]].mean(axis=1)
    )
    return df


def build_features(verbose: bool = True) -> pd.DataFrame:
    """Full pipeline: raw export -> analysis-ready frame."""
    if verbose:
        print("Loading raw export...")
    df = load_raw()
    if verbose:
        print(f"  {len(df):,} clients, {df[TARGET].mean():.2%} defaulted")

    df = add_delay_features(df)
    df = add_balance_features(df)

    if verbose:
        n_bad = int(df["AVG_PAY_RATIO"].isna().sum())
        print(f"  AVG_PAY_RATIO undefined for {n_bad:,} clients "
              f"(no positive bill in any month)")
    return df


if __name__ == "__main__":
    data = build_features()
    data.head(50).to_csv(SAMPLE_CSV, index=False)
    print(f"\nwrote a 50-row sample to {SAMPLE_CSV.name}")
