"""Three hypotheses about who defaults, tested on the derived risk metrics.

    python -m src.hypothesis_tests

Figures are written to figures/.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # write files; never block on a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.features import TARGET, MissingDataError, build_features

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
ALPHA = 0.05
HIGH_UTILISATION = 0.8


def _header(number: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"H{number}  {title}")
    print("=" * 72)


def _verdict(p_value: float) -> str:
    return "SUPPORTED" if p_value < ALPHA else "NOT SUPPORTED"


def _save(fig, filename: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)
    print(f"  saved figures/{filename}")


# ----------------------------------------------------------------------- H1
def test_utilisation(df: pd.DataFrame) -> dict:
    """H1 — do high-utilisation clients default more often?

    The outcome is binary, so this compares two *proportions*. A t-test assumes
    roughly normal data; a 0/1 variable is Bernoulli, and the t-statistic it
    produces does not mean what it appears to mean.
    """
    _header(1, "High credit utilisation raises default risk")

    high = df.loc[df["AVG_UTIL"] > HIGH_UTILISATION, TARGET]
    low = df.loc[df["AVG_UTIL"] <= HIGH_UTILISATION, TARGET]

    table = np.array([
        [int(high.sum()), int(len(high) - high.sum())],
        [int(low.sum()), int(len(low) - low.sum())],
    ])
    chi2, p_value, _, _ = stats.chi2_contingency(table)

    rate_high, rate_low = high.mean(), low.mean()
    # Cramer's V on a 2x2 reduces to phi; report it so a large chi-square at
    # n = 30,000 is not mistaken for a large effect.
    phi = np.sqrt(chi2 / table.sum())

    print(f"  utilisation > {HIGH_UTILISATION:.0%}   default rate "
          f"{rate_high:>6.2%}   (n = {len(high):,})")
    print(f"  utilisation <= {HIGH_UTILISATION:.0%}  default rate "
          f"{rate_low:>6.2%}   (n = {len(low):,})")
    print(f"  difference {rate_high - rate_low:+.2%}   "
          f"relative risk {rate_high / rate_low:.2f}x")
    print(f"  chi2 = {chi2:,.1f}   p = {p_value:.4g}   phi = {phi:.3f}")

    supported = p_value < ALPHA and rate_high > rate_low
    print(f"\n  {'SUPPORTED' if supported else 'NOT SUPPORTED'}.")

    _plot_utilisation(rate_high, rate_low, len(high), len(low), p_value)
    return {"rate_high": float(rate_high), "rate_low": float(rate_low),
            "chi2": float(chi2), "p": float(p_value), "phi": float(phi),
            "supported": bool(supported)}


def _plot_utilisation(rate_high, rate_low, n_high, n_low, p_value) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"> {HIGH_UTILISATION:.0%}\n(n = {n_high:,})",
         f"<= {HIGH_UTILISATION:.0%}\n(n = {n_low:,})"],
        [rate_high * 100, rate_low * 100],
        color=["indianred", "steelblue"], alpha=0.85,
    )
    for bar, rate in zip(bars, (rate_high, rate_low)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{rate:.2%}", ha="center", va="bottom")
    ax.set_ylabel("Default rate (%)")
    ax.set_xlabel("Average credit utilisation")
    ax.set_title(f"Default rate by utilisation  (p = {p_value:.4g})")
    fig.tight_layout()
    _save(fig, "h1_utilisation.png")


# ----------------------------------------------------------------------- H2
def test_delay_trend(df: pd.DataFrame) -> dict:
    """H2 — does a worsening repayment trend predict default?

    DELAY_TREND is fitted oldest -> newest (see src/features.py), so a positive
    slope means delays are getting worse. Fitting in the source column order
    reverses the sign and makes this relationship look counterintuitive.
    """
    _header(2, "A worsening repayment trend predicts default")

    trend = df["DELAY_TREND"]
    outcome = df[TARGET]
    correlation, p_value = stats.pointbiserialr(trend, outcome)

    print("  DELAY_TREND is fitted oldest -> newest:")
    print("    positive = delays worsening,  negative = delays improving")
    print(f"\n  point-biserial r = {correlation:+.3f}   p = {p_value:.4g}   "
          f"(n = {len(df):,})")

    defaulted = trend[outcome == 1]
    repaid = trend[outcome == 0]
    print(f"  mean trend, defaulted {defaulted.mean():+.4f}")
    print(f"  mean trend, repaid    {repaid.mean():+.4f}")

    supported = p_value < ALPHA and correlation > 0
    print(f"\n  {'SUPPORTED' if supported else 'NOT SUPPORTED'}: "
          f"clients whose delays worsen default "
          f"{'more' if correlation > 0 else 'less'} often.")
    print(f"  |r| = {abs(correlation):.3f} is a "
          f"{'weak' if abs(correlation) < 0.3 else 'moderate'} association — "
          f"trend alone is not a decision rule.")

    _plot_delay_trend(defaulted, repaid, correlation)
    return {"r": float(correlation), "p": float(p_value),
            "mean_defaulted": float(defaulted.mean()),
            "mean_repaid": float(repaid.mean()), "supported": bool(supported)}


def _plot_delay_trend(defaulted, repaid, correlation) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([repaid, defaulted], tick_labels=["Repaid", "Defaulted"],
               patch_artist=True,
               boxprops=dict(facecolor="mediumseagreen", alpha=0.6))
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Delay trend (slope, oldest -> newest)")
    ax.set_title(f"Repayment trend by outcome  (r = {correlation:+.3f})\n"
                 "above zero = delays worsening")
    fig.tight_layout()
    _save(fig, "h2_delay_trend.png")


# ----------------------------------------------------------------------- H3
def test_bill_volatility(df: pd.DataFrame) -> dict:
    """H3 — does bill volatility differ between defaulters and non-defaulters?

    Bill volatility is a standard deviation: strictly positive and heavily
    right-skewed, so Mann-Whitney is used rather than a t-test, and the medians
    are reported rather than the means.
    """
    _header(3, "Bill volatility differs between defaulters and non-defaulters")

    defaulted = df.loc[df[TARGET] == 1, "BILL_VOLATILITY"].dropna().to_numpy()
    repaid = df.loc[df[TARGET] == 0, "BILL_VOLATILITY"].dropna().to_numpy()

    u_statistic, p_value = stats.mannwhitneyu(
        defaulted, repaid, alternative="two-sided"
    )
    effect = 1.0 - (2.0 * u_statistic) / (len(defaulted) * len(repaid))

    print(f"  defaulted  median {np.median(defaulted):>12,.0f}   "
          f"(n = {len(defaulted):,})")
    print(f"  repaid     median {np.median(repaid):>12,.0f}   "
          f"(n = {len(repaid):,})")
    print(f"  p = {p_value:.4g}   rank-biserial effect size = {effect:+.3f}")

    supported = p_value < ALPHA
    direction = "lower" if np.median(defaulted) < np.median(repaid) else "higher"
    print(f"\n  {_verdict(p_value)}: defaulters show {direction} bill volatility.")
    if supported and abs(effect) < 0.1:
        print("  The effect size is negligible — detectable at this sample size,")
        print("  but too small to separate the groups in practice.")

    _plot_volatility(defaulted, repaid, p_value)
    return {"median_defaulted": float(np.median(defaulted)),
            "median_repaid": float(np.median(repaid)),
            "p": float(p_value), "effect_size": float(effect),
            "supported": bool(supported)}


def _plot_volatility(defaulted, repaid, p_value) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [np.clip(repaid, None, np.quantile(repaid, 0.95)),
         np.clip(defaulted, None, np.quantile(defaulted, 0.95))],
        tick_labels=["Repaid", "Defaulted"], patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
    )
    ax.set_ylabel("Bill volatility (std of monthly bills)")
    ax.set_title(f"Bill volatility by outcome  (p = {p_value:.4g})")
    fig.tight_layout()
    _save(fig, "h3_bill_volatility.png")


def main() -> None:
    try:
        df = build_features()
    except MissingDataError as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)

    test_utilisation(df)
    test_delay_trend(df)
    test_bill_volatility(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
