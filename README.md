# Credit Card Default Analysis

Default-risk analysis of 30,000 Taiwanese credit card clients (April–September
2005), combining engineered risk metrics, three hypothesis tests, two
classifiers, and a Power BI dashboard.

Of the 30,000 clients, **22.12% (6.64K) defaulted** the following month.

| Deliverable | |
|---|---|
| `src/features.py` | Derived risk metrics from raw repayment and billing history |
| `src/hypothesis_tests.py` | Three hypotheses about who defaults |
| `src/modeling.py` | Logistic Regression baseline, then Random Forest |
| `Credit_card.pbix` | Three-page Power BI dashboard |

## Dashboard

**Page 1 — Delay behaviour by demographics.** Average payment delay and
maximum delay streak, both by age and by sex. Clients aged 21–25 show the best
payment-delay behaviour; those aged 23–57 average under one month of delay
streak. Female clients show more delays on average, while male clients carry a
longer delay streak (higher by 0.13).

**Page 2 — Default drivers.** Default rate by most recent repayment status
(`PAY_0`) against an industry-acceptable benchmark, average utilisation against
a benchmark, credit limit by age, and default counts broken down by education,
marital status and sex. Includes the overall 77.88% / 22.12% split.

**Page 3 — Risk segmentation.** Utilisation against average payment delay
coloured by outcome, plus default counts by delay streak and by credit limit,
sliceable by age and limit.

## Risk metrics

Built in `src/features.py` from the raw monthly columns:

| Metric | Meaning |
|---|---|
| `AVG_PAY_DELAY` | Mean repayment status across the six months |
| `DELAY_TREND` | Slope of repayment status over time — **positive = worsening** |
| `MAX_DELAY_STREAK` | Longest run of consecutive months two-plus payments late |
| `AVG_UTIL` | Mean of monthly bill ÷ credit limit |
| `AVG_PAY_RATIO` | Mean of payment ÷ bill, over months with a positive bill |
| `BILL_VOLATILITY` | Standard deviation of the six monthly bills |
| `LAST_MONTH_SHOCK` | Most recent bill minus the average of the preceding five |

### On `DELAY_TREND` and column order

The source orders repayment columns **newest to oldest**: `PAY_0` is September
2005, `PAY_6` is April 2005. Fitting a slope in that column order produces a
value whose sign is inverted — a positive slope would mean delays *improving*.

`DELAY_TREND` is therefore fitted **oldest → newest**, so a positive value
means delays are getting worse, which is what the hypothesis assumes. This
matters: fitted in source order, the correlation with default comes out
negative and looks like a counterintuitive finding, when it is only the
columns being backwards.

### On `AVG_PAY_RATIO`

Payment ÷ bill is undefined when the bill is zero and meaningless when it is
negative (a credit balance). Those months are excluded from the average rather
than contributing `inf`, which would otherwise propagate through every
downstream statistic.

## Hypothesis tests

```bash
pip install -r requirements.txt
python -m src.hypothesis_tests
```

| # | Hypothesis | Test |
|---|---|---|
| H1 | High utilisation raises default risk | Chi-square on proportions + φ |
| H2 | A worsening repayment trend predicts default | Point-biserial correlation |
| H3 | Bill volatility differs by outcome | Mann-Whitney U + rank-biserial |

**H1 compares proportions, not means.** The outcome is binary, so the question
is whether the *default rate* differs between utilisation groups. A t-test
assumes roughly normal data; a 0/1 Bernoulli variable is not, and the
t-statistic it returns does not mean what it appears to.

**H3 uses a rank test.** Bill volatility is a standard deviation — strictly
positive and heavily right-skewed — so medians and Mann-Whitney are reported
rather than means and a t-test.

**Effect sizes accompany every p-value.** At n = 30,000 nearly any difference
reaches significance, so φ, rank-biserial correlation and relative risk are
reported alongside; where an effect is negligible, the output says so.

## Modelling

```bash
python -m src.modeling
```

Logistic Regression as the baseline, then a Random Forest that has to beat it.
70/30 stratified split, `random_state=42`.

Features: `AVG_UTIL`, `AVG_PAY_DELAY`, `DELAY_TREND`, `BILL_VOLATILITY`,
`MAX_DELAY_STREAK`, `AGE`, `LIMIT_BAL`.

**Both models use `class_weight="balanced"`.** With 78% of clients not
defaulting, an unweighted model maximises accuracy by predicting "no default"
almost always — respectable accuracy, near-useless recall on the class that
actually matters.

**The decision threshold is tuned, not assumed at 0.5.** With the positive
class at 22%, the default cut-off trades away most of the recall that makes the
model useful. The scripts report F1 at both 0.5 and the tuned threshold.

**ROC-AUC is reported with PR-AUC and per-class precision/recall.** On an
imbalanced problem ROC-AUC alone flatters a model; precision-recall on the
default class is the honest view.

Outputs `figures/model_curves.png` (ROC and PR for both models) and
`figures/feature_importance.png`.

## Data

The raw file is **not committed**. Download the *Default of Credit Card
Clients* dataset from
[UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
and save it as `Credit_card.xlsx` in the repository root. Both scripts fail
with that instruction if it is missing.

### Source columns

| Column | Meaning |
|---|---|
| `LIMIT_BAL` | Credit limit in NT dollars |
| `SEX` | 1 = male, 2 = female |
| `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = other |
| `MARRIAGE` | 1 = married, 2 = single, 3 = other |
| `AGE` | Years |
| `PAY_0`…`PAY_6` | Repayment status, September → April. −1 = paid in full, 0 = revolving credit, *n* = *n* months late |
| `BILL_AMT1`…`6` | Monthly bill, September → April |
| `PAY_AMT1`…`6` | Monthly payment, September → April |
| `default.payment.next.month` | 1 = defaulted, 0 = repaid |

## Structure

```
src/features.py           risk metrics from raw history
src/hypothesis_tests.py   H1-H3 and their figures
src/modeling.py           both classifiers, curves, importance
figures/                  generated output
Credit_card.pbix          Power BI dashboard
requirements.txt          pinned versions
```

## Limitations

- **One market, six months, 2005.** Taiwanese credit behaviour two decades ago;
  nothing here transfers to another market or period without revalidation.
- **Prediction, not explanation.** The models rank who is likely to default.
  They do not establish why, and feature importance is not a causal claim.
- **Modest separation.** These features leave the classes substantially
  overlapping. The output is useful for prioritising review, not for automated
  decisions about individual clients.
- **Demographic fields are present in the data.** `SEX`, `EDUCATION` and
  `MARRIAGE` are deliberately excluded from the model features. Using them to
  decide credit outcomes would be discriminatory regardless of predictive
  value; they appear in the dashboard for descriptive purposes only.
- **`EDUCATION` and `MARRIAGE` contain undocumented codes** (0, 5, 6) not
  covered by the source description.
