import pandas as pd
import numpy as np

df = pd.read_excel('Credit_card.xlsx')

"""---------------------------------"""
"""Feature Engineering (New Metrics)"""
"""---------------------------------"""
pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
payamt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

# Average delay
df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)

# Delay trend (slope)
def delay_trend(row):
    y = row.values
    x = np.arange(len(y))
    if np.all(pd.isna(y)):
        return np.nan
    return np.polyfit(x, y, 1)[0]

df['DELAY_TREND'] = df[pay_cols].apply(delay_trend, axis=1)

# Utilization
for i in range(1, 7):
    df[f'UTIL_{i}'] = df[f'BILL_AMT{i}'] / df['LIMIT_BAL']

df['AVG_UTIL'] = df[[f'UTIL_{i}' for i in range(1,7)]].mean(axis=1)

# Payment to bill ratio
for i in range(1, 7):
    df[f'PAY_RATIO_{i}'] = df[f'PAY_AMT{i}'] / df[f'BILL_AMT{i}']

df['AVG_PAY_RATIO'] = df[[f'PAY_RATIO_{i}' for i in range(1,7)]].mean(axis=1)

# Balance volatility
df['BILL_VOLATILITY'] = df[bill_cols].std(axis=1)

# Consecutive delay streak
def max_delay_streak(row):
    streak = max_streak = 0
    for v in row:
        if v >= 2:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

df['MAX_DELAY_STREAK'] = df[pay_cols].apply(max_delay_streak, axis=1)

# Shock in last month -> A large positive value = sudden spike in spending/debt
df['LAST_MONTH_SHOCK'] = df['BILL_AMT1'] - df[bill_cols[1:]].mean(axis=1)
df.head(50).to_csv('Feature_selected_credits_sample.csv', index=False)

""""-----------------"""
"""Hypothesis Testing"""
""""-----------------"""
#H1: High utilization increase default
from scipy.stats import ttest_ind

high_util = df[df['AVG_UTIL'] > 0.8]['default.payment.next.month']
low_util = df[df['AVG_UTIL'] <= 0.8]['default.payment.next.month']

# Welch's t-test
Welch_test = ttest_ind(high_util, low_util, equal_var=False)

print("Welch test: ", Welch_test)
### t>0 & p<0.05 -> High utilization customers default significantly more

#H2: Increasing delay trend -> more default
from scipy.stats import pointbiserialr

Pearson_corr = pointbiserialr(df['DELAY_TREND'], df['default.payment.next.month'])
print("Pearson correlation: ", Pearson_corr)
### r<0 & p<0.05 -> Worsening delay trend->fewer defaults(counterintuitive)

#H3: Volatility differs by default group
defaulted = df[df['default.payment.next.month'] == 1]['BILL_VOLATILITY']
not_defaulted = df[df['default.payment.next.month'] == 0]['BILL_VOLATILITY']

Welch_test3 = ttest_ind(defaulted, not_defaulted, equal_var=False)
print("Welch test2: ", Welch_test3)
### t<0 & p<0.05 -> Defaulters have significantly lower bill volatility

""""-----------------------------"""
"""Modeling (Logistic Regression)"""
""""-----------------------------"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

features = [
    'AVG_UTIL', 'AVG_PAY_DELAY', 'DELAY_TREND',
    'BILL_VOLATILITY', 'MAX_DELAY_STREAK', 'AGE', 'LIMIT_BAL'
]

X = df[features]
y = df['default.payment.next.month']

X_train, X_Test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_Test_scaled = scaler.transform(X_Test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_Test_scaled)
y_prob = model.predict_proba(X_Test_scaled)[:, 1]

print("AUC:", roc_auc_score(y_test, y_prob)) #-> 74% is not bad
print(classification_report(y_test, y_pred)) #-> misses too many actual defaulters

"""--------------------------"""
"""Tree Model (Random Forest)"""
"""--------------------------"""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_Test)[:, 1]

print("RF AUC:", roc_auc_score(y_test, rf_prob)) #-> 77% now is good
