"""Phase 1a: inspect the raw dataset (no modelling yet)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from preprocessing import *

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
df = load_raw("data/german.data")

print("SHAPE:", df.shape)
print("\nFIRST 5 ROWS\n", df.head())
print("\nDTYPES\n", df.dtypes)
print("\nMISSING VALUES per column (total =", df.isna().sum().sum(), ")")
print("DUPLICATE ROWS:", df.duplicated().sum())
print("\nNUMERICAL FEATURES:", NUMERICAL_FEATURES)
print("CATEGORICAL FEATURES:", CATEGORICAL_FEATURES)
print("\nTARGET DISTRIBUTION (1=Good, 2=Bad)")
print(df[TARGET_ORIGINAL].value_counts().sort_index())
print((df[TARGET_ORIGINAL].value_counts(normalize=True).sort_index()*100).round(1))
print("\nUNIQUE VALUES in numerical columns")
for c in NUMERICAL_FEATURES: print(f"  {c}: {df[c].nunique()} unique, values e.g. {sorted(df[c].unique())[:8]}")
print("\nNUMERICAL SUMMARY\n", df[NUMERICAL_FEATURES].describe().round(2).T)
print("\nSKEWNESS\n", df[NUMERICAL_FEATURES].skew().round(2))
print("\nCATEGORICAL DISTRIBUTIONS")
for c in CATEGORICAL_FEATURES:
    print(f"\n{c} ({df[c].nunique()} levels)")
    print(df[c].value_counts().to_string())
print("\nBad-risk rate by category (%), quick look at signal")
for c in ["status_checking_account","credit_history","foreign_worker"]:
    print((df.groupby(c)[TARGET_ORIGINAL].apply(lambda s:(s==2).mean()*100).round(1)).to_string(), "\n")
print("Whitespace/odd values check:", {c:int(df[c].astype(str).str.contains(r'\s|\?|nan').sum()) for c in CATEGORICAL_FEATURES if df[c].astype(str).str.contains(r'\s|\?|nan').sum()})
