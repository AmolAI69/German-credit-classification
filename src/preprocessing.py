"""Shared preprocessing helpers for the German Credit project.

Used by train.py (and later by app.py) so that training and the Streamlit
app use exactly the same column names, feature order and preprocessing.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column names in the exact order given in the official UCI documentation
# (german.doc). The raw file german.data has NO header row.
FEATURE_NAMES = [
    "status_checking_account",   # Attr 1  (categorical)
    "duration_months",           # Attr 2  (numerical)
    "credit_history",            # Attr 3  (categorical)
    "purpose",                   # Attr 4  (categorical)
    "credit_amount",             # Attr 5  (numerical)
    "savings_account",           # Attr 6  (categorical)
    "employment_since",          # Attr 7  (categorical)
    "installment_rate",          # Attr 8  (numerical, 1-4)
    "personal_status_sex",       # Attr 9  (categorical)
    "other_debtors",             # Attr 10 (categorical)
    "residence_since",           # Attr 11 (numerical, 1-4)
    "property",                  # Attr 12 (categorical)
    "age_years",                 # Attr 13 (numerical)
    "other_installment_plans",   # Attr 14 (categorical)
    "housing",                   # Attr 15 (categorical)
    "existing_credits",          # Attr 16 (numerical)
    "job",                       # Attr 17 (categorical)
    "people_liable",             # Attr 18 (numerical)
    "telephone",                 # Attr 19 (categorical)
    "foreign_worker",            # Attr 20 (categorical)
]
TARGET_ORIGINAL = "credit_risk"   # original labels: 1 = Good, 2 = Bad
TARGET = "target"                 # our label: 1 = Bad (positive class), 0 = Good

NUMERICAL_FEATURES = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "age_years", "existing_credits", "people_liable",
]
CATEGORICAL_FEATURES = [c for c in FEATURE_NAMES if c not in NUMERICAL_FEATURES]


def load_raw(path):
    """Load the original space-separated german.data file."""
    return pd.read_csv(path, sep=r"\s+", header=None,
                       names=FEATURE_NAMES + [TARGET_ORIGINAL])


def build_preprocessor():
    """One-hot encode categorical columns, standardise numerical columns.

    Fitted ONLY on training data when used inside a Pipeline.
    handle_unknown='ignore' makes the app robust to unseen categories.
    sparse_output=False gives a dense array (needed by GaussianNB).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
