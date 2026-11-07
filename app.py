"""Streamlit app: evaluate saved German Credit classifiers on an uploaded test CSV.

The app ONLY loads already-trained pipelines from model/ and calls predict().
It never trains a model and never fits any preprocessing on the uploaded data.
Run locally with:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1-score", "MCC"]


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
@st.cache_resource
def load_metadata():
    """Read feature order, target name and model file names saved at training time."""
    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(file_name):
    """Load one saved pipeline (preprocessing + classifier). No training happens here."""
    return joblib.load(MODEL_DIR / file_name)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_upload(df, meta):
    """Check the uploaded dataframe.

    Returns (X, y_bad, warnings) where
      X      = the 20 feature columns in the exact training order (target NOT included),
      y_bad  = actual target as 1 = Bad credit risk (positive class), 0 = Good,
      warnings = list of non-fatal messages.
    Raises ValueError with a readable message for fatal problems.
    """
    features, target = meta["feature_names"], meta["target_column"]
    warnings = []

    if df.empty:
        raise ValueError("The uploaded file has no rows.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    extra = [c for c in df.columns if c not in features + [target]]
    if extra:
        warnings.append("Extra columns were ignored: " + ", ".join(extra))

    df = df[features + [target]]            # exact training column order
    na_cols = df.columns[df.isna().any()].tolist()
    if na_cols:
        raise ValueError("Missing values found in column(s): " + ", ".join(na_cols)
                         + ". The saved models cannot handle missing values; please fill or remove them.")

    # Numerical features must really be numbers
    for col in meta["numerical_features"]:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            raise ValueError(f"Column '{col}' must contain only numbers.")
        df[col] = converted

    # Categorical codes: tidy whitespace
    for col in meta["categorical_features"]:
        df[col] = df[col].astype(str).str.strip()

    # Target must use the ORIGINAL labels: 1 = Good, 2 = Bad
    target_num = pd.to_numeric(df[target], errors="coerce")
    bad_values = sorted(df.loc[target_num.isna() | ~target_num.isin([1, 2]), target].astype(str).unique())
    if bad_values:
        raise ValueError(f"Target column '{target}' must contain only 1 (Good) or 2 (Bad). "
                         f"Found other values: {', '.join(bad_values[:10])}")
    y_bad = (target_num == 2).astype(int).to_numpy()

    X = df[features]                        # the target column is NOT part of X
    return X, y_bad, warnings


def unseen_categories(pipeline, X, meta):
    """List categorical codes in the upload that the model never saw in training."""
    try:
        encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        notes = []
        for col, known in zip(meta["categorical_features"], encoder.categories_):
            new = sorted(set(X[col]) - set(known))
            if new:
                notes.append(f"{col}: {', '.join(new)}")
        return notes
    except Exception:
        return []


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(pipeline, X, y_bad):
    """Predict with the saved pipeline and compute the six metrics (positive class = Bad)."""
    classes = list(pipeline.named_steps["classifier"].classes_)
    if classes != [0, 1]:
        raise ValueError(f"Unexpected class order in saved model: {classes}")
    y_pred = pipeline.predict(X)
    p_bad = pipeline.predict_proba(X)[:, 1]            # probability of Bad credit risk
    both_classes = len(np.unique(y_bad)) == 2
    metrics = {
        "Accuracy": accuracy_score(y_bad, y_pred),
        "AUC": roc_auc_score(y_bad, p_bad) if both_classes else None,
        "Precision": precision_score(y_bad, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_bad, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_bad, y_pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_bad, y_pred),
    }
    return metrics, y_pred, p_bad, both_classes


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


# --------------------------------------------------------------------------
# User interface
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="German Credit Risk Classifier", layout="wide")
    st.title("German Credit Risk - Model Evaluation App")
    st.write(
        "Upload a **test CSV** (20 input features + the actual target column), choose one of the "
        "five **already-trained** classifiers, and see its evaluation metrics on your file. "
        "Models are loaded from disk and are **never retrained** on your data."
    )
    st.caption("Positive class = Bad credit risk (original label 2). Academic project only - "
               "not for real credit decisions.")

    try:
        meta = load_metadata()
    except Exception as e:
        st.error(f"Could not load model/metadata.json: {e}")
        st.stop()

    if meta.get("sklearn_version") != sklearn.__version__:
        st.warning(f"Models were trained with scikit-learn {meta.get('sklearn_version')} but this "
                   f"app runs {sklearn.__version__}. Loading may fail or behave differently.")

    with st.expander("Required CSV format"):
        st.write("Columns needed (any order): " + ", ".join(meta["feature_names"])
                 + f", **{meta['target_column']}** (actual target: 1 = Good, 2 = Bad).")
        st.write("The file `test_data.csv` in the GitHub repository follows this format.")

    col_a, col_b = st.columns(2)
    with col_a:
        uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    with col_b:
        model_name = st.selectbox("Select a classification model", list(meta["model_files"].keys()))

    # Reference results from training time (a different thing from the uploaded-file results)
    ref_path = BASE_DIR / "outputs" / "metrics_comparison.csv"
    if ref_path.exists():
        with st.expander("Reference: results recorded during training (held-out split of 200 rows)"):
            st.caption("These numbers were computed once at training time. "
                       "They are NOT the results for your uploaded file.")
            st.dataframe(pd.read_csv(ref_path))

    if uploaded is None:
        st.info("Please upload a CSV file to begin.")
        return

    # ---- read + validate -------------------------------------------------
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the file as CSV: {e}")
        return
    try:
        X, y_bad, warnings = validate_upload(raw, meta)
    except ValueError as e:
        st.error(str(e))
        return
    for w in warnings:
        st.warning(w)

    st.success(f"File accepted: {len(X)} rows, {X.shape[1]} input features. "
               f"Actual labels: {int((y_bad == 0).sum())} Good, {int(y_bad.sum())} Bad.")

    # ---- load + evaluate -------------------------------------------------
    try:
        pipeline = load_model(meta["model_files"][model_name])
    except Exception as e:
        st.error(f"Could not load the saved model '{model_name}': {e}")
        return

    new_cats = unseen_categories(pipeline, X, meta)
    if new_cats:
        st.warning("Category codes not seen in training (treated as all-zero): " + "; ".join(new_cats))

    try:
        metrics, y_pred, p_bad, both_classes = evaluate(pipeline, X, y_bad)
    except Exception as e:
        st.error(f"Prediction or metric calculation failed: {e}")
        return
    if not both_classes:
        st.warning("The uploaded file contains only one actual class, so AUC cannot be computed "
                   "and other metrics may be unreliable.")

    st.header(f"Results on your uploaded file - {model_name}")
    first, second = st.columns(3), st.columns(3)
    for slot, name in zip(first + second, METRIC_NAMES):
        slot.metric(name, fmt(metrics[name]))

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_bad, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y_bad, y_pred, labels=[0, 1], target_names=["Good", "Bad"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4))

    with st.expander("Predictions (first 20 rows)"):
        out = pd.DataFrame({
            "Actual": np.where(y_bad == 1, "Bad", "Good"),
            "Predicted": np.where(y_pred == 1, "Bad", "Good"),
            "P(Bad risk)": p_bad.round(4),
        })
        st.dataframe(out.head(20))
        st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")

    with st.expander("Compare all five models on your uploaded file"):
        rows = {}
        for name, file_name in meta["model_files"].items():
            try:
                m, _, _, _ = evaluate(load_model(file_name), X, y_bad)
                rows[name] = {k: fmt(v) for k, v in m.items()}
            except Exception as e:
                rows[name] = {k: "error" for k in METRIC_NAMES}
                st.error(f"{name}: {e}")
        st.dataframe(pd.DataFrame(rows).T)


main()
