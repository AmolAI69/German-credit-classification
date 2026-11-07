"""Phases 1b-4: split, preprocess, train 5 models, evaluate, save.

Run from the project root:   python src/train.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, joblib, sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay)
from preprocessing import (load_raw, build_preprocessor, FEATURE_NAMES, TARGET_ORIGINAL,
                           NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

RANDOM_STATE = 42
DATA_PATH, MODEL_DIR, OUT_DIR = "data/german.data", "model", "outputs"
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

# ======================= PHASE 1: preprocessing =======================
print("=" * 70, "\nPHASE 1: SPLIT + PREPROCESSING\n" + "=" * 70)
df = load_raw(DATA_PATH)
assert df.shape == (1000, 21) and df.isna().sum().sum() == 0
assert set(df[TARGET_ORIGINAL].unique()) == {1, 2}

X = df[FEATURE_NAMES].copy()
# Re-encode target: Bad credit risk (original 2) -> 1 (POSITIVE class); Good (1) -> 0
y = (df[TARGET_ORIGINAL] == 2).astype(int).rename("target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("y_train:", y_train.shape, "| y_test:", y_test.shape)
print("Bad-risk share  full: %.3f | train: %.3f | test: %.3f"
      % (y.mean(), y_train.mean(), y_test.mean()))
print("Train class counts:", y_train.value_counts().sort_index().to_dict(),
      "| Test class counts:", y_test.value_counts().sort_index().to_dict())

# Demonstration of leakage-free preprocessing: fit on TRAIN only
pre = build_preprocessor().fit(X_train)
Xtr_t, Xte_t = pre.transform(X_train), pre.transform(X_test)
names = pre.get_feature_names_out()
print("\nTransformed train shape:", Xtr_t.shape, "| test shape:", Xte_t.shape)
print("  numerical columns: %d | one-hot columns: %d" %
      (len(NUMERICAL_FEATURES), Xtr_t.shape[1] - len(NUMERICAL_FEATURES)))
print("  NaNs in transformed train/test:", int(np.isnan(Xtr_t).sum()), "/", int(np.isnan(Xte_t).sum()))
scaler = pre.named_transformers_["num"]
print("  Scaler mean_ learned from TRAIN credit_amount: %.2f (train raw mean %.2f, full-data mean %.2f)"
      % (scaler.mean_[1], X_train.credit_amount.mean(), X.credit_amount.mean()))
print("  Scaled TRAIN numeric mean ~0 / std ~1:",
      np.round(Xtr_t[:, :7].mean(0), 3), np.round(Xtr_t[:, :7].std(0), 3))
print("  Scaled TEST numeric mean (not exactly 0, as expected):", np.round(Xte_t[:, :7].mean(0), 2))

# ======================= PHASE 2: models =======================
print("\n" + "=" * 70, "\nPHASE 2: TRAIN FIVE MODELS\n" + "=" * 70)
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                            random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Gaussian Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
               random_state=RANDOM_STATE),
}
FILENAMES = {"Logistic Regression": "logistic_regression.joblib",
             "Decision Tree": "decision_tree.joblib", "KNN": "knn.joblib",
             "Gaussian Naive Bayes": "naive_bayes.joblib", "SVM": "svm.joblib"}

pipelines = {}
for name, clf in classifiers.items():
    pipe = Pipeline([("preprocess", build_preprocessor()), ("classifier", clf)])
    pipe.fit(X_train, y_train)               # preprocessing fitted on training data only
    pipelines[name] = pipe
    assert list(pipe.named_steps["classifier"].classes_) == [0, 1]
    print(f"Trained {name:<22} | params: {clf.get_params(deep=False)}")
print("All classifiers have classes_ == [0, 1]  (column 1 of predict_proba = Bad risk)")

# ======================= PHASE 3: evaluation =======================
print("\n" + "=" * 70, "\nPHASE 3: EVALUATION ON HELD-OUT TEST SET (positive class = Bad risk)\n" + "=" * 70)

def evaluate(pipe, X_eval, y_eval):
    """Return the six metrics; positive class = 1 (Bad credit risk)."""
    y_pred = pipe.predict(X_eval)
    y_score = pipe.predict_proba(X_eval)[:, 1]          # P(Bad risk)
    return {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "AUC": roc_auc_score(y_eval, y_score),
        "Precision": precision_score(y_eval, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, pos_label=1),
        "F1-score": f1_score(y_eval, y_pred, pos_label=1),
        "MCC": matthews_corrcoef(y_eval, y_pred),
    }, y_pred, y_score

rows, preds, scores = {}, {}, {}
for name, pipe in pipelines.items():
    try:
        rows[name], preds[name], scores[name] = evaluate(pipe, X_test, y_test)
    except Exception as e:
        raise RuntimeError(f"Metric calculation failed for {name}: {e}")
results = pd.DataFrame(rows).T.round(4)
results.index.name = "Model"
print(results.to_string())
results.to_csv(f"{OUT_DIR}/metrics_comparison.csv")

print("\nConfusion matrices (rows = actual, cols = predicted; order: Good, Bad)")
cms = {}
for name in pipelines:
    cms[name] = confusion_matrix(y_test, preds[name], labels=[0, 1])
    tn, fp, fn, tp = cms[name].ravel()
    print(f"{name:<22} TN={tn:3d} FP={fp:3d} FN={fn:3d} TP={tp:3d}  "
          f"(bad applicants missed: {fn} of {fn+tp})")
print("\nBaseline: always predicting 'Good' gives accuracy = %.3f" % (1 - y_test.mean()))

# --- plots ---
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150); plt.close()

plt.figure(figsize=(6.5, 5.5))
for name in pipelines:
    fpr, tpr, _ = roc_curve(y_test, scores[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={rows[name]['AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall for Bad risk)")
plt.title("ROC curves - positive class: Bad credit risk"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150); plt.close()

results.plot(kind="bar", figsize=(11, 5), width=0.8)
plt.title("Metric comparison on held-out test set"); plt.ylim(0, 1); plt.xticks(rotation=20)
plt.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/metric_comparison.png", dpi=150); plt.close()
print("Saved plots to", OUT_DIR)

# ======================= PHASE 4: save artifacts =======================
print("\n" + "=" * 70, "\nPHASE 4: SAVE MODELS + test_data.csv\n" + "=" * 70)
for name, pipe in pipelines.items():
    joblib.dump(pipe, os.path.join(MODEL_DIR, FILENAMES[name]))

# test_data.csv: 20 features + actual target, using ORIGINAL labels (1 = Good, 2 = Bad)
test_df = X_test.copy()
test_df[TARGET_ORIGINAL] = np.where(y_test == 1, 2, 1)
test_df.to_csv("test_data.csv", index=False)
print("test_data.csv shape:", test_df.shape)

# Metadata artifact so the app uses the exact same column order / label mapping
meta = {"feature_names": FEATURE_NAMES, "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES, "target_column": TARGET_ORIGINAL,
        "label_mapping": {"1": "Good credit risk", "2": "Bad credit risk (positive class)"},
        "model_files": FILENAMES, "random_state": RANDOM_STATE, "test_size": 0.2,
        "sklearn_version": sklearn.__version__}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# Verification: reload from disk, predict on the saved CSV, compare with in-memory results
print("\nVERIFYING saved models (reload -> predict on test_data.csv)")
chk = pd.read_csv("test_data.csv")
Xc = chk[FEATURE_NAMES]; yc = (chk[TARGET_ORIGINAL] == 2).astype(int)
assert list(chk.columns) == FEATURE_NAMES + [TARGET_ORIGINAL]
all_ok = True
for name, fn in FILENAMES.items():
    loaded = joblib.load(os.path.join(MODEL_DIR, fn))
    m, p, s = evaluate(loaded, Xc, yc)
    same_pred = np.array_equal(p, preds[name])
    same_metrics = all(abs(m[k] - rows[name][k]) < 1e-12 for k in m)
    ok = same_pred and same_metrics; all_ok &= ok
    print(f"  {fn:<26} loads OK | predictions identical: {same_pred} | metrics identical: {same_metrics}")
print("ALL SAVED MODELS VERIFIED:", all_ok)
print("\nFiles in model/:", sorted(os.listdir(MODEL_DIR)))
