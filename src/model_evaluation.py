
# ============================================================
# PHASE 6: MODEL EVALUATION & PERFORMANCE COMPARISON
# Chronic Kidney Disease Prediction
# ============================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = RESULTS_DIR / "optimized_models.joblib"
DATA_PATH = RESULTS_DIR / "hybrid_feature_data.joblib"

OUTPUT_DIR = RESULTS_DIR / "phase6_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n================================================")
print("PHASE 6: MODEL EVALUATION")
print("================================================")

# ------------------------------------------------------------
# 2. LOAD MODELS AND TEST DATA
# ------------------------------------------------------------

print("\nLoading optimized models and test data...")

models = joblib.load(MODEL_PATH)
data = joblib.load(DATA_PATH)

X_test = data["X_test"]
y_test = np.asarray(data["y_test"]).ravel()

print("Models loaded successfully!")
print("Testing shape:", X_test.shape)
print("Testing samples:", len(y_test))

# ------------------------------------------------------------
# 3. SELECT FOUR OPTIMIZED MODELS
# ------------------------------------------------------------

model_names = [
    "Tuned Random Forest",
    "Tuned SVM",
    "Tuned XGBoost",
    "Tuned Voting Ensemble"
]

for name in model_names:
    if name not in models:
        raise KeyError(f"Model not found: {name}")

print("\nModels selected:")
for name in model_names:
    print("-", name)

# ------------------------------------------------------------
# 4. MODEL EVALUATION
# ------------------------------------------------------------

print("\n================================================")
print("MODEL PERFORMANCE EVALUATION")
print("================================================")

metrics_results = []
all_predictions = {}
roc_results = {}

for name in model_names:

    print(f"\nEvaluating: {name}")
    model = models[name]

    # Prediction
    y_pred = np.asarray(model.predict(X_test)).ravel()

    # Probability / decision scores for ROC-AUC
    y_score = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)

        if probabilities.ndim == 2:
            classes = list(model.classes_)
            positive_index = classes.index(1)
            y_score = probabilities[:, positive_index]
        else:
            y_score = probabilities.ravel()

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)

        if np.ndim(scores) == 2:
            classes = list(model.classes_)
            positive_index = classes.index(1)
            y_score = scores[:, positive_index]
        else:
            y_score = np.asarray(scores).ravel()

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    auc = np.nan

    if y_score is not None and len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_score)

        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_results[name] = (fpr, tpr, auc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Save predictions
    all_predictions[name] = y_pred

    metrics_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    })

    print("Accuracy  :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1-Score  :", round(f1, 4))
    print("ROC-AUC   :", round(auc, 4) if not np.isnan(auc) else "N/A")

    # Save classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not CKD", "CKD"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    safe_name = name.lower().replace(" ", "_")
    report_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_classification_report.csv"
    )

# ------------------------------------------------------------
# 5. SAVE PERFORMANCE COMPARISON CSV
# ------------------------------------------------------------

print("\nSaving performance comparison...")

metrics_df = pd.DataFrame(metrics_results)

metrics_df = metrics_df.sort_values(
    by="F1_Score",
    ascending=False
)

metrics_df.to_csv(
    OUTPUT_DIR / "model_performance_comparison.csv",
    index=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print(metrics_df.to_string(index=False))

# ------------------------------------------------------------
# 6. CONFUSION MATRIX PLOTS
# ------------------------------------------------------------

print("\nGenerating Confusion Matrix plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, name in zip(axes.ravel(), model_names):

    cm = confusion_matrix(
        y_test,
        all_predictions[name],
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not CKD", "CKD"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(name)

plt.suptitle("Confusion Matrix Comparison")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: confusion_matrix_comparison.png")

# ------------------------------------------------------------
# 7. ROC CURVE COMPARISON
# ------------------------------------------------------------

print("\nGenerating ROC Curve...")

plt.figure(figsize=(10, 8))

for name, (fpr, tpr, auc) in roc_results.items():

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: roc_curve_comparison.png")

# ------------------------------------------------------------
# 8. PERFORMANCE COMPARISON GRAPH
# ------------------------------------------------------------

print("\nGenerating Performance Comparison Graph...")

plot_df = metrics_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1_Score"]
]

ax = plot_df.plot(
    kind="bar",
    figsize=(13, 7),
    ylim=(0, 1.05),
    width=0.8
)

ax.set_title("Performance Comparison of Optimized Models")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Models")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.grid(axis="y", alpha=0.3)

plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: model_performance_comparison.png")

# ------------------------------------------------------------
# 9. IDENTIFY BEST MODEL
# ------------------------------------------------------------

best_model = metrics_df.iloc[0]

print("\n================================================")
print("BEST MODEL RESULT")
print("================================================")

print("Best Model :", best_model["Model"])
print("Accuracy   :", round(best_model["Accuracy"], 4))
print("Precision  :", round(best_model["Precision"], 4))
print("Recall     :", round(best_model["Recall"], 4))
print("F1-Score   :", round(best_model["F1_Score"], 4))
print("ROC-AUC    :", round(best_model["ROC_AUC"], 4)
      if pd.notna(best_model["ROC_AUC"]) else "N/A")

# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\n================================================")
print("PHASE 6 COMPLETED SUCCESSFULLY!")
print("================================================")

print("\nGenerated Files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nAll Phase 6 results saved in:")
print(OUTPUT_DIR)

print("\n================================================")
