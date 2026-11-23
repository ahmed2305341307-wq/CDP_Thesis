
# ============================================================
# PHASE 5: EXPLAINABLE AI (SHAP + LIME)
# Chronic Kidney Disease Prediction
# ============================================================

import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = RESULTS_DIR / "optimized_models.joblib"
FEATURE_DATA_PATH = RESULTS_DIR / "hybrid_feature_data.joblib"

OUTPUT_DIR = RESULTS_DIR / "phase5_explainability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n================================================")
print("PHASE 5: EXPLAINABLE AI")
print("================================================")

# ------------------------------------------------------------
# 2. LOAD TRAINED MODELS AND DATA
# ------------------------------------------------------------

print("\nLoading trained models and feature data...")

models = joblib.load(MODEL_PATH)
data = joblib.load(FEATURE_DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

feature_names = [str(name) for name in data["selected_features"]]

print("Models loaded successfully!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Selected features:", len(feature_names))

if X_train.shape[1] != len(feature_names):
    raise ValueError("Feature count does not match selected feature names!")

# ------------------------------------------------------------
# 3. SELECT MODELS
# ------------------------------------------------------------

xgb_model = models["Tuned XGBoost"]
voting_model = models["Tuned Voting Ensemble"]

print("\nSelected models:")
print("1. Tuned XGBoost - SHAP")
print("2. Tuned Voting Ensemble - LIME")

# ------------------------------------------------------------
# 4. SHAP EXPLAINABILITY
# ------------------------------------------------------------

print("\n================================================")
print("SHAP EXPLAINABILITY")
print("================================================")

print("\nCreating SHAP TreeExplainer...")

shap_explainer = shap.TreeExplainer(xgb_model)

shap_values = shap_explainer(X_test)

print("SHAP values calculated successfully!")

# Convert test data into DataFrame for readable feature names
X_test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)

# ------------------------------------------------------------
# 5. SHAP SUMMARY PLOT
# ------------------------------------------------------------

print("\nGenerating SHAP Summary Plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_test_df,
    feature_names=feature_names,
    show=False
)

plt.title("SHAP Summary Plot - Tuned XGBoost")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_summary_plot.png")

# ------------------------------------------------------------
# 6. SHAP FEATURE IMPORTANCE BAR PLOT
# ------------------------------------------------------------

print("\nGenerating SHAP Feature Importance...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_test_df,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance - Tuned XGBoost")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_feature_importance.png")

# ------------------------------------------------------------
# 7. SHAP WATERFALL PLOT
# ------------------------------------------------------------

print("\nGenerating SHAP Waterfall Plot...")

plt.figure(figsize=(12, 8))

shap.plots.waterfall(
    shap_values[0],
    show=False
)

plt.title("SHAP Individual Prediction Explanation")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_waterfall_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_waterfall_plot.png")

# ------------------------------------------------------------
# 8. SAVE SHAP FEATURE IMPORTANCE AS CSV
# ------------------------------------------------------------

print("\nSaving SHAP feature importance data...")

importance_values = shap_values.values

# Handle binary/multiclass output if SHAP returns 3 dimensions
if importance_values.ndim == 3:
    importance_values = importance_values[:, :, 1]

mean_importance = abs(importance_values).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP_Value": mean_importance
})

shap_importance_df = shap_importance_df.sort_values(
    by="Mean_Absolute_SHAP_Value",
    ascending=False
)

shap_importance_df.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False
)

print(shap_importance_df.to_string(index=False))
print("Saved: shap_feature_importance.csv")

# ------------------------------------------------------------
# 9. LIME EXPLAINABILITY
# ------------------------------------------------------------

print("\n================================================")
print("LIME EXPLAINABILITY")
print("================================================")

print("\nCreating LIME Tabular Explainer...")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=["Not CKD", "CKD"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

print("LIME Explainer created successfully!")

# ------------------------------------------------------------
# 10. EXPLAIN ONE TEST PREDICTION
# ------------------------------------------------------------

print("\nGenerating LIME explanation...")

sample_index = 0

lime_explanation = lime_explainer.explain_instance(
    data_row=X_test[sample_index],
    predict_fn=voting_model.predict_proba,
    num_features=15,
    top_labels=2
)

# Save LIME explanation as HTML
lime_explanation.save_to_file(
    str(OUTPUT_DIR / "lime_explanation.html")
)

print("Saved: lime_explanation.html")

# ------------------------------------------------------------
# 11. SAVE LIME EXPLANATION PLOT
# ------------------------------------------------------------

print("\nGenerating LIME explanation plot...")

lime_figure = lime_explanation.as_pyplot_figure(
    label=1
)

lime_figure.set_size_inches(12, 8)
lime_figure.tight_layout()

lime_figure.savefig(
    OUTPUT_DIR / "lime_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(lime_figure)

print("Saved: lime_explanation.png")

# ------------------------------------------------------------
# 12. SAVE LIME EXPLANATION DATA
# ------------------------------------------------------------

lime_results = lime_explanation.as_list(label=1)

lime_df = pd.DataFrame(
    lime_results,
    columns=["Feature_Condition", "LIME_Weight"]
)

lime_df.to_csv(
    OUTPUT_DIR / "lime_explanation.csv",
    index=False
)

print("\nLIME Explanation:")
print(lime_df.to_string(index=False))

# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n================================================")
print("PHASE 5 COMPLETED SUCCESSFULLY!")
print("================================================")

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nAll Phase 5 results saved in:")
print(OUTPUT_DIR)

print("\n================================================")
