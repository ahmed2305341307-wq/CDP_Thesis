
import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from stacking_architecture import (build_hierarchical_stacking, build_single_stacking, build_soft_voting)
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_ITER = 15
CV_FOLDS = 5

DATA_PATH = "results/hybrid_feature_data.joblib"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading hybrid feature data...")

    data = joblib.load(DATA_PATH)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ============================================================
# CROSS VALIDATION
# ============================================================

def create_cv():

    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

def tune_model(name, model, parameters, X_train, y_train):

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION:", name)
    print("=" * 60)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=N_ITER,
        scoring="f1",
        cv=create_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise"
    )

    search.fit(X_train, y_train)

    print("\nBest CV F1-Score:", round(search.best_score_, 4))
    print("Best Parameters:")

    for key, value in search.best_params_.items():
        print(f"{key}: {value}")

    return search


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, model, X_test, y_test):

    print("\n" + "=" * 60)
    print("TEST EVALUATION:", name)
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Not CKD", "CKD"],
            zero_division=0
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("Confusion Matrix:")
    print(cm)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }, cm


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(name, cm):

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not CKD", "CKD"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    display.plot(ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")

    fig.tight_layout()

    filename = name.lower().replace(" ", "_")

    path = os.path.join(
        RESULTS_DIR,
        f"tuned_confusion_matrix_{filename}.png"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)

    print("Saved:", path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHASE 4: HYPERPARAMETER OPTIMIZATION")
    print("Chronic Kidney Disease Prediction")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # --------------------------------------------------------
    # RANDOM FOREST SEARCH SPACE
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    rf_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    # --------------------------------------------------------
    # SVM SEARCH SPACE
    # --------------------------------------------------------

    svm = SVC(
        probability=True,
        random_state=RANDOM_STATE
    )

    svm_params = {
        "C": [0.1, 1, 5, 10, 50, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"]
    }

    # --------------------------------------------------------
    # XGBOOST SEARCH SPACE
    # --------------------------------------------------------

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    xgb_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1, 5, 10]
    }

    # --------------------------------------------------------
    # OPTIMIZE MODELS
    # --------------------------------------------------------

    rf_search = tune_model(
        "Random Forest",
        rf,
        rf_params,
        X_train,
        y_train
    )

    svm_search = tune_model(
        "SVM",
        svm,
        svm_params,
        X_train,
        y_train
    )

    xgb_search = tune_model(
        "XGBoost",
        xgb,
        xgb_params,
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # COLLECT BEST MODELS
    # --------------------------------------------------------

    best_models = {
        "Tuned Random Forest": rf_search.best_estimator_,
        "Tuned SVM": svm_search.best_estimator_,
        "Tuned XGBoost": xgb_search.best_estimator_
    }

    # --------------------------------------------------------
    # CREATE TUNED HIERARCHICAL STACKING
    # --------------------------------------------------------

    hierarchical_stacking = build_hierarchical_stacking(
        rf_search.best_estimator_, svm_search.best_estimator_,
        xgb_search.best_estimator_, random_state=RANDOM_STATE,
        n_splits=CV_FOLDS
    )

    base = (rf_search.best_estimator_, svm_search.best_estimator_, xgb_search.best_estimator_)
    best_models["Tuned Soft Voting Baseline"] = build_soft_voting(*base).fit(X_train, y_train)
    best_models["Tuned Single-Level Stacking"] = build_single_stacking(
        *base, random_state=RANDOM_STATE, n_splits=CV_FOLDS
    ).fit(X_train, y_train)

    print("\nTraining Tuned Hierarchical Stacking...")
    hierarchical_stacking.fit(X_train, y_train)
    best_models["Tuned Hierarchical Stacking"] = hierarchical_stacking

    # --------------------------------------------------------
    # EVALUATE ALL MODELS
    # --------------------------------------------------------

    results = []

    for name, model in best_models.items():

        result, cm = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(result)

        save_confusion_matrix(name, cm)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_path = os.path.join(
        RESULTS_DIR,
        "tuned_model_results.csv"
    )

    results_df.to_csv(results_path, index=False)

    print("\nTuned model results saved:", results_path)

    print("\nFINAL OPTIMIZED MODEL COMPARISON")
    print(results_df.to_string(index=False))

    # --------------------------------------------------------
    # SAVE BEST PARAMETERS
    # --------------------------------------------------------

    best_parameters = {
        "Random Forest": rf_search.best_params_,
        "SVM": svm_search.best_params_,
        "XGBoost": xgb_search.best_params_
    }

    params_path = os.path.join(
        RESULTS_DIR,
        "best_hyperparameters.json"
    )

    with open(params_path, "w") as file:
        json.dump(best_parameters, file, indent=4, default=str)

    print("\nBest parameters saved:", params_path)

    # --------------------------------------------------------
    # SAVE TRAINED MODELS
    # --------------------------------------------------------

    models_path = os.path.join(
        RESULTS_DIR,
        "optimized_models.joblib"
    )

    joblib.dump(best_models, models_path)

    print("Optimized models saved:", models_path)

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_index = results_df["F1-Score"].idxmax()
    best_result = results_df.loc[best_index]

    print("\n" + "=" * 60)
    print("BEST OPTIMIZED MODEL")
    print("=" * 60)

    print("Model:", best_result["Model"])
    print(f"Accuracy : {best_result['Accuracy']:.4f}")
    print(f"Precision: {best_result['Precision']:.4f}")
    print(f"Recall   : {best_result['Recall']:.4f}")
    print(f"F1-Score : {best_result['F1-Score']:.4f}")
    print(f"ROC-AUC  : {best_result['ROC-AUC']:.4f}")

    print("\nPHASE 4 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
