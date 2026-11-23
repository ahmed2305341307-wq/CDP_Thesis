
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
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


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "results/hybrid_feature_data.joblib"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading hybrid feature selection data...")

    data = joblib.load(DATA_PATH)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    selected_features = data["selected_features"]

    print("Data loaded successfully!")
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
    print("Selected features:", len(selected_features))

    return X_train, X_test, y_train, y_test, selected_features


# ============================================================
# CREATE MACHINE LEARNING MODELS
# ============================================================

def create_models():

    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    xgboost = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    voting_ensemble = VotingClassifier(
        estimators=[
            ("rf", random_forest),
            ("svm", svm),
            ("xgb", xgboost)
        ],
        voting="soft"
    )

    return {
        "Random Forest": random_forest,
        "SVM": svm,
        "XGBoost": xgboost,
        "Voting Ensemble": voting_ensemble
    }


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 55)
    print("Training:", name)
    print("=" * 55)

    model.fit(X_train, y_train)

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

    result = {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }

    return result, cm


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

    save_path = os.path.join(
        RESULTS_DIR,
        f"confusion_matrix_{filename}.png"
    )

    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print("Saved:", save_path)


# ============================================================
# SAVE MODEL COMPARISON RESULTS
# ============================================================

def save_results(results):

    results_df = pd.DataFrame(results)

    save_path = os.path.join(
        RESULTS_DIR,
        "ensemble_results.csv"
    )

    results_df.to_csv(save_path, index=False)

    print("\nModel comparison results saved!")
    print("\n", results_df.to_string(index=False))

    return results_df


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("PHASE 3: ENSEMBLE MACHINE LEARNING")
    print("Chronic Kidney Disease Prediction")
    print("=" * 60)

    # Load dataset
    X_train, X_test, y_train, y_test, selected_features = load_data()

    # Create models
    models = create_models()

    results = {}

    # Train and evaluate
    for name, model in models.items():

        result, cm = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results[name] = result

        save_confusion_matrix(name, cm)

    # Save comparison results
    results_df = save_results(list(results.values()))

    # Save trained models
    model_path = os.path.join(
        RESULTS_DIR,
        "trained_ensemble_models.joblib"
    )

    joblib.dump(models, model_path)

    print("\nTrained models saved:", model_path)

    # Identify best baseline model
    best_model = results_df.loc[
        results_df["F1-Score"].idxmax()
    ]

    print("\n" + "=" * 60)
    print("BEST BASELINE MODEL")
    print("=" * 60)

    print("Model:", best_model["Model"])
    print(f"Accuracy : {best_model['Accuracy']:.4f}")
    print(f"Precision: {best_model['Precision']:.4f}")
    print(f"Recall   : {best_model['Recall']:.4f}")
    print(f"F1-Score : {best_model['F1-Score']:.4f}")
    print(f"ROC-AUC  : {best_model['ROC-AUC']:.4f}")

    print("\nPHASE 3 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
