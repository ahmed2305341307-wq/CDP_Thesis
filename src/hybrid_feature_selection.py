
# ============================================================
# PHASE 2: HYBRID FEATURE SELECTION
# ANOVA FILTER + RFE WRAPPER
#
# Thesis: Explainable Ensemble Machine Learning Framework
# for Chronic Disease Prediction
# ============================================================

import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_DIR / "results"

INPUT_PATH = RESULTS_DIR / "preprocessed_data.joblib"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD PREPROCESSED DATA
# ============================================================

def load_preprocessed_data():

    print("\n========== LOADING PREPROCESSED DATA ==========")

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Preprocessed data not found. "
            "Run preprocessing_pipeline.py first."
        )

    artifacts = joblib.load(INPUT_PATH)

    X_train = artifacts["X_train"]
    X_test = artifacts["X_test"]

    y_train = artifacts["y_train"]
    y_test = artifacts["y_test"]

    feature_names = np.array(
        artifacts["feature_names"]
    )

    print("Training Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    print("Original Features:", len(feature_names))

    return X_train, X_test, y_train, y_test, feature_names


# ============================================================
# 3. ANOVA FILTER FEATURE SELECTION
# ============================================================

def anova_feature_selection(
    X_train,
    y_train,
    feature_names,
    k=25
):

    print("\n========== ANOVA FEATURE SELECTION ==========")

    if k > X_train.shape[1]:

        raise ValueError(
            "ANOVA k cannot exceed the total feature count."
        )

    anova_selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    # Fit only on training data
    X_train_anova = anova_selector.fit_transform(
        X_train,
        y_train
    )

    # Transform training-independent test data
    X_test_anova = anova_selector.transform(
        X_test_global
    )

    selected_mask = anova_selector.get_support()

    anova_features = feature_names[selected_mask]

    scores = anova_selector.scores_

    score_table = pd.DataFrame({

        "Feature": feature_names,

        "ANOVA_F_Score": scores,

        "Selected": selected_mask

    })

    score_table = score_table.sort_values(
        by="ANOVA_F_Score",
        ascending=False
    )

    score_table.to_csv(
        RESULTS_DIR / "anova_feature_scores.csv",
        index=False
    )

    print("Original Features:", len(feature_names))

    print("ANOVA Selected Features:", len(anova_features))

    print("\nTop ANOVA Features:")

    print(score_table.head(15).to_string(index=False))

    return (
        X_train_anova,
        X_test_anova,
        anova_features,
        anova_selector
    )


# ============================================================
# 4. RFE WRAPPER FEATURE SELECTION
# ============================================================

def rfe_feature_selection(
    X_train_anova,
    X_test_anova,
    anova_features,
    y_train,
    n_features=15
):

    print("\n========== RFE FEATURE SELECTION ==========")

    if n_features > X_train_anova.shape[1]:

        raise ValueError(
            "RFE feature count cannot exceed ANOVA features."
        )

    estimator = LogisticRegression(

        max_iter=3000,

        class_weight="balanced",

        random_state=42

    )

    rfe_selector = RFE(

        estimator=estimator,

        n_features_to_select=n_features,

        step=1

    )

    # Fit only on training data
    X_train_rfe = rfe_selector.fit_transform(
        X_train_anova,
        y_train
    )

    X_test_rfe = rfe_selector.transform(
        X_test_anova
    )

    rfe_mask = rfe_selector.get_support()

    rfe_features = anova_features[rfe_mask]

    ranking_table = pd.DataFrame({

        "Feature": anova_features,

        "RFE_Ranking": rfe_selector.ranking_,

        "Selected": rfe_mask

    })

    ranking_table = ranking_table.sort_values(
        by="RFE_Ranking"
    )

    ranking_table.to_csv(
        RESULTS_DIR / "rfe_feature_ranking.csv",
        index=False
    )

    print("ANOVA Input Features:", len(anova_features))

    print("RFE Selected Features:", len(rfe_features))

    print("\nSelected RFE Features:")

    for feature in rfe_features:

        print("-", feature)

    return (
        X_train_rfe,
        X_test_rfe,
        rfe_features,
        rfe_selector
    )


# ============================================================
# 5. HYBRID FEATURE SELECTION
# ============================================================

def hybrid_feature_selection():

    global X_test_global

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    ) = load_preprocessed_data()

    # Keep test data accessible for ANOVA transformation
    X_test_global = X_test

    # Step 1: ANOVA
    (
        X_train_anova,
        X_test_anova,
        anova_features,
        anova_selector
    ) = anova_feature_selection(

        X_train,
        y_train,
        feature_names,

        k=25

    )

    # Step 2: RFE
    (
        X_train_hybrid,
        X_test_hybrid,
        hybrid_features,
        rfe_selector
    ) = rfe_feature_selection(

        X_train_anova,
        X_test_anova,
        anova_features,
        y_train,

        n_features=15

    )

    # ========================================================
    # 6. SAVE HYBRID FEATURE SELECTION RESULTS
    # ========================================================

    print("\n========== HYBRID SELECTION RESULTS ==========")

    print("Original Features:", len(feature_names))

    print("After ANOVA:", len(anova_features))

    print("After RFE:", len(hybrid_features))

    print("\nFinal Hybrid Features:")

    for feature in hybrid_features:

        print("-", feature)

    # Save final selected features
    pd.DataFrame({

        "Selected_Feature": hybrid_features

    }).to_csv(

        RESULTS_DIR / "hybrid_selected_features.csv",

        index=False

    )

    # Save transformed datasets and selectors
    hybrid_artifacts = {

        "X_train": X_train_hybrid,

        "X_test": X_test_hybrid,

        "y_train": y_train,

        "y_test": y_test,

        "selected_features": hybrid_features,

        "anova_features": anova_features,

        "anova_selector": anova_selector,

        "rfe_selector": rfe_selector

    }

    joblib.dump(

        hybrid_artifacts,

        RESULTS_DIR / "hybrid_feature_data.joblib"

    )

    print("\nSaved Files:")

    print("- anova_feature_scores.csv")

    print("- rfe_feature_ranking.csv")

    print("- hybrid_selected_features.csv")

    print("- hybrid_feature_data.joblib")

    print("\n========== HYBRID FEATURE SELECTION COMPLETED ==========")


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    hybrid_feature_selection()
