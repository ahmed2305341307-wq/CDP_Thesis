
# ============================================================
# THESIS PROJECT
# An Explainable Ensemble Machine Learning Framework
# with Hybrid Feature Selection for Chronic Disease Prediction
#
# PHASE 1: DATA ACQUISITION AND PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "kidney_disease.csv"

RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DEFINE FEATURE COLUMNS
# ============================================================

NUMERICAL_FEATURES = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc"
]

CATEGORICAL_FEATURES = [
    "rbc", "pc", "pcc", "ba", "htn",
    "dm", "cad", "appet", "pe", "ane"
]

TARGET_COLUMN = "classification"


# ============================================================
# 3. LOAD AND CLEAN DATASET
# ============================================================

def load_and_clean_data():

    print("\n========== DATA ACQUISITION ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Original Dataset Shape:", df.shape)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove unnecessary ID column
    df.drop(columns=["id"], errors="ignore", inplace=True)

    # Check required columns
    required_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET_COLUMN]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # CLEAN NUMERICAL FEATURES
    # --------------------------------------------------------

    for column in NUMERICAL_FEATURES:

        df[column] = pd.to_numeric(
            df[column]
            .astype("string")
            .str.strip()
            .replace({
                "?": pd.NA,
                "": pd.NA
            }),
            errors="coerce"
        )

    # --------------------------------------------------------
    # CLEAN CATEGORICAL FEATURES
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        cleaned_values = []

        for value in df[column]:

            if pd.isna(value):

                cleaned_values.append(np.nan)

            else:

                value = str(value).strip().lower()

                if value in ["", "?", "nan", "none"]:
                    cleaned_values.append(np.nan)

                else:
                    cleaned_values.append(value)

        # Force standard Python object dtype
        df[column] = pd.Series(
            cleaned_values,
            index=df.index,
            dtype=object
        )

    # --------------------------------------------------------
    # CLEAN TARGET COLUMN
    # --------------------------------------------------------

    target_values = []

    for value in df[TARGET_COLUMN]:

        if pd.isna(value):

            target_values.append(np.nan)

        else:

            target_values.append(
                str(value).strip().lower()
            )

    df[TARGET_COLUMN] = target_values

    # Convert target into binary classes
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({

        "ckd": 1,
        "notckd": 0

    })

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Invalid or missing target values detected."
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    print("\nDataset cleaned successfully!")

    print("Cleaned Dataset Shape:", df.shape)

    print("\nMissing Values Before Imputation:")

    print("Total Missing Values:", df.isna().sum().sum())

    return df


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

def separate_features_target(df):

    print("\n========== FEATURE AND TARGET SEPARATION ==========")

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    print("Number of Features:", X.shape[1])

    print("\nTarget Distribution:")

    print(
        y.value_counts().rename({
            1: "CKD",
            0: "Not CKD"
        })
    )

    return X, y


# ============================================================
# 5. TRAIN-TEST SPLITTING
# ============================================================

def split_dataset(X, y):

    print("\n========== TRAIN TEST SPLITTING ==========")

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("Training Samples:", len(X_train))

    print("Testing Samples:", len(X_test))

    print("\nTraining Target Distribution:")

    print(y_train.value_counts())

    print("\nTesting Target Distribution:")

    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


# ============================================================
# 6. BUILD NUMERICAL AND CATEGORICAL PIPELINES
# ============================================================

def build_preprocessor():

    print("\n========== BUILDING PREPROCESSING PIPELINE ==========")

    # Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )

        ),

        (
            "scaler",

            StandardScaler()

        )

    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True
            )

        ),

        (
            "encoder",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

        )

    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),

            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )

        ],

        remainder="drop",

        verbose_feature_names_out=False

    )

    return preprocessor


# ============================================================
# 7. APPLY PREPROCESSING
# ============================================================

def preprocess_dataset():

    # Load dataset
    df = load_and_clean_data()

    # Separate X and y
    X, y = separate_features_target(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor()

    print("\n========== APPLYING IMPUTATION, ENCODING AND SCALING ==========")

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert to NumPy arrays
    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float64
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n========== PREPROCESSING VALIDATION ==========")

    print("Training Shape:", X_train_processed.shape)

    print("Testing Shape:", X_test_processed.shape)

    print("Encoded Features:", len(feature_names))

    print(
        "Training Missing Values:",
        np.isnan(X_train_processed).sum()
    )

    print(
        "Testing Missing Values:",
        np.isnan(X_test_processed).sum()
    )

    if np.isnan(X_train_processed).any():

        raise ValueError(
            "Missing values remain in training data!"
        )

    if np.isnan(X_test_processed).any():

        raise ValueError(
            "Missing values remain in testing data!"
        )

    print("\nAll preprocessing validations passed!")

    # --------------------------------------------------------
    # SAVE PREPROCESSING ARTIFACTS
    # --------------------------------------------------------

    artifacts = {

        "preprocessor": preprocessor,

        "X_train": X_train_processed,

        "X_test": X_test_processed,

        "y_train": y_train.to_numpy(),

        "y_test": y_test.to_numpy(),

        "feature_names": np.array(feature_names),

        "numerical_features": NUMERICAL_FEATURES,

        "categorical_features": CATEGORICAL_FEATURES

    }

    # Save complete preprocessing data
    joblib.dump(

        artifacts,

        RESULTS_DIR / "preprocessed_data.joblib"

    )

    # Save feature names
    pd.DataFrame({

        "Feature_Name": feature_names

    }).to_csv(

        RESULTS_DIR / "encoded_feature_names.csv",

        index=False

    )

    # Save cleaned dataset for inspection
    df.to_csv(

        RESULTS_DIR / "cleaned_dataset.csv",

        index=False

    )

    print("\n========== FILE SAVING ==========")

    print("Saved: preprocessed_data.joblib")

    print("Saved: encoded_feature_names.csv")

    print("Saved: cleaned_dataset.csv")

    print("\n========== PREPROCESSING COMPLETED SUCCESSFULLY ==========")

    return artifacts


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    artifacts = preprocess_dataset()
