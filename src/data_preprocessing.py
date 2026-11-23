# ==========================================
# DATA ACQUISITION AND INITIAL DATA ANALYSIS
# Thesis: Chronic Disease Prediction
# ==========================================

import pandas as pd
from pathlib import Path


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "kidney_disease.csv"

RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# 2. Load dataset
def load_dataset():

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET LOADED ==========")

    print("Dataset Shape:", df.shape)

    return df


# 3. Display first five records
def display_dataset(df):

    print("\n========== FIRST FIVE RECORDS ==========")

    print(df.head())


# 4. Dataset information
def dataset_information(df):

    print("\n========== DATASET INFORMATION ==========")

    df.info()


# 5. Check missing values
def missing_value_analysis(df):

    print("\n========== MISSING VALUE ANALYSIS ==========")

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    print(missing)

    # Save missing value report
    missing.to_csv(
        RESULTS_DIR / "missing_values_report.csv",
        header=["Missing Values"]
    )


# 6. Check duplicate records
def duplicate_analysis(df):

    print("\n========== DUPLICATE ANALYSIS ==========")

    duplicates = df.duplicated().sum()

    print("Duplicate Records:", duplicates)


# 7. Analyze target column
def target_analysis(df):

    print("\n========== TARGET ANALYSIS ==========")

    target = (
        df["classification"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    print("Target Distribution:")

    print(target.value_counts(dropna=False))

    print("\nTarget Classes:")

    print(target.unique())


# 8. Main execution
if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    # Display records
    display_dataset(df)

    # Dataset information
    dataset_information(df)

    # Missing values
    missing_value_analysis(df)

    # Duplicate records
    duplicate_analysis(df)

    # Target analysis
    target_analysis(df)

    print("\n========== ANALYSIS COMPLETED ==========")