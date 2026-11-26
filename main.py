
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROJECT EXECUTION PHASES
# ============================================================

PHASES = [
    ("Phase 1.1 - Data Preprocessing",
     "data_preprocessing.py"),

    ("Phase 1.2 - Preprocessing Pipeline",
     "preprocessing_pipeline.py"),

    ("Phase 2 - Hybrid Feature Selection (ANOVA + RFE)",
     "hybrid_feature_selection.py"),

    ("Phase 3 - Multi-Level Hierarchical Stacking",
     "ensemble_models.py"),

    ("Phase 4 - Hyperparameter Tuning",
     "hyperparameter_tuning.py"),

    ("Phase 5 - Explainable AI (SHAP + LIME)",
     "explainable_ai.py"),

    ("Phase 6 - Model Evaluation",
     "model_evaluation.py"),

    ("Phase 7 - Final Prediction",
     "final_prediction.py")
]


# ============================================================
# RUN INDIVIDUAL PHASE
# ============================================================

def run_phase(phase_name, filename):

    script_path = SRC_DIR / filename

    print("\n" + "=" * 70)
    print(phase_name)
    print("=" * 70)

    if not script_path.exists():
        print(f"ERROR: File not found: {script_path}")
        sys.exit(1)

    start_time = time.perf_counter()

    try:

        subprocess.run(
            [
                sys.executable,
                "-u",
                str(script_path)
            ],
            cwd=ROOT,
            check=True
        )

    except subprocess.CalledProcessError as error:

        print(f"\nFAILED: {phase_name}")
        print(f"Exit Code: {error.returncode}")
        print("Project execution stopped.")

        sys.exit(error.returncode)

    elapsed_time = time.perf_counter() - start_time

    print(f"\nCOMPLETED: {phase_name}")
    print(f"Execution Time: {elapsed_time:.2f} seconds")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    total_start_time = time.perf_counter()

    print("=" * 70)
    print("CHRONIC DISEASE PREDICTION")
    print("Multi-Level Hierarchical Stacking Ensemble")
    print("=" * 70)

    print("\nProposed Model Architecture:")
    print("Feature Selection: ANOVA + RFE")
    print("Level 0: Random Forest + SVM + XGBoost")
    print("Level 1: Logistic Regression")
    print("Level 2: Gradient Boosting")
    print("Explainable AI: SHAP + LIME")

    print(f"\nPython Version: {sys.version.split()[0]}")
    print(f"Project Directory: {ROOT}")

    for phase_name, filename in PHASES:

        run_phase(phase_name, filename)

    total_time = time.perf_counter() - total_start_time

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Models Directory: {MODELS_DIR}")

    print("\nProject Execution Finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
