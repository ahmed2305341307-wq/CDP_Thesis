
import os
import joblib
import pandas as pd
import numpy as np

from datetime import datetime


# =====================================================
# PHASE 7: FINAL MODEL SELECTION & PREDICTION
# Chronic Kidney Disease Prediction
# =====================================================

print("\n" + "=" * 60)
print("PHASE 7: FINAL MODEL SELECTION & PREDICTION")
print("=" * 60)


# -----------------------------------------------------
# STEP 1: CREATE OUTPUT DIRECTORIES
# -----------------------------------------------------

os.makedirs("models", exist_ok=True)

output_dir = "results/phase7_prediction"
os.makedirs(output_dir, exist_ok=True)


# -----------------------------------------------------
# STEP 2: LOAD OPTIMIZED MODELS AND FEATURE DATA
# -----------------------------------------------------

print("\nLoading optimized models and feature data...")

models = joblib.load(
    "results/optimized_models.joblib"
)

feature_data = joblib.load(
    "results/hybrid_feature_data.joblib"
)

print("Models loaded successfully!")
print("Feature data loaded successfully!")


# -----------------------------------------------------
# STEP 3: LOAD SELECTED FEATURES
# -----------------------------------------------------

selected_features = list(
    feature_data["selected_features"]
)

print("\nSelected Features:", len(selected_features))

print("\nFeature Names:")

for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")


# -----------------------------------------------------
# STEP 4: FINAL MODEL SELECTION
# -----------------------------------------------------

print("\n" + "=" * 50)
print("FINAL MODEL SELECTION")
print("=" * 50)

# Select the proposed hierarchical stacking model; this is not a claim of superiority.

best_model_name = "Tuned Hierarchical Stacking"

if best_model_name not in models:
    raise KeyError(
        f"{best_model_name} not found in optimized_models.joblib"
    )

final_model = models[best_model_name]

print("\nSelected Model:", best_model_name)


# -----------------------------------------------------
# STEP 5: SAVE FINAL MODEL
# -----------------------------------------------------

final_model_path = "models/final_model.joblib"

final_model_package = {
    "model": final_model,
    "model_name": best_model_name,
    "selected_features": selected_features,
    "class_mapping": {
        0: "Not CKD",
        1: "CKD"
    }
}

joblib.dump(
    final_model_package,
    final_model_path
)

print("\nFinal model saved successfully!")
print("Location:", final_model_path)


# -----------------------------------------------------
# STEP 6: DEFINE PATIENT PREDICTION FUNCTION
# -----------------------------------------------------

def predict_patient(patient_data):

    # Load saved final model
    saved_package = joblib.load(final_model_path)

    model = saved_package["model"]

    feature_names = saved_package["selected_features"]

    # Check missing features
    missing_features = [
        feature for feature in feature_names
        if feature not in patient_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Arrange features in exactly the training order
    input_data = pd.DataFrame(
        [[patient_data[feature] for feature in feature_names]],
        columns=feature_names
    )

    # Ensure numerical input
    input_data = input_data.astype(float)

    # Prediction
    prediction = int(model.predict(input_data)[0])

    # Prediction probability
    probabilities = model.predict_proba(input_data)[0]

    class_list = list(model.classes_)

    ckd_index = class_list.index(1)
    not_ckd_index = class_list.index(0)

    ckd_probability = float(
        probabilities[ckd_index]
    ) * 100

    not_ckd_probability = float(
        probabilities[not_ckd_index]
    ) * 100

    # Class label
    result = (
        "CKD" if prediction == 1
        else "Not CKD"
    )

    # Display results
    print("\n" + "=" * 50)
    print("PATIENT PREDICTION RESULT")
    print("=" * 50)

    print("Prediction:", result)

    print(
        f"CKD Probability: {ckd_probability:.2f}%"
    )

    print(
        f"Not CKD Probability: {not_ckd_probability:.2f}%"
    )

    # Save prediction result
    prediction_result = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "model": best_model_name,
        "prediction": result,
        "prediction_class": prediction,
        "ckd_probability": round(
            ckd_probability, 2
        ),
        "not_ckd_probability": round(
            not_ckd_probability, 2
        )
    }

    result_df = pd.DataFrame(
        [prediction_result]
    )

    result_file = os.path.join(
        output_dir,
        "prediction_results.csv"
    )

    result_df.to_csv(
        result_file,
        mode="a",
        header=not os.path.exists(result_file),
        index=False
    )

    print("\nPrediction result saved successfully!")

    print("Saved at:", result_file)

    return prediction_result


# -----------------------------------------------------
# STEP 7: GET NEW PATIENT DATA
# -----------------------------------------------------

print("\n" + "=" * 50)
print("NEW PATIENT DATA INPUT")
print("=" * 50)

print(
    "\nEnter the 15 selected, preprocessed feature values."
)

print(
    "Values must use the same encoding and scaling "
    "as the training data."
)

patient_data = {}

for feature in selected_features:

    while True:

        try:

            value = float(
                input(f"Enter {feature}: ")
            )

            if not np.isfinite(value):
                print("Please enter a valid finite number.")
                continue

            patient_data[feature] = value

            break

        except ValueError:

            print("Invalid input! Enter a numerical value.")


# -----------------------------------------------------
# STEP 8: PREDICT NEW PATIENT
# -----------------------------------------------------

print("\nProcessing patient data...")

prediction_result = predict_patient(
    patient_data
)


# -----------------------------------------------------
# STEP 9: PHASE 7 COMPLETION
# -----------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 7 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated Files:")
print("1. models/final_model.joblib")
print("2. results/phase7_prediction/prediction_results.csv")

print("\nFinal Model:", best_model_name)

print("\nAll Phase 7 operations completed!")
