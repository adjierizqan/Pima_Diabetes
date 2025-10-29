"""Flask application for serving diabetes predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

SCALER_PATH = Path("scaler.pkl")
MODEL_PATH = Path("best_model.pkl")


def load_artifacts():
    if not SCALER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Please run `python model_training.py` before starting the app."
        )
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    return scaler, model


scaler, model = load_artifacts()

app = Flask(__name__)
app.secret_key = "pima-diabetes-secret-key"


@app.route("/")
def index():
    default_values = {name: "" for name in FEATURE_NAMES}
    return render_template("index.html", feature_names=FEATURE_NAMES, values=default_values)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = {name: float(request.form.get(name, 0)) for name in FEATURE_NAMES}
    except ValueError:
        flash("Please enter valid numeric values for all fields.")
        return redirect(url_for("index"))

    input_df = pd.DataFrame([features])
    transformed = scaler.transform(input_df)
    probabilities = model.predict_proba(transformed)[0]
    probability = float(probabilities[1])
    prediction = int(probability >= 0.5)

    label = "Diabetic" if prediction == 1 else "Non-Diabetic"
    indicator_class = "danger" if prediction == 1 else "success"

    return render_template(
        "result.html",
        prediction=label,
        probability=f"{probability:.2f}",
        indicator_class=indicator_class,
        values=features,
        feature_names=FEATURE_NAMES,
    )


@app.errorhandler(Exception)
def handle_exception(error):  # pragma: no cover - defensive programming for runtime issues
    flash(str(error))
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=False)
