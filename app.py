
"""
ChurnSense - Telco Customer Churn Prediction
Flask Application

Loads an already trained ML model and preprocessor.
Does not retrain the model.
"""

import os
import logging
import joblib
import pandas as pd

from flask import Flask, render_template, request


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    "churn_pipeline.pkl",
    "logestic.pkl",
    "logistic.pkl",
    "Logistic.pkl",
    "model.pkl"
]


# =========================================================
# FEATURE COLUMNS
# =========================================================

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


NUMERIC_FIELDS = {
    "tenure": int,
    "MonthlyCharges": float,
    "TotalCharges": float
}


YES_NO = ["Yes", "No"]

INTERNET_ADDON = [
    "Yes",
    "No",
    "No internet service"
]


# =========================================================
# DROPDOWN OPTIONS
# =========================================================

CHOICES = {

    "gender": ["Female", "Male"],

    "SeniorCitizen": ["0", "1"],

    "Partner": YES_NO,

    "Dependents": YES_NO,

    "PhoneService": YES_NO,

    "MultipleLines": [
        "No",
        "Yes",
        "No phone service"
    ],

    "InternetService": [
        "DSL",
        "Fiber optic",
        "No"
    ],

    "OnlineSecurity": INTERNET_ADDON,

    "OnlineBackup": INTERNET_ADDON,

    "DeviceProtection": INTERNET_ADDON,

    "TechSupport": INTERNET_ADDON,

    "StreamingTV": INTERNET_ADDON,

    "StreamingMovies": INTERNET_ADDON,

    "Contract": [
        "Month-to-month",
        "One year",
        "Two year"
    ],

    "PaperlessBilling": YES_NO,

    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

def load_artifacts():

    model_path = None

    for filename in MODEL_CANDIDATES:

        path = os.path.join(BASE_DIR, filename)

        if os.path.isfile(path):
            model_path = path
            break

    if model_path is None:

        raise FileNotFoundError(
            "Model file not found. Available candidates: "
            + ", ".join(MODEL_CANDIDATES)
        )

    logger.info("Loading model: %s", model_path)

    saved = joblib.load(model_path)

    preprocessor = None
    model = None

    # Case 1: Dictionary containing preprocessor and model
    if isinstance(saved, dict):

        lowered = {
            str(key).lower(): value
            for key, value in saved.items()
        }

        preprocessor_keys = [
            "preprocessor",
            "preprocess",
            "pre_processor",
            "transformer",
            "encoder",
            "column_transformer"
        ]

        model_keys = [
            "model",
            "classifier",
            "clf",
            "logistic_regression",
            "logreg",
            "estimator"
        ]

        for key in preprocessor_keys:

            if key in lowered:
                preprocessor = lowered[key]
                break

        for key in model_keys:

            if key in lowered:
                model = lowered[key]
                break

        # A dictionary may contain a complete pipeline
        if model is None and "pipeline" in lowered:
            model = lowered["pipeline"]

        if model is None:

            raise ValueError(
                "Model not found inside pickle dictionary. "
                f"Available keys: {list(saved.keys())}"
            )

    # Case 2: A complete sklearn pipeline or estimator
    else:

        model = saved

    if not hasattr(model, "predict"):

        raise TypeError(
            "Loaded object does not have predict() method."
        )

    if preprocessor is not None and not hasattr(
        preprocessor, "transform"
    ):

        raise TypeError(
            "Preprocessor does not have transform() method."
        )

    logger.info("Model loaded successfully.")

    return preprocessor, model


# =========================================================
# INITIALIZE MODEL
# =========================================================

try:

    PREPROCESSOR, MODEL = load_artifacts()

    STARTUP_ERROR = None

except Exception as exc:

    PREPROCESSOR = None
    MODEL = None

    STARTUP_ERROR = str(exc)

    logger.exception("Model loading failed.")


# =========================================================
# FORM VALIDATION
# =========================================================

def validate_form(form):

    values = {}
    errors = []

    for column in FEATURE_COLUMNS:

        raw = (form.get(column) or "").strip()

        if not raw:

            errors.append(
                f"{column} is required."
            )

            continue

        # Numeric validation
        if column in NUMERIC_FIELDS:

            try:

                number = NUMERIC_FIELDS[column](raw)

                if number < 0:

                    errors.append(
                        f"{column} cannot be negative."
                    )

                    continue

                if column == "tenure" and number > 120:

                    errors.append(
                        "Tenure must be between 0 and 120 months."
                    )

                    continue

                values[column] = number

            except (ValueError, OverflowError):

                errors.append(
                    f"{column} must contain a valid number."
                )

        elif column == "SeniorCitizen":

            if raw not in CHOICES[column]:

                errors.append(
                    "SeniorCitizen must be 0 or 1."
                )

                continue

            values[column] = int(raw)

        else:

            if raw not in CHOICES[column]:

                errors.append(
                    f"Invalid value for {column}."
                )

                continue

            values[column] = raw

    # Logical validation
    if not errors:

        if (
            values["PhoneService"] == "No"
            and values["MultipleLines"] != "No phone service"
        ):

            errors.append(
                "MultipleLines must be No phone service "
                "when PhoneService is No."
            )

        if values["InternetService"] == "No":

            addon_columns = [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies"
            ]

            for column in addon_columns:

                if values[column] != "No internet service":

                    errors.append(
                        f"{column} must be No internet service."
                    )

        if (
            values["TotalCharges"] < values["MonthlyCharges"]
            and values["tenure"] > 1
        ):

            errors.append(
                "TotalCharges looks too low for the given tenure "
                "and MonthlyCharges."
            )

    return values, errors


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def is_churn_label(label):

    return str(label).strip().lower() in [
        "1",
        "yes",
        "true",
        "churn"
    ]


def run_prediction(values):

    if MODEL is None:

        raise RuntimeError(
            "Model is not loaded. Check your pickle file."
        )

    # Preserve original feature order
    df = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )

    # Transform data using saved preprocessor
    if PREPROCESSOR is not None:

        X = PREPROCESSOR.transform(df)

    else:

        # Complete pipeline handles raw input itself
        X = df

    # Predict class
    prediction = MODEL.predict(X)[0]

    churned = is_churn_label(prediction)

    # Probability calculation
    if not hasattr(MODEL, "predict_proba"):

        raise RuntimeError(
            "This model does not support predict_proba(). "
            "Use a model trained with probability support."
        )

    probabilities = MODEL.predict_proba(X)[0]

    classes = list(MODEL.classes_)

    churn_indices = [
        i for i, label in enumerate(classes)
        if is_churn_label(label)
    ]

    if not churn_indices:

        raise ValueError(
            f"Churn class not found in model classes: {classes}"
        )

    churn_index = churn_indices[0]

    churn_probability = float(
        probabilities[churn_index]
    ) * 100

    no_churn_probability = 100 - churn_probability

    confidence = (
        churn_probability if churned
        else no_churn_probability
    )

    return {

        "churned": churned,

        "label": (
            "Churn" if churned
            else "No Churn"
        ),

        "confidence": round(confidence, 2),

        "churn_prob": round(
            churn_probability, 2
        ),

        "no_churn_prob": round(
            no_churn_probability, 2
        )
    }


# =========================================================
# PREDICTION EXPLANATION
# =========================================================

def explain(values, result):

    risk_factors = []
    protective_factors = []

    if values["Contract"] == "Month-to-month":

        risk_factors.append(
            "Month-to-month contract"
        )

    else:

        protective_factors.append(
            f"{values['Contract']} contract"
        )

    if values["tenure"] < 12:

        risk_factors.append(
            "Short customer tenure"
        )

    elif values["tenure"] >= 36:

        protective_factors.append(
            "Long customer tenure"
        )

    if values["InternetService"] == "Fiber optic":

        risk_factors.append(
            "Fiber optic internet"
        )

    if values["PaymentMethod"] == "Electronic check":

        risk_factors.append(
            "Electronic check payment"
        )

    if (
        values["InternetService"] != "No"
        and values["TechSupport"] == "No"
    ):

        risk_factors.append(
            "No technical support"
        )

    if (
        values["InternetService"] != "No"
        and values["OnlineSecurity"] == "No"
    ):

        risk_factors.append(
            "No online security"
        )

    if result["churned"]:

        explanation = (
            f"The model estimates a "
            f"{result['churn_prob']}% probability "
            "that this customer may leave. "
            "Consider proactive retention strategies."
        )

    else:

        explanation = (
            f"The model estimates a "
            f"{result['no_churn_prob']}% probability "
            "that this customer may stay. "
            "Continue monitoring customer satisfaction."
        )

    notes = []

    if risk_factors:

        notes.append(
            "Potential risk factors: "
            + ", ".join(risk_factors)
            + "."
        )

    if protective_factors:

        notes.append(
            "Potential retention factors: "
            + ", ".join(protective_factors)
            + "."
        )

    return explanation, notes


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html",
        choices=CHOICES,
        form={},
        result=None,
        errors=[],
        startup_error=STARTUP_ERROR
    )


# =========================================================
# PREDICTION ROUTE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if STARTUP_ERROR:

        return render_template(
            "index.html",
            choices=CHOICES,
            form=request.form,
            result=None,
            errors=[],
            startup_error=STARTUP_ERROR
        ), 500

    values, errors = validate_form(request.form)

    if errors:

        return render_template(
            "index.html",
            choices=CHOICES,
            form=request.form,
            result=None,
            errors=errors,
            startup_error=None
        ), 400

    try:

        result = run_prediction(values)

        explanation, notes = explain(
            values,
            result
        )

        result["explanation"] = explanation

        result["notes"] = notes

        logger.info(
            "Prediction completed: %s",
            result["label"]
        )

    except Exception as exc:

        logger.exception(
            "Prediction error"
        )

        return render_template(
            "index.html",
            choices=CHOICES,
            form=request.form,
            result=None,
            errors=[
                f"Prediction failed: {str(exc)}"
            ],
            startup_error=None
        ), 500

    return render_template(
        "index.html",
        choices=CHOICES,
        form=request.form,
        result=result,
        errors=[],
        startup_error=None
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    if STARTUP_ERROR:

        return {
            "status": "error",
            "model_loaded": False,
            "message": STARTUP_ERROR
        }, 500

    return {
        "status": "healthy",
        "model_loaded": True
    }, 200


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
