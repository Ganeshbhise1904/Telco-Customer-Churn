"""
ChurnSense - Telco Customer Churn Prediction
Flask Application
"""

import os
import logging
import joblib
import pandas as pd

from flask import Flask, render_template_string, request

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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>ChurnSense AI | Customer Churn Prediction</title>

<style>

:root {
    --bg: #0b1020;
    --panel: #121b2e;
    --panel-soft: #17243a;
    --border: #263752;
    --text: #e8eef8;
    --muted: #9aabc3;
    --blue: #60a5fa;
    --green: #4ade80;
    --red: #fb7185;
    --input: #0c1527;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    color: var(--text);

    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(37,99,235,.18),
            transparent 32%
        ),
        radial-gradient(
            circle at 95% 15%,
            rgba(14,165,233,.12),
            transparent 28%
        ),
        var(--bg);

    font-family: Inter, system-ui, sans-serif;
}

.container {
    width: min(1120px, calc(100% - 32px));
    margin: 0 auto;
    padding: 42px 0 60px;
}

.topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 30px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 800;
}

.brand-mark {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 13px;
    background: linear-gradient(135deg,#2563eb,#06b6d4);
    color: white;
    font-size: 20px;
}

.status {
    color: #a7f3d0;
    border: 1px solid rgba(74,222,128,.25);
    background: rgba(74,222,128,.08);
    padding: 7px 11px;
    border-radius: 999px;
    font-size: 12px;
}

.hero {
    text-align: center;
    margin: 34px auto 30px;
    max-width: 720px;
}

.eyebrow {
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 800;
}

h1 {
    font-size: clamp(32px,5vw,48px);
    margin: 12px 0;
    letter-spacing: -1.5px;
}

.hero p {
    color: var(--muted);
    line-height: 1.7;
}

.panel {
    background: linear-gradient(
        145deg,
        rgba(23,36,58,.96),
        rgba(18,27,46,.96)
    );

    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 18px 55px rgba(0,0,0,.2);
    margin-top: 20px;
}

.section-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 22px;
}

h2 {
    margin: 0;
    font-size: 21px;
}

.sub {
    color: var(--muted);
    font-size: 13px;
    margin-top: 7px;
    line-height: 1.5;
}

.grid {
    display: grid;
    grid-template-columns: repeat(3,minmax(0,1fr));
    gap: 18px;
}

.field label {
    display: block;
    color: #c8d5e8;
    font-size: 12px;
    font-weight: 700;
    margin: 0 0 8px;
}

input, select {
    width: 100%;
    min-height: 46px;
    padding: 11px 12px;
    color: var(--text);
    background: var(--input);
    border: 1px solid #344761;
    border-radius: 10px;
    outline: none;
    font: inherit;
    font-size: 14px;
}

input:focus, select:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 3px rgba(96,165,250,.12);
}

select option {
    background: #111c30;
}

.button-row {
    display: flex;
    justify-content: flex-end;
    margin-top: 24px;
}

button {
    border: 0;
    border-radius: 11px;
    padding: 14px 22px;
    color: white;
    font-weight: 800;
    font-size: 14px;
    cursor: pointer;

    background: linear-gradient(
        100deg,
        #2563eb,
        #0891b2
    );

    box-shadow: 0 10px 25px rgba(37,99,235,.2);
    transition: transform .15s,filter .15s;
}

button:hover {
    transform: translateY(-1px);
    filter: brightness(1.1);
}

.alert {
    border-radius: 12px;
    padding: 15px 17px;
    margin: 18px 0;
    border: 1px solid;
    line-height: 1.55;
}

.alert-error {
    color: #fecdd3;
    background: rgba(244,63,94,.09);
    border-color: rgba(251,113,133,.35);
}

.alert-warning {
    color: #fde68a;
    background: rgba(245,158,11,.08);
    border-color: rgba(245,158,11,.3);
}

.result-panel {
    border-color: rgba(96,165,250,.5);
}

.result-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
}

.result-label {
    font-size: 30px;
    font-weight: 900;
    margin-top: 9px;
}

.churn {
    color: var(--red);
}

.no-churn {
    color: var(--green);
}

.confidence {
    text-align: right;
}

.confidence strong {
    display: block;
    font-size: 30px;
    margin-top: 5px;
}

.stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 24px;
}

.stat {
    background: rgba(7,15,29,.45);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 17px;
}

.stat span {
    color: var(--muted);
    font-size: 12px;
}

.stat strong {
    display: block;
    font-size: 25px;
    margin-top: 8px;
}

.bar {
    height: 8px;
    background: #263752;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 12px;
}

.bar > div {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg,#2563eb,#22d3ee);
}

.explanation {
    color: #cbd8e9;
    line-height: 1.7;
    margin-top: 20px;
}

.notes {
    color: var(--muted);
    line-height: 1.7;
    padding-left: 20px;
}

.footer {
    text-align: center;
    color: #71829c;
    font-size: 12px;
    margin-top: 28px;
}

@media (max-width:800px) {
    .grid {
        grid-template-columns: repeat(2,minmax(0,1fr));
    }
}

@media (max-width:560px) {
    .container {
        width: calc(100% - 22px);
        padding-top: 22px;
    }

    .panel {
        padding: 19px;
        border-radius: 15px;
    }

    .grid {
        grid-template-columns: 1fr;
    }

    .topline {
        margin-bottom: 20px;
    }

    .stats {
        grid-template-columns: 1fr;
    }

    .confidence {
        text-align: left;
    }

    .button-row button {
        width: 100%;
    }
}

</style>
</head>

<body>

<div class="container">

    <div class="topline">
        <div class="brand">
            <div class="brand-mark">CS</div>
            <span>ChurnSense AI</span>
        </div>

        <div class="status">● ML MODEL APP</div>
    </div>

    <header class="hero">
        <div class="eyebrow">
            Customer intelligence platform
        </div>

        <h1>Customer Churn Prediction</h1>

        <p>
            Enter customer details to estimate churn risk
            using your trained machine learning model.
        </p>
    </header>

    {% if startup_error %}
    <div class="alert alert-warning">
        <strong>Model loading issue</strong><br>
        {{ startup_error }}
    </div>
    {% endif %}

    {% if errors %}
    <div class="alert alert-error">
        <strong>Please check the following:</strong>
        <ul>
            {% for error in errors %}
            <li>{{ error }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <form action="/predict" method="POST">

        <section class="panel">

            <div class="section-head">
                <div>
                    <h2>Customer profile</h2>
                    <div class="sub">
                        Complete all fields. Choose the closest
                        matching customer information.
                    </div>
                </div>
            </div>

            <div class="grid">

                {% for field, options in choices.items() %}

                <div class="field">

                    <label for="{{ field }}">
                        {{ field }}
                    </label>

                    <select
                        id="{{ field }}"
                        name="{{ field }}"
                        required
                    >

                        <option value="">
                            Select {{ field }}
                        </option>

                        {% for option in options %}

                        <option
                            value="{{ option }}"
                            {% if form.get(field, '')|string == option|string %}
                            selected
                            {% endif %}
                        >
                            {{ option }}
                        </option>

                        {% endfor %}

                    </select>

                </div>

                {% endfor %}

                <div class="field">
                    <label for="tenure">
                        Tenure (months)
                    </label>

                    <input
                        id="tenure"
                        type="number"
                        name="tenure"
                        min="0"
                        max="120"
                        step="1"
                        placeholder="e.g. 12"
                        value="{{ form.get('tenure', '') }}"
                        required
                    >
                </div>

                <div class="field">
                    <label for="MonthlyCharges">
                        Monthly charges
                    </label>

                    <input
                        id="MonthlyCharges"
                        type="number"
                        name="MonthlyCharges"
                        min="0"
                        step="0.01"
                        placeholder="e.g. 75.50"
                        value="{{ form.get('MonthlyCharges', '') }}"
                        required
                    >
                </div>

                <div class="field">
                    <label for="TotalCharges">
                        Total charges
                    </label>

                    <input
                        id="TotalCharges"
                        type="number"
                        name="TotalCharges"
                        min="0"
                        step="0.01"
                        placeholder="e.g. 900.00"
                        value="{{ form.get('TotalCharges', '') }}"
                        required
                    >
                </div>

            </div>

            <div class="button-row">
                <button type="submit">
                    Predict Customer Churn →
                </button>
            </div>

        </section>

    </form>

    {% if result %}

    <section class="panel result-panel" id="result">

        <div class="section-head">

            <div>
                <div class="eyebrow">
                    Prediction complete
                </div>

                <h2 style="margin-top:8px">
                    Model result
                </h2>
            </div>

            <div class="status">
                ✓ Prediction generated
            </div>

        </div>

        <div class="result-top">

            <div>
                <div class="sub">
                    Predicted customer status
                </div>

                <div class="result-label {{ 'churn' if result.churned else 'no-churn' }}">
                    {{ result.label }}
                </div>
            </div>

            <div class="confidence">
                <span class="sub">
                    Model confidence
                </span>

                <strong>
                    {{ result.confidence }}%
                </strong>
            </div>

        </div>

        <div class="stats">

            <div class="stat">
                <span>Churn probability</span>

                <strong class="churn">
                    {{ result.churn_prob }}%
                </strong>

                <div class="bar">
                    <div style="width:{{ result.churn_prob }}%"></div>
                </div>
            </div>

            <div class="stat">
                <span>No churn probability</span>

                <strong class="no-churn">
                    {{ result.no_churn_prob }}%
                </strong>

                <div class="bar">
                    <div style="width:{{ result.no_churn_prob }}%"></div>
                </div>
            </div>

        </div>

        <div class="explanation">
            <strong>Interpretation</strong><br>
            {{ result.explanation }}
        </div>

        {% if result.notes %}

        <div class="explanation">
            <strong>Customer insights</strong>

            <ul class="notes">
                {% for note in result.notes %}
                <li>{{ note }}</li>
                {% endfor %}
            </ul>
        </div>

        {% endif %}

    </section>

    {% endif %}

    <div class="footer">
        ChurnSense AI · Machine Learning Customer Analytics
    </div>

</div>

</body>
</html>
"""
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
# CUSTOMER INSIGHTS
# =========================================================

def generate_explanation(values, result):

    notes = []

    if values["Contract"] == "Month-to-month":

        notes.append(
            "Month-to-month contract may increase churn risk."
        )

    elif values["Contract"] == "Two year":

        notes.append(
            "Two-year contract indicates a longer customer commitment."
        )

    if values["tenure"] <= 6:

        notes.append(
            "Customer has a short tenure. Early customer engagement "
            "may help improve retention."
        )

    elif values["tenure"] >= 36:

        notes.append(
            "Customer has a relatively long relationship with the company."
        )

    if values["InternetService"] == "Fiber optic":

        notes.append(
            "Customer uses Fiber optic internet service."
        )

    if values["PaymentMethod"] == "Electronic check":

        notes.append(
            "Customer uses Electronic check as the payment method."
        )

    if values["TechSupport"] == "No":

        notes.append(
            "Customer does not have Tech Support service."
        )

    if result["churned"]:

        explanation = (
            "The trained machine learning model predicts that this "
            "customer may leave the company. Consider reviewing the "
            "customer experience and offering suitable retention support."
        )

    else:

        explanation = (
            "The trained machine learning model predicts that this "
            "customer is likely to continue the service. Continue "
            "monitoring customer satisfaction and engagement."
        )

    return explanation, notes


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return render_template_string(
        HTML_TEMPLATE,
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

    form_data = request.form.to_dict()

    values, errors = validate_form(form_data)

    if errors:

        return render_template_string(
            HTML_TEMPLATE,
            choices=CHOICES,
            form=form_data,
            result=None,
            errors=errors,
            startup_error=STARTUP_ERROR
        ), 400

    try:

        prediction_result = run_prediction(values)

        explanation, notes = generate_explanation(
            values,
            prediction_result
        )

        prediction_result["explanation"] = explanation
        prediction_result["notes"] = notes

        return render_template_string(
            HTML_TEMPLATE,
            choices=CHOICES,
            form=form_data,
            result=prediction_result,
            errors=[],
            startup_error=STARTUP_ERROR
        )

    except Exception as exc:

        logger.exception("Prediction failed.")

        return render_template_string(
            HTML_TEMPLATE,
            choices=CHOICES,
            form=form_data,
            result=None,
            errors=[
                "Prediction failed: " + str(exc)
            ],
            startup_error=STARTUP_ERROR
        ), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    if MODEL is None:

        return {
            "status": "error",
            "model_loaded": False,
            "error": STARTUP_ERROR
        }, 500

    return {
        "status": "ok",
        "model_loaded": True,
        "message": "ChurnSense AI is running."
    }, 200


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )