from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Check file existence
model_path = r"C:\Users\ankit\Desktop\Diabetes Risk Prediction\model\diabetes_model.pkl"
data_path = r"C:\Users\ankit\Desktop\csv files\diabetes.csv"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file not found at {data_path}")

# Load model and dataset
model = joblib.load(model_path)
df = pd.read_csv(data_path)
if 'Outcome' not in df.columns:
    raise ValueError("Dataset missing 'Outcome' column")
df = df.drop(columns='Outcome')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate form fields
        required_fields = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        for field in required_fields:
            if field not in request.form:
                raise BadRequest(f"Missing form field: {field}")

        # Collect and validate inputs
        input_data = []
        for field in required_fields:
            value = request.form[field]
            try:
                float_value = float(value)
                if float_value < 0:
                    raise ValueError(f"{field} cannot be negative")
                input_data.append(float_value)
            except ValueError:
                raise BadRequest(f"Invalid value for {field}: {value}")

        # Prepare data for model
        sample = pd.DataFrame([input_data], columns=df.columns)

        # Make prediction
        prediction = model.predict(sample)[0]
        result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"

        # Render result.html
        return render_template("result.html", prediction=result)

    except BadRequest as e:
        return render_template("error.html", error=str(e)), 400
    except ValueError as e:
        return render_template("error.html", error=str(e)), 400
    except Exception as e:
        return render_template("error.html", error="An unexpected error occurred"), 500

if __name__ == "__main__":
    app.run(debug=True)
    