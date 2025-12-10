📌 Project Overview

Diabetes is a chronic condition that requires early detection.
This project analyzes medical parameters and predicts whether a person is diabetic or non-diabetic using a trained machine learning model.

What this project does:

Collects medical input from users

Runs prediction using Random Forest

Displays diabetes risk instantly

🚀 Features

✅ Random Forest Machine Learning Model
✅ Flask Web Application
✅ User-Friendly HTML/CSS Interface
✅ Real-Time Prediction
✅ Easy Setup & Deployment

🧠 Machine Learning Model

Algorithm: Random Forest Classifier
Learning Type: Supervised Classification

Why Random Forest?

High accuracy

Handles complex medical data well

Reduces overfitting

Robust and scalable

📊 Dataset Details

The model is trained using a diabetes dataset with the following features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Target Variable

0 → Non-Diabetic

1 → Diabetic

🛠️ Technology Stack
Backend

Python

Flask

Scikit-learn

Pandas

NumPy

Frontend

HTML

CSS

📂 Project Structure

Diabetes-Risk-Prediction
│
├── data
│ └── diabetes.csv
│
├── model
│ └── random_forest_model.pkl
│
├── static
│ └── css
│ └── style.css
│
├── templates
│ ├── index.html
│ └── result.html
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md

⚙️ Installation & Execution Steps

1️⃣ Clone the repository
git clone https://github.com/your-username/diabetes-risk-prediction.git

2️⃣ Move to project folder
cd diabetes-risk-prediction

3️⃣ Create virtual environment
python -m venv venv

4️⃣ Activate environment
Windows → venv\Scripts\activate
Linux/Mac → source venv/bin/activate

5️⃣ Install dependencies
pip install -r requirements.txt

6️⃣ Train the ML model
python train_model.py

7️⃣ Run Flask app
python app.py

8️⃣ Open in browser
http://127.0.0.1:5000

🖥️ Application Flow

User inputs health data →
Flask receives data →
Random Forest model predicts risk →
Result displayed on the web page

📈 Model Evaluation

Accuracy: 85–90%

Metrics used:

Accuracy Score

Confusion Matrix

Classification Report

🔮 Future Scope

Add more ML models (XGBoost, SVM)

Database integration (MySQL / PostgreSQL)

REST API using FastAPI

Authentication system

Cloud deployment

Model explainability (SHAP)
