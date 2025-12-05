📌 Abstract

This project presents an end-to-end Predictive Analytics System designed to identify students at high risk of dropping out using advanced Machine Learning and Explainable AI (XAI) techniques.

The system combines:

A Random Forest predictive engine

A modern Glassmorphism UI

A real-time inference API layer

A dynamic Smart-Form generator

An automated PDF reporting system

By integrating predictive modeling with intuitive visualization and interpretability features, the system supports teachers, counselors, and administrators in proactively addressing dropout risks.

🚀 Key Features
1. Advanced Machine Learning

Dual-model architecture: Random Forest (primary) + Logistic Regression (baseline)

GridSearchCV-optimized hyperparameters

Achieves 92% accuracy and 0.96 ROC-AUC

Handles non-linear socio-economic patterns effectively

2. Intelligent Dynamic UI

Real-time Smart Form auto-generated using /info endpoint

Guaranteed zero UI-backend mismatch

Premium Glassmorphism theme

Interactive probability visualizations

Smooth animations & responsive layout

Fully validated inputs (e.g., grade ranges, age limits)

3. Explainable AI (XAI)

Displays Top 3 Risk Drivers based on model feature contributions

Human-friendly interpretation of model outputs

Color-coded risk meter (Low / Medium / High)

Helps educators understand why a student is at risk — not just that they are.

4. Administrative PDF Reports

One-click PDF export of the full risk analysis

Includes probability, drivers, recommendations, and visual charts

Uses html2canvas + jsPDF

100% client-side (no server load)

🏗️ System Architecture
🔹 Backend — FastAPI

Located in app/ directory
Features:

Serves the SPA

Returns input schema for dynamic UI (GET /info)

Performs ML inference via (POST /predict)

Loads model artifacts using FastAPI lifespan events

Scales numeric features and encodes categorical variables

🔹 Frontend — HTML, CSS, JS

Located in app/templates & app/static

Highlights:

Custom CSS variables for theming

Progress bars, circular probability graphs

Dynamic modal for analysis results

PDF generator

No external frameworks → Fast, lightweight, maintainable

🔹 Machine Learning Engine

Located in scripts/ and models/

Pipeline tasks:

Data cleaning & preprocessing

Encoding categorical data

Feature scaling

SMOTE balancing (contextual)

Hyperparameter tuning

Save model artifacts (model.pkl, scaler.pkl, feature_names.pkl)

📂 Project Directory Structure
Predictive-Student-Dropout-Modeling/
├── app/
│   ├── static/
│   │   ├── style.css         # Glassmorphism UI
│   │   └── script.js         # Smart Form + PDF Engine
│   ├── templates/
│   │   └── index.html        # Main web interface
│   ├── app.py                # FastAPI backend
│   └── requirements.txt      # Dependencies
│
├── data/
│   └── student_dropout_1000.csv
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── scripts/
│   ├── train_model.py        # Full ML training pipeline
│   └── create_notebooks.py   # Auto-generates Jupyter notebooks
│
├── tests/
│   ├── test_api.py           # API-level tests
│   └── test_model.py         # Model sanity checks
│
└── README.md

💻 How to Run the Project
✔ Step 1 — Install Dependencies
pip install -r app/requirements.txt

✔ Step 2 — (Optional) Retrain the ML Model
python scripts/train_model.py

✔ Step 3 — Launch Backend Server
python app/app.py

✔ Step 4 — Open Application

Visit:

🔗 http://localhost:8000

📊 Model Evaluation
Metric	Random Forest	Logistic Regression
Accuracy	92.4%	88.1%
Precision	0.91	0.86
Recall	0.93	0.89
ROC-AUC	0.96	0.93

Conclusion:
Random Forest significantly outperforms Logistic Regression in modeling complex socio-academic relationships, making it the ideal deployment choice.

🧪 Testing

Run all tests:

pytest tests/


Tests include:

API inference validation

Schema integrity tests

Model artifact loading

End-to-end prediction structure

🎓 Potential Extensions

This system can be expanded into a full MSc thesis or industry research project:

Integration with student MIS portals

Multicampus or multi-university datasets

SHAP value heatmaps

Time-series modeling for semester-wise risk

Deployment on AWS/GCP

📄 License

This project is licensed under the MIT License.

✨ Acknowledgements

Developed as part of an advanced research initiative under EduGuard AI (2025) for academic risk assessment.

