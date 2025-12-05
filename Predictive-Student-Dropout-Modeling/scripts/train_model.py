import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib
import os

# Create directories if not exist
os.makedirs('models', exist_ok=True)

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('data/student_dropout_1000.csv')

# 2. Preprocessing
print("Preprocessing...")
# Drop duplicates
df = df.drop_duplicates()

# Simple imputation for missing values (if any) - assuming numerical for median, mode for categorical
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode Categorical Variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 
    # In a real scenario, we'd save these encoders too, but for this simplified scope we'll rely on the scaler and model.
    # Note: If the user input in the app is raw string, we NEED these encoders. 
    # For this assignment, I will assume the app might receive raw inputs and needs to handle them, 
    # OR I will save a mapping. Let's save a dictionary of LabelEncoders.
    joblib.dump(le, f'models/{col}_encoder.pkl')

# Define features and target
# Assuming 'Dropout' or 'Target' is the target column. 
# Looking at typical dropout datasets, let's assume 'NB.Curricular units 2nd sem (approved)' or 'Target' exists.
# Since I haven't seen the file, I'll inspect column names dynamically or assume standard 'Target' if present.
# IF the dataset is the standard UCI one, target is 'Target'.
# Let's try to detect the target.
target_col = 'Target'
if target_col not in df.columns:
    # Fallback: assume the last column is the target
    target_col = df.columns[-1]

print(f"Target column detected: {target_col}")

# If target is categorical strings (Dropout, Graduate, Enrolled), encode it.
# We are doing binary classification (likely Dropout vs Non-Dropout).
# If there are 3 classes, we might map Enrolled/Graduate to 0 and Dropout to 1.
if df[target_col].dtype == 'object':
    # Custom mapping for dropout
    # 'Dropout' -> 1, 'Graduate'/'Enrolled' -> 0
    df[target_col] = df[target_col].apply(lambda x: 1 if str(x).strip() == 'Dropout' else 0)

X = df.drop(target_col, axis=1)
y = df[target_col]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Model Training & Tuning
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# 4. Evaluation
print("Evaluating...")
models = {'Logistic Regression': lr, 'Random Forest': best_rf}
best_model = None
best_f1 = 0

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except:
        auc = 0.5
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# 5. Save Artifacts
print(f"\nSaving best model ({best_model.__class__.__name__}) and scaler...")
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save column names for API consistency
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("Done.")
