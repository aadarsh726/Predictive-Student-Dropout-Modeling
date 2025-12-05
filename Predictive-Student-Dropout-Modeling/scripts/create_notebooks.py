import json
import os
import textwrap

os.makedirs('notebooks', exist_ok=True)

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

def get_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    }

def get_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split('\n')]
    }

def create_eda_notebook():
    cells = [
        get_markdown_cell("# Exploratory Data Analysis (EDA)\n\n## 1. Introduction\nThis notebook analyzes the student dropout dataset."),
        get_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom scipy import stats"),
        get_markdown_cell("## 2. Load Dataset"),
        get_code_cell("df = pd.read_csv('../data/student_dropout_1000.csv')\ndf.head()"),
        get_markdown_cell("## 3. Data Overview\nShape, Info, and duplicates."),
        get_code_cell("print(f'Shape: {df.shape}')\ndf.info()\nprint(f'Duplicates: {df.duplicated().sum()}')"),
        get_markdown_cell("## 4. Null Value Report"),
        get_code_cell("df.isnull().sum()"),
        get_markdown_cell("## 5. Statistical Summary"),
        get_code_cell("df.describe()"),
        get_markdown_cell("## 6. Correlation Heatmap"),
        get_code_cell("plt.figure(figsize=(12, 10))\nnumeric_df = df.select_dtypes(include=[np.number])\nsns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')\nplt.title('Correlation Heatmap')\nplt.show()"),
        get_markdown_cell("## 7. Distribution Analysis"),
        get_code_cell("df.hist(figsize=(15, 15), bins=20)\nplt.tight_layout()\nplt.show()"),
        get_markdown_cell("## 8. Statistical Hypothesis Tests\n\n### T-Test\nTesting if there is a significant difference in a feature between Dropouts and Non-Dropouts."),
        get_code_cell(textwrap.dedent("""
            # Assuming 'Target' column exists, else detecting it
            target = 'Target' if 'Target' in df.columns else df.columns[-1]
            
            # Identify a numeric column for t-test
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col_to_test = numeric_cols[0]
                group1 = df[df[target] == 'Dropout'][col_to_test]
                group2 = df[df[target] != 'Dropout'][col_to_test] # Graduate/Enrolled
                
                # If groups are empty (encoding might be needed or target values differ), skip or handle
                if len(group1) > 0 and len(group2) > 0:
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    print(f"T-Test for {col_to_test}: Stat={t_stat}, P-value={p_val}")
        """)),
        get_markdown_cell("### Chi-Square Test\nTesting independence for categorical features."),
        get_code_cell(textwrap.dedent("""
            # Identify a categorical column
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 1: # Need feature + target
                col_test = cat_cols[0] if cat_cols[0] != target else cat_cols[1]
                contingency_table = pd.crosstab(df[col_test], df[target])
                chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
                print(f"Chi-Square Test for {col_test}: Stat={chi2}, P-value={p}")
        """))
    ]
    create_notebook('notebooks/01_EDA.ipynb', cells)

def create_training_notebook():
    cells = [
        get_markdown_cell("# Model Training\n\n## 1. Mathematical Foundations (MSc Level)\n\n### Entropy\n$$ H(S) = -\\sum p_i \\log_2(p_i) $$\nMeasure of impurity in a dataset.\n\n### Gini Index\n$$ Gini = 1 - \\sum p_i^2 $$\nProbability of incorrect classification.\n\n### Logistic Regression\nUses the sigmoid function:\n$$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$\nCost function optimized via Gradient Descent.\n\n### Metrics\n- **Precision**: TP / (TP + FP)\n- **Recall**: TP / (TP + FN)\n- **F1-Score**: Harmonic mean of Precision and Recall.\n- **ROC-AUC**: Area under the Receiver Operating Characteristic curve."),
        get_code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib"),
        get_markdown_cell("## 2. Load and Prepare Data"),
        get_code_cell("df = pd.read_csv('../data/student_dropout_1000.csv')\n\n# Basic Preprocessing duplication\ndf = df.drop_duplicates()\n\n# Encode Target\ntarget = 'Target' if 'Target' in df.columns else df.columns[-1]\nif df[target].dtype == 'object':\n    df[target] = df[target].apply(lambda x: 1 if str(x).strip() == 'Dropout' else 0)\n\n# Encode others\nfor col in df.select_dtypes(include=['object']).columns:\n    if col != target:\n        le = LabelEncoder()\n        df[col] = le.fit_transform(df[col])\n        \n# Fill NaNs\ndf = df.fillna(df.median(numeric_only=True))"),
        get_markdown_cell("## 3. Splitting and Scaling"),
        get_code_cell("X = df.drop(target, axis=1)\ny = df[target]\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)"),
        get_markdown_cell("## 4. Logistic Regression"),
        get_code_cell("lr = LogisticRegression()\nlr.fit(X_train, y_train)\ny_pred_lr = lr.predict(X_test)\nprint(classification_report(y_test, y_pred_lr))"),
        get_markdown_cell("## 5. Random Forest & GridSearchCV"),
        get_code_cell(textwrap.dedent("""
            rf = RandomForestClassifier(random_state=42)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
            grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_rf = grid.best_estimator_
            y_pred_rf = best_rf.predict(X_test)
            print("Best Params:", grid.best_params_)
            print(classification_report(y_test, y_pred_rf))
        """)),
        get_markdown_cell("## 6. Confusion Matrix & ROC"),
        get_code_cell("sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')\nplt.title('Confusion Matrix (RF)')\nplt.show()"),
        get_markdown_cell("## 7. Save Models"),
        get_code_cell("joblib.dump(best_rf, '../models/model.pkl')\njoblib.dump(scaler, '../models/scaler.pkl')\nprint('Models saved.')")
    ]
    create_notebook('notebooks/02_Model_Training.ipynb', cells)

if __name__ == "__main__":
    create_eda_notebook()
    create_training_notebook()
    print("Notebooks generated successfully.")
