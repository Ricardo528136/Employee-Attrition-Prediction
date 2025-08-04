# Import libraries
import pandas as pd
import numpy as np
import json
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Configurations
DATA_PATH = "outputs\cleaned\cleaned_data.csv"
METADATA_PATH = "outputs\metadata.json"
OUTPUT_DIR = "outputs/modeling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COLUMN = "Attrition"
RANDOM_STATE = 7
TEST_SIZE = 0.2

# Load data
df = pd.read_csv(DATA_PATH)
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

categorical_cols = metadata['categorical_columns']
numerical_cols = metadata['numerical_columns']
features = categorical_cols + numerical_cols
X = df[features]
y = df[TARGET_COLUMN]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Preprocessing
preprocessor = ColumnTransformer(transformers =[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
}

results = []

# Train, evaluate and save models
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{model_name}_pipeline.pkl")
    joblib.dump(pipeline, model_path)

    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df['roc_auc'] = roc_auc
    report_csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_metrics.csv")
    report_df.to_csv(report_csv_path)

    # Save predictions
    preds_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Probability': y_proba
    })
    preds_csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_predictions.csv")
    preds_df.to_csv(preds_csv_path, index=False)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Attrition', 'Attrition'], yticklabels=['No Attrition', 'Attrition'])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_roc_curve.png"))
    plt.close()

    print(f"{model_name} training and evaluation completed.")