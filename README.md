# HR Analytics: Employee Attrition Prediction

This project analyzes and predicts employee attrition using the **IBM HR Analytics Employee Attrition & Performance dataset**. It includes **exploratory data analysis (EDA)**, **predictive modeling (Logistic Regression & Random Forest)**, and a **Power BI dashboard** to support HR decision-making.

---

## Dataset

- **Source:** IBM HR Analytics Employee Attrition & Performance  
- **Records:** ~1,470 employees  
- **Target Variable:** `Attrition` (Yes/No)  
- **Features:** Demographics, job role, salary, satisfaction, performance, etc.

---

## Exploratory Data Analysis (EDA)

EDA was performed using Python and exported into visual and tabular form.

### Key Highlights:

---

## Modeling Pipeline

Implemented in `attrition_modeling_pipeline.py` using:
- **Logistic Regression**
- **Random Forest**

### Features:
- Preprocessing with `ColumnTransformer` (OneHot + Scaling)
- Train/Test split with stratification
- Outputs:
  - Classification report (CSV)
  - ROC curves & confusion matrices (PNG)
  - Predictions with probabilities (CSV)
  - Serialized pipelines (`.pkl`)

### Metrics:
- ROC AUC scores
- Precision / Recall / F1 per class
- Visual performance summaries

---

## Tech Stack

| Tool         | Use Case                     |
|--------------|------------------------------|
| Python (Pandas, Sklearn, Seaborn, Joblib) | EDA, modeling, outputs |
| Power BI     | Dashboard & visualization    |
| Excel        | Optional summaries            |

---
