# Customer Churn Prediction Project

## Project Overview

This project focuses on predicting customer churn using supervised machine learning techniques.

Customer churn prediction helps businesses identify customers who are likely to discontinue services, enabling proactive retention strategies.

The project follows a production-inspired machine learning workflow, including:

- exploratory data analysis
- data preprocessing
- feature engineering
- model benchmarking
- hyperparameter tuning
- final model deployment preparation

---

## Problem Statement

Customer retention is critical for subscription-based businesses.

Acquiring new customers is often significantly more expensive than retaining existing ones.

The objective of this project is to build a machine learning model capable of predicting whether a customer is likely to churn.

Target variable:

- Churn
    - Yes → customer likely to churn
    - No → customer likely to remain

---

## Dataset

Dataset used:

Telco Customer Churn Dataset

Features include:

- customer demographics
- subscription details
- billing information
- service usage patterns

Key features:

- tenure
- MonthlyCharges
- TotalCharges
- Contract
- InternetService
- PaymentMethod
- TechSupport
- StreamingTV
- OnlineSecurity

---

## Project Structure

```text
project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
│
├── model.pkl
├── requirements.txt
├── README.md
└── app.py
```

---

## Workflow

### 1. Exploratory Data Analysis

Performed:

- dataset inspection
- missing value analysis
- target distribution analysis
- numerical feature analysis
- categorical feature analysis
- business insight extraction

---

### 2. Data Preprocessing

Preprocessing pipeline includes:

- missing value imputation
- median imputation for numerical features
- most frequent imputation for categorical features
- standard scaling
- one-hot encoding
- unknown category handling

Implementation:

- sklearn Pipeline
- ColumnTransformer

---

### 3. Model Benchmarking

Benchmarked models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

### 4. Hyperparameter Tuning

Performed using:

- GridSearchCV
- 5-fold cross-validation

Tuned models:

- Logistic Regression
- Gradient Boosting

---

### 5. Final Model Selection

Selected model:

**Logistic Regression**

Reasons:

- strong recall performance
- competitive ROC-AUC
- simpler deployment
- lower complexity
- interpretability
- stable generalization performance

---

## Model Performance
  
    Final evaluation metrics:
    Metrics
    ==================================================
- Accuracy : 0.7991
- Precision: 0.6608
- Recall   : 0.5000
- F1 Score : 0.5693
- ROC AUC  : 0.8457
---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run notebook:

```bash
jupyter notebook
```

Run application:

```bash
streamlit run app.py
```

---

## Future Improvements

Potential enhancements:

- XGBoost benchmarking
- threshold optimization
- SHAP explainability
- feature selection
- deployment on cloud platforms

---

## Author
vtjaseem7@gmail.com
--Machine Learning Project--
