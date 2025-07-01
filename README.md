
💡Customer Churn Prediction Project

 📋 Table of Contents

1. [Project Overview](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#project-overview)
2. [Dataset Description](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#dataset-description)
3. [Environment & Requirements](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#environment--requirements)
4. [Installation](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#installation)
5. [Project Structure](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#project-structure)
6. [Exploratory Data Analysis (EDA)](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#exploratory-data-analysis-eda)
7. [Feature Engineering](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#feature-engineering)
8. [Modeling](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#modeling)
    - [Logistic Regression](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#logistic-regression)
    - [Random Forest](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#random-forest)
    - [XGBoost](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#xgboost)
9. [Evaluation & Results](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#evaluation--results)
10. [Insights & Recommendations](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#insights--recommendations)
11. [Usage](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#usage)
12. [Future Work](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#future-work)
13. [References](https://chatgpt.com/c/6863cd17-0c5c-8005-92b1-77bec8678e20#references)

---

 🔍 Project Overview

Predicting customer churn is critical for telecommunications companies to reduce revenue loss and retain high-value customers. This project leverages machine learning techniques to:

- **Analyze** customer behavior patterns
- **Identify** key drivers of churn
- **Build** predictive models with strong performance on imbalanced data
- **Provide** actionable business recommendations

The final deliverables include a cleaned dataset, trained model artifact, performance metrics, and visual interpretations (SHAP, PCA clusters).

---
 📂 Dataset Description

- **Source**: Telco Customer Churn dataset (7,032 records, 20 features).
- **Key Features**:
    - `gender`, `SeniorCitizen`, `Partner`, `Dependents`
    - `tenure`, `PhoneService`, `MultipleLines`, `InternetService`
    - `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
    - `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`
    - `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
    - **Target**: `Churn` (0 = No, 1 = Yes)

---
🛠️ Environment & Requirements

- Python 3.8+
- Libraries:
    - pandas, numpy, scikit-learn
    - imbalanced-learn, xgboost, shap
    - matplotlib, seaborn, plotly

```bash
pip install -r requirements.txt

```

---
⚙️ Installation

```bash
git clone https://github.com/username/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt

```

---
 🗂️ Project Structure

```
├── data
│   ├── raw              # Original Telco CSV
│   └── processed        # Cleaned dataset (cleaned_churn_data.csv)
├── notebooks
│   └── EDA.ipynb        # Exploratory Data Analysis
├── models
│   └── best_churn_model.pkl
├── src
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── predict.py
├── visuals              # Plots: boxplots, violin, SHAP, PCA clusters
├── requirements.txt
└── README.md

```

---
🔎 Exploratory Data Analysis (EDA)

1. **Univariate & Bivariate Analysis**:
    - **Boxplots & Violin Plots** for `tenure`, `MonthlyCharges`, `TotalCharges`:
        - Churners have significantly shorter tenure (median ~10 months vs ~38 months) and higher monthly charges (~80–100 vs ~25–75).
        - TotalCharges distributions show skew; churners concentrated at lower total spend.
    - **Density (KDE) Plots**:
        - `tenure`: churners peak at low tenures (<20 months), non-churners show bimodal distribution with peaks at ~20 and ~75 months.
        - `MonthlyCharges`: churners concentrated in mid-to-high charge bands (~60–100), non-churners spread wider with a peak at lower charges (~25–40).
        - `TotalCharges_log`: churners cluster at lower log values (~3.5–7), whereas non-churners peak at higher values (~7–9), indicating higher lifetime spending among retained customers.
2. **Feature Distributions & Transformations**:
    - Skewed distribution in `TotalCharges` addressed via log-transform, yielding clearer separation between churn groups.
3. **Customer Segmentation (PCA + K-Means)**:
    - After reducing dimensions via PCA, K-Means clustering identified four distinct segments with these characteristics:
        
        
        | Cluster | Churn Rate | Avg. CLV | Avg. MonthlyCharges | Avg. Tenure(In Months) |
        | --- | --- | --------  | ---     | ---------------     |
        | 0         |5.7%       | ₹1,919.85 | ₹36.98            | 53.56        |
        | 1         | 15.4%     | ₹5,586.17 | ₹93.63            | 59.81        |
        | 2         | 24.2%     | ₹306.38   | ₹31.80            | 10.60        |
        | 3         | 48.6%     | ₹1,253.17 | ₹81.07            | 15.35        |
    - **Insights**:
        - **Cluster 0**: Long-tenure, low-spend customers exhibit the lowest churn (5.7%).
        - **Cluster 1**: High-value, long-tenure segment with moderate churn (15.4%), representing potential upsell targets.
        - **Cluster 2**: New, low-CLV customers face moderate churn (24.2%), requiring onboarding engagement.
        - **Cluster 3**: New-to-mid tenure, mid-spend customers show the highest churn (48.6%), main targets for retention offers.
4. **Churn Pattern Visualization (Bubble Dashboard)**:
    - Bubble chart (`tenure` vs `MonthlyCharges`, bubble size = CLV):
        - High-CLV, new customers with steep monthly charges light up in yellow (high churn probability).
        - Low-CLV long-tenure customers (small blue bubbles) show minimal churn.
5. **SHAP Analysis**:
    - **Global Feature Importance**:
        - Top drivers: `PaymentMethod_Electronic check`, `PaperlessBilling_No`, `Dependents_Yes`, `SeniorCitizen_1`, `Contract_Month-to-month`, `TechSupport_No`, `OnlineBackup_No`, `gender_Male`, `OnlineSecurity_No`, `DeviceProtection_No`.
    - **SHAP Beeswarm Insights**:
        - **PaymentMethod_Electronic check**: Customers using electronic check show the highest positive SHAP values, significantly increasing churn risk.
        - **PaperlessBilling_No**: Absence of paperless billing contributes positively to churn likelihood.
        - **Dependents_Yes**: Having dependents correlates with higher churn probability in many cases, suggesting family plans may require tailored retention offers.
        - **SeniorCitizen_1**: Senior citizens exhibit elevated churn propensity when compared to non-seniors.
        - **Contract_Month-to-month**: Month-to-month contracts show concentrated positive SHAP values, reinforcing higher churn risk for short-term commitments.
        - **Service Breaks**: Lack of tech support, online backup, security, or device protection all push SHAP values positive, indicating service dissatisfaction drives churn.

---
🔧 Feature Engineering

- **Missing Value Handling**: Replaced blanks in `TotalCharges`, imputed median.
- **Encoding**:
    - One-hot encoding for categorical features (e.g., `PaymentMethod_Electronic check`).
    - Binary mapping for Yes/No features.
- **Scaling**: StandardScaler for numerical features.
- **Derived Features**:
    - `CLV` (Customer Lifetime Value)
    - Log-transform of `TotalCharges`
    - Interaction terms (e.g., `SeniorCitizen` × `MonthlyCharges`)

---
🤖 Modeling

### Logistic Regression

- **Pipeline**: ColumnTransformer → LogisticRegression (balanced weights)
- **Performance**:
    - Accuracy: 76.7%
    - Precision: 54.5%
    - Recall: 74.6%
    - AUC-PR: 0.652 — *Best balance of precision & recall*

### Random Forest

- **Parameters**: GridSearchCV tuning
- **Performance**:
    - Accuracy: 78.3%
    - Precision: 59.4%
    - Recall: 57.5%
    - AUC-PR: 0.609

### XGBoost

- **Parameters**: GridSearchCV tuning
- **Performance**:
    - Accuracy: 77.8%
    - Precision: 57.5%
    - Recall: 63.9%
    - AUC-PR: 0.641

**Recommended Model**: Logistic Regression (highest AUC-PR)

---
 📊 Evaluation & Results

| Model | Accuracy | Precision | Recall | AUC-PR |
| --- | --- | --- | --- | --- |
| **LogisticRegression** | 0.7669 | 0.5449 | 0.7460 | 0.6520 |
| XGBoost | 0.7783 | 0.5745 | 0.6390 | 0.6410 |
| RandomForest | 0.7825 | 0.5939 | 0.5749 | 0.6089 |
- **SHAP Analysis**:
    - Top drivers: `PaymentMethod_Electronic check`, `PaperlessBilling_No`, `Dependents_Yes`, `SeniorCitizen_1`.

---
 💡 Insights & Recommendations

1. **Top Churn Correlates**:
    - High monthly charges, electronic payment methods, short-tenure segments.
2. **High-Risk Customer Segments**:
    - New & Month-to-month contracts (55.2% churn)
    - Mid-tenure & Month-to-month (39.6% churn)
3. **Business Actions**:
    - Incentivize longer contracts with discounts/add-ons.
    - Tailored retention campaigns for new/mid-tenure customers.
    - Introduce tiered bundles to lower perceived cost for high monthly spenders.
    - Senior-focused plans and dedicated support.
4. **Financial Impact**:
    - Each 1% churn reduction = ~$4,557 savings per month.
5. **CLV Analysis**:
    - Churn = 0 customers: average CLV ₹2,555.20
    - Churn = 1 customers: average CLV ₹1,531.61
    - Non-churned customers are worth ~66.8% more than churned ones.

---
 🚀 Usage

1. **Predict New Data**:
    python src/predict.py --input new_customers.csv --model models/best_churn_model.pkl --output predictions.csv
    
2. **Interpret Predictions**:
    - Use SHAP scripts in `src/interpretation.py` to generate feature impact plots.

---

 🔮 Future Work

- Experiment with advanced ensemble methods (Stacking, AutoML).
- Integrate time-series analysis for churn timing prediction.
- Develop a real-time prediction API with Flask or FastAPI.
- A/B test targeted retention offers to validate impact.

---
📚 References

1. IBM Telco Customer Churn dataset
2. Lundberg SHAP Documentation
3. Scikit-learn ML Pipelines
   
