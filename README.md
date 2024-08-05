# Customer Churn Prediction
This repository contains code and resources for predicting customer churn using various machine learning techniques. The project focuses on understanding the key factors that influence customer churn and building predictive models to identify customers who are likely to churn.

# Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

# Introduction
Customer churn is a critical problem faced by many businesses. Identifying customers who are likely to churn helps companies to take proactive measures to retain them. This project aims to analyze and predict customer churn using the Telco Customer Churn dataset.

# Dataset
The dataset used in this project is the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn) which contains information about customer demographics, account information, and services availed.

# Data Preprocessing
- **Missing Values**: Handled missing values in the 'TotalCharges' column by dropping rows with empty strings.
- **Encoding**: Categorical features were encoded using Label Encoding and One-Hot Encoding techniques.
- **Scaling**: Numerical features were scaled using StandardScaler.

# Exploratory Data Analysis
Exploratory Data Analysis (EDA) was conducted to understand the distribution and relationships of various features:
- **Pie Charts**: Distribution of churn vs. non-churn customers.
- **Bar Plots**: Churn rates by gender, dependents, senior citizens, internet service, and payment methods.
- **KDE Plots**: Distribution of numerical features like MonthlyCharges and TotalCharges by churn status.
- **Scatter Plots**: Relationships between numerical features like tenure, monthly charges, and total charges.

## Feature Engineering
- **Customer Lifetime Value (CLV)**: Calculated as `MonthlyCharges * Tenure`.
- **TotalCharges Groups**: Created bins for `TotalCharges` to group customers based on total charges.

## Modeling
Several machine learning models were used to predict customer churn:
- **Logistic Regression**
- **Random Forest Classifier**
- **K-Means Clustering**

**GridSearchCV** was used for hyperparameter tuning. The dataset was split into training and testing sets, and models were evaluated using various metrics.

## Evaluation
The performance of the models was evaluated using the following metrics:
- **Accuracy**
- **Recall**
- **Precision**
- **F1 Score**
- **ROC AUC Score**

Confusion matrix and classification reports were generated to provide detailed insights into model performance.

## Conclusion
This project provides a comprehensive analysis and prediction of customer churn. The models developed can help businesses identify potential churners and take necessary actions to retain them.

## Requirements
- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- plotly
- scikit-learn
- rich

## Usage
1. Clone the repository:
    git clone https://github.com/vj0212/Customer_churn_Prediction.git
    cd Customer_Churn_Prediction

2. Install the required packages:
    pip install -r requirements.txt

3. Run the Jupyter notebook:
    jupyter notebook

4. Open `Customer_Churn_Prediction.ipynb` to explore the analysis and models.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

