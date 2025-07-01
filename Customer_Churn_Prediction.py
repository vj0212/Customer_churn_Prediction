import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import shap
import joblib
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load and Clean Data
def load_data(path):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True) 
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() < 10:
            df[col] = df[col].astype('category')
    return df

# 2. Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("\n\n-*-*-* EDA: Data Overview -*-*-*-\n")
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nTarget Distribution:\n", df['Churn'].value_counts(normalize=True))

    df['TotalCharges_log'] = np.log1p(df['TotalCharges'])
    df['CLV'] = df['MonthlyCharges'] * df['tenure']

    # Violin Plots
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(['tenure', 'MonthlyCharges', 'TotalCharges'], 1):
        plt.subplot(1, 3, i)
        sns.violinplot(data=df, x='Churn', y=col)
        plt.title(f'Violin Plot - {col} by Churn')
    plt.tight_layout()
    plt.show()

    # Correlation Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Dynamic Bubble Dashboard
    fig = px.scatter(
        df,
        x='tenure',
        y='MonthlyCharges',
        color='Churn',
        size='CLV',
        hover_data=['Contract', 'PaymentMethod', 'InternetService'],
        title='Churn Pattern by Tenure & Monthly Charges (Bubble Size = CLV)',
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    fig.show()

    plt.figure(figsize=(12, 4))
    for i, col in enumerate(['tenure', 'MonthlyCharges', 'TotalCharges_log'], 1):
        plt.subplot(1, 3, i)
        sns.kdeplot(data=df, x=col, hue='Churn', fill=True)
        plt.title(f"{col} Distribution by Churn")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    for i, col in enumerate(['tenure', 'MonthlyCharges', 'TotalCharges'], 1):
        plt.subplot(1, 3, i)
        sns.boxplot(data=df, x='Churn', y=col)
        plt.title(f"{col} vs Churn")
    plt.tight_layout()
    plt.show()

    print("\n\n-*-*-*  CLV Analysis *-*-*-\n")
    clv_mean = df.groupby('Churn')['CLV'].mean()
    gain = ((clv_mean[0] - clv_mean[1]) / clv_mean[1]) * 100
    print(f"Churn=0 CLV: ₹{clv_mean[0]:,.2f}, Churn=1 CLV: ₹{clv_mean[1]:,.2f}")
    print(f"\n Non-churned customers are worth ~{gain:.1f}% more than churned ones.\n")

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='Churn', y='CLV')
    plt.title('CLV Distribution by Churn')
    plt.show()

# 3. Feature Engineering
def engineer_features(df):
    df['TenureSegment'] = pd.cut(df['tenure'], bins=[0, 6, 24, 72], labels=['New', 'Mid', 'Loyal'])
    df['SpendingRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] / df['tenure'].replace(0, 0.1))
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = df[services].apply(lambda x: sum(x != 'No'), axis=1)
    df['HighCostLongTenure'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].median()) & 
                                (df['tenure'] > df['tenure'].median())).astype(int)
    df['SeniorCitizen_MonthlyCharges'] = df['SeniorCitizen'] * df['MonthlyCharges']  # Added interaction term
    return df

# 4. Clustering for Segmentation
def cluster_customers(df):
    X_cluster = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV']]
    X_scaled = StandardScaler().fit_transform(X_cluster)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='tab10')
    plt.title("Customer Segments via K-Means (PCA-reduced)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()

    print("\nCluster Summary:\n")
    print(df.groupby('Cluster')[['Churn', 'CLV', 'MonthlyCharges', 'tenure']].mean())
    return df

# 5. Model Training and Comparison
def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500, class_weight='balanced'),  # Added class_weight
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),  # Added class_weight
        'XGBoost': XGBClassifier(eval_metric='aucpr', random_state=42, n_jobs=-1)
    }
    results = {}
    for name, clf in models.items():
        pipe = make_imb_pipeline(
            preprocessor,
            SMOTE(random_state=42),
            clf
        )
        if name == 'XGBoost':
            param_grid = {
                'xgbclassifier__max_depth': [3, 5],
                'xgbclassifier__learning_rate': [0.05, 0.1],
                'xgbclassifier__subsample': [0.8, 1.0],
                'xgbclassifier__colsample_bytree': [0.8, 1.0]
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='recall', verbose=1, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            results[name] = grid_search.best_estimator_
        else:
            pipe.fit(X_train, y_train)
            results[name] = pipe
    return results

# 6. Model Evaluation
def evaluate_models(models, X_test, y_test):
    summary = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        summary.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'AUC-PR': pr_auc
        })

        print(f"\n {name} Performance Metrics:")
        print(f"Confusion Matrix:\n{cm}")
        tn, fp, fn, tp = cm.ravel()
        print("\n Confusion Matrix Interpretation:")
        print(f"""
        True Negatives (TN): {tn} — Correctly predicted non-churners
        False Positives (FP): {fp} — Predicted churn but actually stayed (false alarm)
        False Negatives (FN): {fn} — Missed actual churners (critical loss)
        True Positives (TP): {tp} — Correctly predicted churners
        """)
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        print("\n Metric Definitions:")
        print(f"- Accuracy: {accuracy_score(y_test, y_pred):.2%} — Overall correctness of predictions.")
        print(f"- Precision: {precision_score(y_test, y_pred):.2%} — Proportion of predicted churners who actually churned.")
        print(f"- Recall: {recall_score(y_test, y_pred):.2%} — Proportion of actual churners correctly identified.")
        print(f"- AUC-PR: {pr_auc:.3f} — Balances precision and recall, ideal for imbalanced data.")

    eval_df = pd.DataFrame(summary).sort_values('AUC-PR', ascending=False)  # Fixed syntax
    print("\n Model Comparison (Sorted by AUC-PR):")
    print(eval_df)

    best_model = eval_df.iloc[0]['Model']
    print(f"\n Recommended Model: {best_model}")
    print(f"- Basis: Highest AUC-PR ({eval_df.iloc[0]['AUC-PR']:.3f}) for best precision-recall balance.")
    print(f"- Usage: {best_model} for proactive churn intervention and targeting high-risk segments.")
    print(f"- Logistic Regression: For interpretable insights (e.g., feature impact).")
    print(f"- Random Forest: For robust non-linear patterns.")
    print(f"- XGBoost: Optimized for recall, ideal for catching most churners.")

    return eval_df

# 7. SHAP Interpretation for Top 2 Models
def explain_xgboost_top2(models, X_sample, cat_features, num_features, eval_df):
    top_2_models = eval_df.head(2)['Model'].tolist()
    print(f"\n🔍 Analyzing SHAP for top 2 models: {top_2_models}")
    
    for model_name in top_2_models:
        print(f"\nProcessing {model_name}...")
        model = models[model_name]
        preprocessor = model.named_steps['columntransformer']
        
        X_proc = preprocessor.transform(X_sample)
        if hasattr(X_proc, 'toarray'):
            X_proc = X_proc.toarray()
        ohe_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features))
        all_features = num_features + ohe_cols

        if model_name in ['XGBoost', 'RandomForest']:
            xgb = model.named_steps['xgbclassifier'] if 'xgbclassifier' in model.named_steps else model.named_steps['randomforestclassifier']
            try:
                explainer = shap.TreeExplainer(xgb)
                shap_values = explainer.shap_values(X_proc)
                if shap_values is None or len(shap_values) == 0:
                    print(f" SHAP values for {model_name} are empty. Skipping visualization.")
                    continue
                
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_importance_df = pd.DataFrame({
                    'Feature': all_features,
                    'Mean SHAP Value': mean_abs_shap
                }).sort_values(by='Mean SHAP Value', ascending=False).head(10)

                plt.figure(figsize=(10, 6))
                sns.barplot(data=feature_importance_df, x='Mean SHAP Value', y='Feature', palette='viridis')
                plt.title(f'Top 10 Features for Churn - {model_name} (SHAP)')
                plt.xlabel('Mean SHAP Value')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.show()
                
                shap.summary_plot(shap_values, X_proc, feature_names=all_features)
            except Exception as e:
                print(f" Error computing SHAP for {model_name}: {str(e)}. Skipping.")
        else:
            print(f"\n {model_name} is a linear model; using coefficients.")
            coefs = model.named_steps['logisticregression'].coef_[0]
            feature_importance_df = pd.DataFrame({
                'Feature': all_features,
                'Coefficient': coefs
            }).sort_values(by='Coefficient', ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance_df, x='Coefficient', y='Feature', palette='viridis')
            plt.title(f'Top 10 Features by Coefficient - {model_name}')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

# 8. Predict Churn for Random Customers
def predict_multiple_customers(X_test, model, n=10, seed=42):
    np.random.seed(seed)
    sampled_customers = X_test.sample(n=n).copy()
    predictions = []
    for _, row in sampled_customers.iterrows():
        result = predict_churn(row.to_dict(), model=model)
        result['customer_index'] = row.name
        predictions.append(result)
    result_df = pd.DataFrame(predictions).set_index('customer_index')
    output = sampled_customers.join(result_df)[['churn_prediction', 'churn_probability']]
    output.to_csv("random_customer_predictions.csv")
    print("\n Predictions for 10 Random Customers Saved to 'random_customer_predictions.csv'")
    print(output)
    return output

def predict_churn(customer_data, model=None, model_path='best_churn_model.pkl'):
    if model is None:
        model = joblib.load(model_path)
    if not isinstance(customer_data, pd.DataFrame):
        customer_data = pd.DataFrame([customer_data])
    if 'TotalCharges_log' not in customer_data.columns:
        customer_data['TotalCharges_log'] = np.log1p(customer_data['TotalCharges'])
    return {
        'churn_prediction': 'Yes' if model.predict(customer_data)[0] == 1 else 'No',
        'churn_probability': float(model.predict_proba(customer_data)[:, 1][0]),
        'model_version': '1.0'
    }

# 9. Dynamic High-Risk Segment Identification for Top 2 Models
def identify_high_risk_segments_top2(df, models, cat_features, num_features, eval_df=None, top_n_segments=3):
    model_names = eval_df.head(2)['Model'].tolist() if eval_df is not None else list(models.keys())[:1]
    
    for model_name in model_names:
        model = models.get(model_name, models['best'] if 'best' in models else models)
        if model is None:
            print(f"\n Model '{model_name}' not found; skipping.")
            continue
        
        preprocessor = model.named_steps['columntransformer']
        ohe_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features))
        all_features = num_features + ohe_cols

        if model_name in ['XGBoost', 'RandomForest'] or ('xgbclassifier' in model.named_steps or 'randomforestclassifier' in model.named_steps):
            xgb = model.named_steps['xgbclassifier'] if 'xgbclassifier' in model.named_steps else model.named_steps['randomforestclassifier']
            importances = xgb.feature_importances_
            fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        else:
            try:
                coefs = np.abs(model.named_steps['logisticregression'].coef_[0])
                fi_df = pd.DataFrame({'Feature': all_features, 'Importance': coefs}).sort_values(by='Importance', ascending=False)
            except KeyError:
                print(f"\n {model_name} model type not supported for feature importance.")
                fi_df = pd.DataFrame({'Feature': all_features, 'Importance': np.zeros(len(all_features))})

        cat_features_set = set(cat_features)
        top_cat_features = [f for f in fi_df['Feature'] if any(c in f for c in cat_features_set)][:2]
        top_cat_features_base = [cat for cat in cat_features if any(cat in f for f in top_cat_features)]

        selected_features = ['TenureSegment', 'Contract'] if all(f in df.columns for f in ['TenureSegment', 'Contract']) else top_cat_features_base[:2]
        print(f"\n High-Risk Segments for {model_name}:")
        if selected_features:
            high_risk_segments = df.groupby(selected_features)['Churn'].mean().sort_values(ascending=False).head(top_n_segments)
            if isinstance(high_risk_segments.index, pd.MultiIndex):
                high_risk_segments.index = high_risk_segments.index.map(lambda x: '|'.join(x))
            plt.figure(figsize=(10, 6))
            sns.barplot(x=high_risk_segments.index, y=high_risk_segments.values, palette='Blues')
            plt.title(f'Top {top_n_segments} High-Risk Segments - {model_name}')
            plt.xlabel('Segment')
            plt.ylabel('Churn Rate (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            print(high_risk_segments.to_string(float_format=lambda x: f'{x*100:.1f}%'))
        else:
            print("No suitable categorical features identified.")
    return high_risk_segments

# 10. Stakeholder Report Highlights
def stakeholder_summary(df, model, cat_features, num_features):
    print("\n=== BUSINESS INSIGHTS ===")
    print(f"1. Overall Churn Rate: {df['Churn'].mean():.1%}")
    
    churn_corr = df.corr(numeric_only=True)['Churn'].sort_values(ascending=False)
    print("\n2. Top Churn Correlations:")
    print(churn_corr[1:6])
    
    identify_high_risk_segments_top2(df, {'best': model}, cat_features, num_features, None)
    
    avg_revenue = df['MonthlyCharges'].mean()
    print(f"\n4. Financial Impact: Each 1% churn reduction = ~${avg_revenue*len(df)*0.01:,.0f} savings/month")
    
    clv_mean = df.groupby('Churn')['CLV'].mean()
    clv_diff = ((clv_mean[0] - clv_mean[1]) / clv_mean[1]) * 100
    print(f"\n5. CLV Analysis:")
    print(f"Churn=0 CLV: ₹{clv_mean[0]:,.2f}, Churn=1 CLV: ₹{clv_mean[1]:,.2f}")
    print(f" Non-churned customers are worth ~{clv_diff:.1f}% more than churned ones.")
    
    print("\n6. Growth Strategy:")
    print("- Promote Longer-Term Contracts: Offer discounts for switching to one/two-year contracts.")
    print("- Targeted Retention: Personalized offers for new/mid-tenure month-to-month customers.")
    print("- Optimize Pricing: Tiered pricing for high MonthlyCharges customers.")
    print("- Engage Senior Citizens: Senior-friendly packages to lower churn.")
    print("- Loyalty Program: Exclusive benefits for high-CLV customers.")
    print(f"- Proactive Intervention: Use {model.named_steps[list(model.named_steps.keys())[0]].__class__.__name__} to target high-risk customers monthly.")
    
    df.to_csv("cleaned_churn_data.csv", index=False)
    print("\nProcessed data saved to 'cleaned_churn_data.csv'")

# Main execution flow
if __name__ == '__main__':
    df = load_data(r"WA_Fn-UseC_-Telco-Customer-Churn.csv")
    perform_eda(df)
    df = engineer_features(df)
    df = cluster_customers(df)
    y = df['Churn']
    X = df.drop(['Churn'], axis=1)
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges_log', 'CLV', 'SpendingRatio', 'ServiceCount', 'HighCostLongTenure', 'SeniorCitizen_MonthlyCharges']
    cat_features = [col for col in X.columns if col not in num_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ])

    models = train_models(X_train, y_train)

    eval_df = evaluate_models(models, X_test, y_test)
    print("\n-*-*-*  Model Comparison -*-*-* ")
    print(eval_df)

    explain_xgboost_top2(models, X_train.sample(100), cat_features, num_features, eval_df)

    identify_high_risk_segments_top2(df, models, cat_features, num_features, eval_df)

    best_model_name = eval_df.iloc[0]['Model']
    best_model = models[best_model_name]
    joblib.dump(best_model, 'best_churn_model.pkl')

    predict_multiple_customers(X_test, best_model, n=10, seed=42)

    stakeholder_summary(df, best_model, cat_features, num_features)

    print(f"\n Best model saved as 'best_churn_model.pkl' ({best_model_name}) and used for predictions and segmentation.")
