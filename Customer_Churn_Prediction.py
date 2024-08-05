from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import Counter
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from matplotlib.ticker import FixedLocator
from sklearn.ensemble import RandomForestClassifier
from statistics import stdev
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from rich import print
from rich.panel import Panel

Initial_df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
Initial_df.info()
print(" ")

print("\nInitial_df shape:",Initial_df.shape)

Initial_df.head()
Initial_df.describe(include='all')
print(" ")

print('\nValues in gender:', Initial_df.gender.unique())
print('Values in SeniorCitizen:',Initial_df.SeniorCitizen.unique())
print('Values in Partner:',Initial_df.Partner.unique())
print('Values in Dependents:',Initial_df.Dependents.unique())
print('Values in PhoneService:',Initial_df.PhoneService.unique())
print('Values in MultipleLines:',Initial_df.MultipleLines.unique())
print('Values in InternetService:',Initial_df.InternetService.unique())
print('Values in OnlineSecurity:', Initial_df.OnlineSecurity.unique())
print('Values in OnlineBackup:', Initial_df.OnlineBackup.unique())
print('Values in DeviceProtection:', Initial_df.DeviceProtection.unique())
print('Values in TechSupport:', Initial_df.TechSupport.unique())
print('Values in StreamingTV:', Initial_df.StreamingTV.unique())
print('Values in StreamingMovies:', Initial_df.StreamingMovies.unique())
print('Values in Contract:', Initial_df.Contract.unique())
print('Values in PaperlessBilling:', Initial_df.PaperlessBilling.unique())
print('Values in PaymentMethod:', Initial_df.PaymentMethod.unique())
print('Values in Churn:', Initial_df.Churn.unique())

Processed_df = Initial_df.drop('customerID', axis=1)
print(" ")

Processed_df.isna().sum()
print(" ")

Empty_String_Total_Charges1 = [len(i.split()) for i in Processed_df['TotalCharges']] 
Empty_String_Total_Charges2 = [i for i in range(len(Empty_String_Total_Charges1)) if Empty_String_Total_Charges1[i] != 1] 
print('\nNumber of entries with empty string: ', len(Empty_String_Total_Charges2))

Processed_df = Processed_df.drop(Empty_String_Total_Charges2, axis = 0).reset_index(drop=True) 
Processed_df['TotalCharges'] = Processed_df['TotalCharges'].astype(float)

print('\nNumber of duplicated values in training dataset: ', Processed_df.duplicated().sum())

Processed_df.drop_duplicates(inplace=True)
print("\nDuplicated values dropped succesfully")
print("*" * 20,"#"*30,"*"*20)

columns = list(Processed_df.columns)
categoric_columns = []
numeric_columns = []
for i in columns:
    if len(Processed_df[i].unique()) > 6:
        numeric_columns.append(i)
    else:
        categoric_columns.append(i)

categoric_columns = categoric_columns[:-1]

print("\n numeric_columns :",numeric_columns)

le = LabelEncoder()
Processed_df1 = Processed_df.copy()
Processed_df1[categoric_columns] = Processed_df1[categoric_columns].apply(le.fit_transform)
Processed_df1[['Churn']] = Processed_df1[['Churn']].apply(le.fit_transform)

Processed_df1[numeric_columns].describe()

palette = ['#FF6347', '#FFA07A']

l1 = list(Processed_df1['Churn'].value_counts())
pie_values = [l1[0] / sum(l1) * 100, l1[1] / sum(l1) * 100]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
axs[0].pie(pie_values, labels=['Not-Churn Customers', 'Churn Customers'], 
           autopct='%1.2f%%', explode=(0.1, 0), colors=palette,
           wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
axs[0].set_title('Churn and Not-Churn Customers %')

ax = sns.countplot(data=Processed_df1, x='Churn', hue='Churn', palette=palette, edgecolor='black', legend=False)
for i in ax.containers:
    ax.bar_label(i,)
ax.xaxis.set_major_locator(FixedLocator([0, 1]))
ax.set_xticklabels(['Not-Churn Customers', 'Churn Customers'])    
axs[1].set_title('Churn and Not-Churn Customers')

plt.tight_layout()
plt.show()

test_normality = lambda x: shapiro(x.fillna(0))[1] < 0.01
numerical_features = [f for f in Processed_df1.columns if Processed_df1.dtypes[f] != 'object']
normal = pd.DataFrame(Processed_df1[numerical_features])
normal = normal.apply(test_normality)
print("Are all numerical features normally distributed? ", not normal.any())

def dist_custom(dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols,figsize=(20,5))
    fig.suptitle(suptitle,y=1, size=25)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        sns.kdeplot(dataset[data], ax=axs[i], fill=True, alpha=0.8, linewidth=0, color='#FFA07A')
        axs[i].set_title(data + ', skewness is '+str(round(dataset[data].skew(axis = 0, skipna = True),2)))
dist_custom(dataset=Processed_df1, columns_list=numeric_columns, rows=1, cols=3, suptitle='Distibution for each numerical feature')
plt.tight_layout()

gender = Processed_df1.groupby('gender')['Churn'].value_counts(normalize=False).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
gender.set_title("Churn by Gender")
plt.show()
gender_pct = Processed_df1.groupby('gender')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
gender_pct.set_title("Percentage of churn by Gender")
plt.show()

dependents = Processed_df1.groupby('Dependents')['Churn'].value_counts(normalize=False).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
dependents.set_title("Churn by Dependents")
plt.show()
dependents_pct = Processed_df1.groupby('Dependents')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
dependents_pct.set_title("Percentage of churn by Dependents")
plt.show()

senior = Processed_df1.groupby('SeniorCitizen')['Churn'].value_counts(normalize=False).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
senior.set_title("Churn by SeniorCitizen")
plt.show()
senior_pct = Processed_df1.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
senior_pct.set_title("Percentage of churn by SeniorCitizen")
plt.show()

internet = Processed_df1.groupby('InternetService')['Churn'].value_counts(normalize=False).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
internet.set_title("Churn by InternetService")
plt.show()
internet_pct = Processed_df1.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
internet_pct.set_title("Percentage of churn by InternetService")
plt.show()

bill = Processed_df1.groupby('PaperlessBilling')['Churn'].value_counts(normalize=False).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
bill.set_title("Churn by PaperlessBilling")
plt.show()
bill_pct = Processed_df1.groupby('PaperlessBilling')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(10,5), color=['#FF6347', '#FFA07A'])
bill_pct.set_title("Percentage of churn by PaperlessBilling")
plt.show()

tenure_pct = Processed_df1.groupby('tenure')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(20,5), color=['#FF6347', '#FFA07A'])
tenure_pct.set_title("Percentage of churn by tenure")
plt.show()

Processed_df1['CLV'] = Processed_df1['MonthlyCharges'] * Processed_df1['tenure']
Customer_Lifetime_Value = px.scatter(Processed_df1, x='tenure', y='CLV', color='Churn', title='Customer Lifetime Value by Tenure')
Customer_Lifetime_Value.show()

bins= [0,500,1001,2001,4001,9000]
labels = ['Less than £500', '£500-£1000', '£1001-£2000', '£2001-£4000', 'More than £4000']
Processed_df1['TotalChargesGroups'] = pd.cut(Processed_df1['TotalCharges'], bins=bins, labels=labels, right=False)

# churn by TotalChargesGroups
mthcharges = Processed_df1.groupby('TotalChargesGroups')['Churn'].value_counts(normalize=False).unstack().plot(kind='bar', figsize=(20,5), color=['#FF6347', '#FFA07A'])
mthcharges.set_title("Churn by TotalChargesGroups")
plt.show()
mthcharges_pct = Processed_df1.groupby('TotalChargesGroups')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(20,5), color=['#FF6347', '#FFA07A'])
mthcharges_pct.set_title("Percentage of churn by TotalChargesGroups")
plt.show()

def boxplots_custom(dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(20,4))
    fig.suptitle(suptitle,y=1, size=25)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        sns.boxplot(data=dataset[data], orient='h', ax=axs[i], color='#FFA07A')
        axs[i].set_title(data + ', skewness is: '+str(round(dataset[data].skew(axis = 0, skipna = True),2)))
        
boxplots_custom(dataset=Processed_df1, columns_list=numeric_columns, rows=1, cols=3, suptitle='Boxplots for numerical features')
plt.tight_layout()

def IQR_method (Processed_df,n,features):
    outlier_list = []
    for column in features:
        Q1 = np.percentile(Processed_df[column], 25)
        Q3 = np.percentile(Processed_df[column],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_column = Processed_df[(Processed_df[column] < Q1 - outlier_step) | (Processed_df[column] > Q3 + outlier_step )].index
        outlier_list.extend(outlier_list_column)
        
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v > n )
    
    out1 = Processed_df[Processed_df[column] < Q1 - outlier_step]
    out2 = Processed_df[Processed_df[column] > Q3 + outlier_step]
    
    print('Total number of deleted outliers is:', out1.shape[0]+out2.shape[0])
    
    return multiple_outliers

Outliers_IQR = IQR_method(Processed_df,1,numeric_columns)
Processed_df_out = Processed_df.drop(Outliers_IQR, axis = 0).reset_index(drop=True)

print ('The count or number of fraudulent instances recorded in the dataset before dropping outliers: ', len(Processed_df[Processed_df['Churn'] == 1]))
print ('The count or number of fraudulent instances recorded in the dataset after dropping outliers: ', len(Processed_df_out[Processed_df_out['Churn'] == 1]))

fig = plt.subplots(nrows = 1,ncols = 2,figsize = (20,5))

plt.subplot(1,2,1)
ax = sns.kdeplot(Processed_df1.MonthlyCharges[(Processed_df1["Churn"] == 0)], color='#800000', fill= True, alpha=.7, linewidth=0)
ax = sns.kdeplot(Processed_df1.MonthlyCharges[(Processed_df1["Churn"] == 1)], color='#FF6347', fill= True, alpha=.7, linewidth=0)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of Monthly Charges by Churn')

plt.subplot(1,2,2)
ax = sns.kdeplot(Processed_df1.TotalCharges[(Processed_df1["Churn"] == 0)], color='#800000', fill= True, alpha=.7, linewidth=0)
ax = sns.kdeplot(Processed_df1.TotalCharges[(Processed_df1["Churn"] == 1)], color='#FF6347', fill= True, alpha=.7, linewidth=0)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Total Charges')
ax.set_title('Distribution of Total Charges by Churn')
plt.show()

palette2 = ['#FF0000','#FFFF00']
fig = plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
ax1 = sns.scatterplot(x = Processed_df['TotalCharges'], y = Processed_df['tenure'], hue = "Churn",
                    data = Processed_df, palette = palette2, edgecolor='grey', alpha = 0.8, s=9)
plt.title('TotalCharges vs tenure')

plt.subplot(1,3,2)
ax2 = sns.scatterplot(x = Processed_df['TotalCharges'], y = Processed_df['MonthlyCharges'], hue = "Churn",
                    data = Processed_df, palette =palette2, edgecolor='grey', alpha = 0.8, s=9)
plt.title('TotalCharges vs MonthlyCharges')

plt.subplot(1,3,3)
ax2 = sns.scatterplot(x = Processed_df['tenure'], y = Processed_df['MonthlyCharges'], hue = "Churn",
                    data = Processed_df, palette =palette2, edgecolor='grey', alpha = 0.8, s=9)
plt.title('MonthlyCharges vs tenure')

fig.suptitle('Numeric features', fontsize = 20)
plt.tight_layout()
plt.show()

payment = Processed_df1[Processed_df1['Churn'] == 1]['PaymentMethod'].value_counts()
pie_values = [payment[0] / sum(payment) * 100, payment[1] / sum(payment) * 100, payment[2] / sum(payment) * 100, payment[3] / sum(payment) * 100]

ax,fig = plt.subplots(figsize = (10,10))
plt.subplot()
plt.pie(pie_values,labels = ['Bank transfer','Credit card','Electronic check','Mailed check'],
        autopct='%1.2f%%',
        startangle = 90,
        explode = (0.1,0.1,0,0.1),
        colors = palette,
        wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
plt.title('Payment method (Churn)')
plt.show()

Processed_df2=Processed_df
Processed_df2[['Churn']] = Processed_df2[['Churn']].apply(le.fit_transform) 
X = Processed_df2.drop('Churn', axis=1)
y = Processed_df2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.25, random_state = 42)
Standard_Scaler = StandardScaler()
Standard_Scaler.fit_transform(X_train[numeric_columns])
Standard_Scaler.transform(X_test[numeric_columns])

print(categoric_columns)

transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), 
     ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
      'PhoneService', 'MultipleLines', 'InternetService', 
      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
      'TechSupport', 'StreamingTV', 'StreamingMovies', 
      'Contract', 'PaperlessBilling', 'PaymentMethod']))

transformed = transformer.fit_transform(X_train)
transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
transformed_df.index = X_train.index
X_train = pd.concat([X_train, transformed_df], axis=1)
X_train.drop(categoric_columns, axis=1, inplace=True)

transformed = transformer.transform(X_test)
transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
transformed_df.index = X_test.index
X_test = pd.concat([X_test, transformed_df], axis=1)
X_test.drop(categoric_columns, axis=1, inplace=True)
print(X_train.columns)

X_train.columns = ['Tenure', 'MonthlyCharges', 'TotalCharges',
       'gender_Female','gender_Male',
       'SeniorCitizen_0','SeniorCitizen_1',
       'Partner_No','Partner_Yes',
       'Dependents_No','Dependents_Yes',
       'PhoneService_No','PhoneService_Yes',
       'MultipleLines_No','MultipleLines_No phone service','MultipleLines_Yes',
       'InternetService_DSL','InternetService_Fiber','InternetService_No',
       'OnlineSecurity_No','OnlineSecurity_NoInternetService','OnlineSecurity_Yes', 
       'OnlineBackup_No','OnlineBackup_NoInternetService','OnlineBackup_Yes', 
       'DeviceProtection_No','DeviceProtection_NoInternetService','DeviceProtection_Yes', 
       'TechSupport_No', 'TechSupport_NoInternetService','TechSupport_Yes', 
       'StreamingTV_No', 'StreamingTV_NoInternetService','StreamingTV_Yes', 
       'StreamingMovies_No','StreamingMovies_NoInternetService','StreamingMovies_Yes',
       'Contract_Month-to-month','Contract_One year', 'Contract_Two year',
       'PaperlessBilling_No','PaperlessBilling_Yes',
       'PaymentMethod_BankTransfer','PaymentMethod_CreditCard','PaymentMethod_ElectronicCheck','PaymentMethod_MailedCheck']

X_test.columns = ['Tenure', 'MonthlyCharges', 'TotalCharges',
       'gender_Female','gender_Male',
       'SeniorCitizen_0','SeniorCitizen_1',
       'Partner_No','Partner_Yes',
       'Dependents_No','Dependents_Yes',
       'PhoneService_No','PhoneService_Yes',
       'MultipleLines_No','MultipleLines_No phone service','MultipleLines_Yes',
       'InternetService_DSL','InternetService_Fiber','InternetService_No',
       'OnlineSecurity_No','OnlineSecurity_NoInternetService','OnlineSecurity_Yes', 
       'OnlineBackup_No','OnlineBackup_NoInternetService','OnlineBackup_Yes', 
       'DeviceProtection_No','DeviceProtection_NoInternetService','DeviceProtection_Yes', 
       'TechSupport_No', 'TechSupport_NoInternetService','TechSupport_Yes', 
       'StreamingTV_No', 'StreamingTV_NoInternetService','StreamingTV_Yes', 
       'StreamingMovies_No','StreamingMovies_NoInternetService','StreamingMovies_Yes',
       'Contract_Month-to-month','Contract_One year', 'Contract_Two year',
       'PaperlessBilling_No','PaperlessBilling_Yes',
       'PaymentMethod_BankTransfer','PaymentMethod_CreditCard','PaymentMethod_ElectronicCheck','PaymentMethod_MailedCheck']
X_train.head()

palette = ["#FF5733", "#FFC300", "#6BFF33", "#337DFF"]
clf = RandomForestClassifier(max_depth=8, min_samples_leaf=3, min_samples_split=3, n_estimators=5000, random_state=13)
clf = clf.fit(X_train, y_train)
Important_Features = pd.Series(data=clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(17,13))
plt.title("Feature importance")
ax = sns.barplot(y=Important_Features.index, x=Important_Features.values, palette=palette, orient='h')

rf = RandomForestClassifier(random_state=13)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rf_Recall = recall_score(y_test, y_pred)
rf_Precision = precision_score(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred)
rf_accuracy = accuracy_score(y_test, y_pred)
rf_roc_auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

K_Fold = cross_val_score(rf, X_train, y_train, cv=5, scoring='recall', error_score="raise")
rf_cv_score = K_Fold.mean()
rf_cv_stdev = stdev(K_Fold)
print('Cross Validation Recall scores are: {}'.format(K_Fold))
print('Average Cross Validation Recall score: ', rf_cv_score)
print('Cross Validation Recall standard deviation: ', rf_cv_stdev)

ndf = [(rf_Recall, rf_Precision, rf_f1, rf_accuracy, rf_roc_auc, rf_cv_score, rf_cv_stdev)]

rf_score = pd.DataFrame(data = ndf, columns=['Recall','Precision','F1 Score', 'Accuracy', 'ROC-AUC Score', 'Avg CV Recall', 'Standard Deviation of CV Recall'])
rf_score.insert(0, 'Model', 'Random Forest')
rf_score

params = {'n_estimators': [130], 'max_depth': [14],  'min_samples_split': [3],'min_samples_leaf': [2],'random_state': [13]}
grid_rf = GridSearchCV(rf, param_grid=params, cv=5, scoring='recall').fit(X_train, y_train)

print('Best parameters:', grid_rf.best_params_)
print('Best score:', grid_rf.best_score_)

y_pred = grid_rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

grid_rf_Recall = recall_score(y_test, y_pred)
grid_rf_Precision = precision_score(y_test, y_pred)
grid_rf_f1 = f1_score(y_test, y_pred)
grid_rf_accuracy = accuracy_score(y_test, y_pred)
grid_roc_auc = roc_auc_score(y_test, y_pred)
print(cm)

K_Fold2 = cross_val_score(grid_rf, X_train, y_train, cv=5, scoring='recall')
grid_cv_score = K_Fold2.mean()
grid_cv_stdev = stdev(K_Fold2)
print('Cross Validation Recall scores are: {}'.format(K_Fold2))
print('Average Cross Validation Recall score: ', grid_cv_score)
print('Cross Validation Recall standard deviation: ', grid_cv_stdev)

ndf2 = [(grid_rf_Recall, grid_rf_Precision, grid_rf_f1, grid_rf_accuracy, grid_roc_auc, grid_cv_score, grid_cv_stdev)]

grid_score = pd.DataFrame(data = ndf2, columns=
                        ['Recall','Precision','F1 Score', 'Accuracy', 'ROC-AUC Score', 'Avg CV Recall', 'Standard Deviation of CV Recall'])
grid_score.insert(0, 'Model', 'Random Forest after tuning')
grid_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_logreg_pred = logreg.predict(X_test)
print('\n\nAccuracy of logistic regression classifier: {:.2f}'.format(logreg.score(X_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_logreg_pred)
print(confusion_matrix)

true_positive = 1156
false_positive = 260
false_negative = 123
true_negative = 222
correct_predictions = true_positive + true_negative
incorrect_predictions = false_positive + false_negative

print(f"\nLooking at the confusion matrix, we have {correct_predictions} ({true_positive} + {true_negative}) correctly classified customers and {incorrect_predictions} ({false_negative} + {false_positive}) incorrect predictions.\n\n")

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

logistic_accuracy = 81
random_forest_accuracy = 79
false_negatives = 222
false_positives = 123

print(Panel("Customer Churn Prediction", title="Model Comparison"))
print(f"Logistic Regression: {logistic_accuracy}% accuracy")
print(f"Random Forest: {random_forest_accuracy}% accuracy")

print(Panel(f"False Negatives: {false_negatives} customers not contacted", title="Confusion Matrix"))
print(Panel(f"False Positives: {false_positives} unnecessary incentives", title="Confusion Matrix"))
print(Panel("The telephone company should focus on reducing false negatives to retain customers.", title="Conclusion"))

X = Processed_df1[['tenure', 'MonthlyCharges']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid()
plt.show()

k = 3  
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
Processed_df1['Cluster'] = clusters

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Tenure (scaled)')
plt.ylabel('Monthly Charges (scaled)')
plt.title('Customer Segmentation by K-means Clustering')
plt.grid()
plt.colorbar(label='Cluster')
plt.show()

text = """
-->Color Coding and Cluster Interpretation

-->Blue (Cluster 1)
The concentration of blue color in the region of higher x and y axis values indicates that within this cluster, customers may exhibit longer tenure with higher monthly charges compared to other clusters.

-->Yellow (Cluster 2)
This cluster likely represents customers with:
‣ Higher values on the y-axis (Monthly Charges)
‣ Lower values on the x-axis (Tenure)
These customers might be newer but opting for higher-cost services, indicating potentially higher profitability early in their tenure.

-->Purple (Cluster 0)
This cluster corresponds to customers with:
‣ Lower values on both axes (lower Monthly Charges and shorter Tenure)
These customers may represent a segment with lower engagement or usage of services, potentially requiring different retention strategies.

-->Insights and Recommendations

‣ Targeting New, High-Value Customers: Focus on nurturing relationships with new customers (yellow cluster) to maximize long-term value.
‣ Retention of Lower Engagement Customers: Develop strategies to engage customers in the purple cluster more effectively to reduce churn and improve satisfaction.

-->Segmentation Benefits
This segmentation provides actionable insights into customer behavior, aiding in strategic decision-making to optimize business outcomes.
"""
print(Panel(text, title="Customer Segmentation Analysis", style="Yellow"))

