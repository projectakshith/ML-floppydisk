# Unit 2: Telecom churn classification
# Extracted from Appendix A of Team_FloppyDisk_Extended_Final_Report.docx.
# For the original Colab notebook where available, see the matching .ipynb file.

import io, zipfile, requests, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             ConfusionMatrixDisplay)
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
print('Libraries loaded')

# Download the public Kaggle dataset directly. No Kaggle API token is required.
url = 'https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn'
response = requests.get(url, timeout=60)
response.raise_for_status()
with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    csv_name = next(name for name in archive.namelist() if name.endswith('.csv'))
    with archive.open(csv_name) as csv_file:
        df = pd.read_csv(csv_file)
print('Dataset:', csv_name)
print('Shape:', df.shape)
df.head()

print(df['Churn'].value_counts())
print('Churn rate:', f"{df['Churn'].eq('Yes').mean():.1%}")
print('Missing values before cleaning:', int(df.isna().sum().sum()))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print('Missing TotalCharges after numeric conversion:', int(df['TotalCharges'].isna().sum()))

service_features = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies'
]
feature_cols = ['tenure', 'Contract', 'MonthlyCharges', 'PaymentMethod'] + service_features
X = df[feature_cols].copy()
y = df['Churn'].map({'No': 0, 'Yes': 1})
numeric_features = ['tenure', 'MonthlyCharges']
categorical_features = [c for c in feature_cols if c not in numeric_features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print('Training rows:', len(X_train), '| Test rows:', len(X_test))
print('Features:', feature_cols)

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
    'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                   class_weight='balanced', random_state=RANDOM_STATE)
}

results = []
trained_models = {}
for name, estimator in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, prob)
    })
    trained_models[name] = pipeline
results_df = pd.DataFrame(results).set_index('Model').round(3)
results_df

best_name = results_df['ROC-AUC'].idxmax()
best_model = trained_models[best_name]
best_pred = best_model.predict(X_test)
print('Selected model:', best_name)
print(classification_report(y_test, best_pred, target_names=['No churn', 'Churn']))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, display_labels=['No churn', 'Churn'], cmap='Blues'
)
plt.title(f'{best_name} confusion matrix')
plt.show()

sample = X_test.iloc[[0]].copy()
risk = best_model.predict_proba(sample)[0, 1]
prediction = 'Yes' if risk >= 0.5 else 'No'
display(sample)
print(f'Predicted Churn: {prediction}')
print(f'Estimated churn probability: {risk:.1%}')

import joblib
joblib.dump(best_model, '/content/telco_churn_model.joblib')
results_df.to_csv('/content/telco_churn_model_results.csv')
print('Saved /content/telco_churn_model.joblib')
print('Saved /content/telco_churn_model_results.csv')
