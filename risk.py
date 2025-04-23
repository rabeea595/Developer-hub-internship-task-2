import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Credit Risk Analysis Dashboard", layout="wide")

def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
        if 'SeriousDlqin2yrs' not in data.columns:
            st.error("Target column 'SeriousDlqin2yrs' not found in dataset.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    st.subheader("Preprocessing Data")
    st.write("Missing Values Before Preprocessing:")
    st.write(data.isnull().sum())
    data['MonthlyIncome'] = data['MonthlyIncome'].fillna(data['MonthlyIncome'].median())
    data['NumberOfDependents'] = data['NumberOfDependents'].fillna(data['NumberOfDependents'].mode()[0])
    st.write("Missing Values After Preprocessing:")
    st.write(data.isnull().sum())
    return data

def engineer_features(data):
    st.subheader("Feature Engineering")
    data['DebtToIncomeRatio'] = data['RevolvingUtilizationOfUnsecuredLines'] * data['MonthlyIncome']
    data['DebtToIncomeRatio'] = data['DebtToIncomeRatio'].replace([np.inf, -np.inf], 0)
    data['TotalPastDue'] = (data['NumberOfTime30-59DaysPastDueNotWorse'] +
                            data['NumberOfTime60-89DaysPastDueNotWorse'] +
                            data['NumberOfTimes90DaysLate'])
    data['IncomePerDependent'] = data['MonthlyIncome'] / (data['NumberOfDependents'] + 1)
    data['IncomePerDependent'] = data['IncomePerDependent'].replace([np.inf, -np.inf], 0)
    data['CreditUtilizationSeverity'] = data['RevolvingUtilizationOfUnsecuredLines'] * data['DebtRatio']
    st.write("New Features Added:", ['DebtToIncomeRatio', 'TotalPastDue', 'IncomePerDependent', 'CreditUtilizationSeverity'])
    st.write("Sample of Processed Data:")
    st.dataframe(data.head())
    csv = data.to_csv(index=False)
    st.download_button("Download Processed Data", csv, "processed_credit_data.csv", "text/csv")
    return data

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    st.write(f"Original Dataset Shape: {X.shape}")
    st.write(f"Resampled Dataset Shape: {X_resampled.shape}")
    return X_resampled, y_resampled

def plot_class_distribution(y, title):
    class_counts = pd.Series(y).value_counts()
    fig = px.bar(x=class_counts.index, y=class_counts.values, labels={'x': 'Class', 'y': 'Count'},
                 title=title, color=class_counts.index)
    st.plotly_chart(fig)

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True"),
                    x=['Non-Default', 'Default'], y=['Non-Default', 'Default'],
                    title=f"Confusion Matrix - {model_name}")
    st.plotly_chart(fig)

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}
    progress = st.progress(0)
    total_steps = len(models)
    for i, (name, model) in enumerate(models.items()):
        st.write(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        }
        st.write(f"\nClassification Report for {name}:")
        st.text(classification_report(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, name)
        progress.progress((i + 1) / total_steps)
    return results

def main():
    st.title("Credit Risk Analysis Dashboard")
    st.markdown("Upload the 'Give Me Some Credit' dataset to analyze creditworthiness and flag high-risk customers.")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is None:
            return
        st.subheader("Raw Data")
        st.dataframe(data.head())
        st.subheader("Class Distribution (Before SMOTE)")
        plot_class_distribution(data['SeriousDlqin2yrs'], "Original Class Distribution")
        preprocess_button = st.button("Preprocess Data", disabled='data' not in st.session_state and uploaded_file is None)
        if preprocess_button:
            data = preprocess_data(data)
            st.session_state['data'] = data
        feature_button = st.button("Engineer Features", disabled='data' not in st.session_state)
        if feature_button and 'data' in st.session_state:
            data = engineer_features(st.session_state['data'])
            st.session_state['data'] = data
        train_button = st.button("Train and Evaluate Models", disabled='data' not in st.session_state)
        if train_button and 'data' in st.session_state:
            data = st.session_state['data']
            X = data.drop('SeriousDlqin2yrs', axis=1)
            y = data['SeriousDlqin2yrs']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            st.subheader("Applying SMOTE")
            X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
            plot_class_distribution(y_train_resampled, "Class Distribution After SMOTE")
            st.subheader("Model Performance")
            results = train_and_evaluate_models(X_train_resampled, y_train_resampled, X_test, y_test)
            st.write("Model Performance Summary:")
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df)
            fig = go.Figure()
            for metric in results_df.columns:
                fig.add_trace(go.Bar(x=results_df.index, y=results_df[metric], name=metric))
            fig.update_layout(title="Model Performance Comparison", barmode='group')
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
