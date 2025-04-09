# Bias Detection Pipeline (Adult Income Dataset) with Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from fairlearn.metrics import (MetricFrame, selection_rate, demographic_parity_difference,
                               equalized_odds_difference, true_positive_rate, false_positive_rate)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from io import StringIO
import os

st.set_page_config(layout="wide")
st.title("AI Bias Detection Dashboard")

# Load Sample Dataset (Adult Income Dataset)
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]

sample_data = """
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
53, Private, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K
28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, >50K
37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, >50K
49, Private, 160187, 9th, 5, Married-spouse-absent, Other-service, Not-in-family, Black, Female, 0, 0, 16, Jamaica, <=50K
"""
data = pd.read_csv(StringIO(sample_data), names=columns, na_values=" ?", skipinitialspace=True)
data.dropna(inplace=True)

# Preprocessing
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop("income", axis=1)
y = data["income"]

# Sidebar Config
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ['logistic', 'random_forest', 'svm'])
sensitive_features = st.sidebar.multiselect("Select Sensitive Features", options=X.columns.tolist(), default=['sex', 'race'])

if st.sidebar.button("Run Bias Analysis"):
    # Split & Scale
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Classifier
    if model_choice == 'logistic':
        clf = LogisticRegression(max_iter=1000)
    elif model_choice == 'random_forest':
        clf = RandomForestClassifier()
    elif model_choice == 'svm':
        clf = SVC()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    st.subheader("Model Performance")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

    for sensitive_feature in sensitive_features:
        st.markdown(f"### Fairness Analysis: {sensitive_feature}")
        sensitive_series = X_test_df[sensitive_feature]

        mf = MetricFrame(metrics={
            'accuracy': lambda y_true, y_pred: np.mean(y_true == y_pred),
            'selection_rate': selection_rate,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate
        },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_series
        )

        st.write("Fairness Metrics by Group:")
        st.dataframe(mf.by_group)

        dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_series)
        eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_series)
        st.write("Demographic Parity Difference:", dpd)
        st.write("Equalized Odds Difference:", eod)

        # Mitigation
        mitigator = ExponentiatedGradient(
            estimator=LogisticRegression(solver="liblinear"),
            constraints=DemographicParity(),
            sample_weight_name="sample_weight"
        )
        mitigator.fit(X_train, y_train, sensitive_features=X_train_df[sensitive_feature])
        y_pred_mitigated = mitigator.predict(X_test)

        st.write("Mitigated Classification Report:")
        st.text(classification_report(y_test, y_pred_mitigated))

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(x=y_pred, hue=sensitive_series, multiple="dodge", bins=2, ax=ax)
        ax.set_title(f"Predicted Outcomes by Group: {sensitive_feature}")
        ax.set_xlabel("Prediction")
        st.pyplot(fig)

        # CSV Logs
        metrics_log = mf.by_group.copy()
        metrics_log["model"] = model_choice
        metrics_log["sensitive_attribute"] = sensitive_feature
        metrics_filename = f"bias_metrics_log_{sensitive_feature}.csv"
        metrics_log.to_csv(metrics_filename)
        st.success(f"Metrics logged to {metrics_filename}")
