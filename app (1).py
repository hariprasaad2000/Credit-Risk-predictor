import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
path = kagglehub.dataset_download("benjaminmcgregor/german-credit-data-set-with-credit-risk")
df = pd.read_csv(os.path.join(path, 'german_credit_data_updated.csv'))

# Clean
df = df.drop(columns=['Unnamed: 0'])
df.loc[df['Checking account'].isna(), 'Checking account'] = 'Unknown'
df = df.dropna(subset=['Saving accounts'])
df['Credit Risk'] = df['Credit Risk'].apply(lambda x: 'High' if x == 1 else 'Low')
df['Credit Risk'] = df['Credit Risk'].map({'Low': 0, 'High': 1})

# Train
X = df[['Credit amount']]
y = df['Credit Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# UI
st.title('Credit Risk Predictor')
st.write('Enter your loan details below to find out your credit risk.')

credit_amount = st.slider(
    'Credit Amount (DM)',
    min_value=250,
    max_value=20000,
    value=5000,
    step=250
)

if st.button('Predict My Risk'):
    input_data = pd.DataFrame([[credit_amount]], columns=['Credit amount'])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f'High Risk — {probability:.0%} probability of default')
        st.write('Based on your credit amount, the model predicts you are a high risk borrower.')
    else:
        st.success(f'Low Risk — {probability:.0%} probability of default')
        st.write('Based on your credit amount, the model predicts you are a low risk borrower.')
