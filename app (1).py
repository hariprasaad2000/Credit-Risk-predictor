import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

@st.cache_data
def load_and_train():
    path = kagglehub.dataset_download("benjaminmcgregor/german-credit-data-set-with-credit-risk")
    df = pd.read_csv(os.path.join(path, 'german_credit_data_updated.csv'))

    # Clean
    df = df.drop(columns=['Unnamed: 0'])
    df.loc[df['Checking account'].isna(), 'Checking account'] = 'Unknown'
    df = df.dropna(subset=['Saving accounts'])
    df['Credit Risk'] = df['Credit Risk'].apply(lambda x: 'High' if x == 1 else 'Low')
    df['Credit Risk'] = df['Credit Risk'].map({'Low': 0, 'High': 1})

    # Train
    X = df[['Credit amount', 'Age', 'Duration']]
    y = df['Credit Risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

# ============================================================
# GAUGE CHART FUNCTION
# ============================================================

def show_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'red' if probability > 0.5 else 'green'},
            'steps': [
                {'range': [0, 40], 'color': '#d5f5e3'},
                {'range': [40, 60], 'color': '#fdebd0'},
                {'range': [60, 100], 'color': '#fadbd8'}
            ],
        },
        title={'text': 'Probability of Default'}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# LOAD MODEL
# ============================================================

model, accuracy = load_and_train()

# ============================================================
# UI
# ============================================================

st.title('Credit Risk Predictor')
st.write('A machine learning tool that predicts whether a borrower is a credit risk based on their loan details.')
st.write(f'Model accuracy: **{accuracy:.0%}**')

st.divider()

st.subheader('Enter your loan details')

credit_amount = st.number_input(
    'Credit Amount (DM)',
    min_value=250,
    max_value=20000,
    value=5000,
    step=250,
    help='How much money are you borrowing?'
)

age = st.number_input(
    'Age',
    min_value=18,
    max_value=75,
    value=30,
    step=1,
    help='Your current age'
)

duration = st.number_input(
    'Loan Duration (months)',
    min_value=4,
    max_value=72,
    value=24,
    step=1,
    help='How many months to repay the loan?'
)

st.divider()

if st.button('Predict My Risk', use_container_width=True):

    input_data = pd.DataFrame(
        [[credit_amount, age, duration]],
        columns=['Credit amount', 'Age', 'Duration']
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    # Gauge chart
    show_gauge(probability)

    # Result
    if prediction == 1:
        st.error(f'High Risk — {probability:.0%} probability of default')
        st.write('Based on your details, the model predicts you are a **high risk** borrower.')
        st.write('This means a bank would likely be cautious about approving your loan.')
    else:
        st.success(f'Low Risk — {probability:.0%} probability of default')
        st.write('Based on your details, the model predicts you are a **low risk** borrower.')
        st.write('This means a bank would likely be comfortable approving your loan.')

    st.divider()

    # Show what was entered
    st.subheader('Your details')
    col1, col2, col3 = st.columns(3)
    col1.metric('Credit Amount', f'{credit_amount} DM')
    col2.metric('Age', f'{age} years')
    col3.metric('Duration', f'{duration} months')

st.divider()
st.caption('Built by Hari Prasaad — Year 1 CS Student | Trained on German Credit Dataset | Model: Logistic Regression')



