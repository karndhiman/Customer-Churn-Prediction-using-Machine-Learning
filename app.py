import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("   Customer Churn Predictor   ")


# We will use this to load our trained models and 
# use cache_data to load faster and avoid refreshing everytime.
@st.cache_data(show_spinner=False)
def load_artifact(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
    
model = load_artifact("best_model.pkl")
encoders = load_artifact("encoder.pkl")
scaler = load_artifact("scaler.pkl")

if model is None:
    st.error("Missing model file 'best_model.pkl', put it in the same folder as app.py and restart.")
    st.stop()


# created a function to show recommendations to improve retention
def get_recommends_row(row):
    recs = []
    contract = row.get('Contract')
    monthly = row.get('MonthlyCharges', 0)
    tech = row.get('TechSupport')
    onlinesec = row.get('OnlineSecurity')

    if contract is not None and "Month" in str(contract):
        recs.append("Offer discounted yearly contract.")
    if monthly and float(monthly) > 80:
        recs.append("Provide loyalty discount or bundle offers.")
    if tech is not None and str(tech).lower() in ["no", "0", "false"]:
        recs.append("Offer free 3-month trial of Tech Support.")
    if onlinesec is not None and str(onlinesec).lower() in ["no", "0", "false"]:
        recs.append("Upsell with security bundle to increase retention.")
    if not recs:
        recs.append("Customers seems stable. Focus on engagement activities.")
    return recs

def apply_encoders(df_in):
    if encoders is None:
        return df_in
    df = df_in.copy()
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(str)
                df[col] = le.transform(df[col])
            except Exception:
                df[col] = -1
    return df

def apply_scaler(df_in, numeric_cols = None):
    if scaler is None or numeric_cols is None:
        return df_in
    df = df_in.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


# Input Form
st.header("Enter Customer Details")

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols =  list(encoders.keys()) if encoders else []

form = st.form("customer_form")
user_input = {}

if encoders:
    st.markdown("### Categorical Fields")
    for col in categorical_cols:
        classes = list(encoders[col].classes_)
        val = form.selectbox(f"{col}", options=classes, key = f"{col}_sel")
        user_input[col] = val

st.markdown("### Numerical Fields")
for col in numerical_cols:
    default = 0.0
    val = form.number_input(col, min_value = 0.0, max_value = 1e6, value=default, step=1.0)
    user_input[col] = val

submit = form.form_submit_button("Predict Churn")

if submit:
    input_df = pd.DataFrame([user_input])
    st.write("Raw Input")
    st.dataframe(input_df.T)

    processed = apply_encoders(input_df)
    processed = apply_scaler(processed, numeric_cols = numerical_cols)

    if hasattr(model, "feature_names_in_"):
        cols_needed = list(model.feature_names_in_)
    else:
        cols_needed = list(processed.columns)
    for c in cols_needed:
        if c not in processed.columns:
            processed[c] = 0
    processed = processed[cols_needed]

    prob = model.predict_proba(processed)[:, 1][0]
    pred = model.predict(processed)[0]

    st.metric("Churn Probability: ", f"{prob:.2f}")
    st.write("Prediction: ", "Churn" if pred == 1 else "No Churn")
    st.divider()
    
    recs = get_recommends_row(user_input)
    st.markdown("### Recommended Retention Actions:")
    for r in recs:
        st.write("- " + r)
    
    st.divider()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        names = getattr(model, "feature_names_in_", processed.columns)
        fi = pd.Series(imp, index= names).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x = fi.head(10).values, y = fi.head(10).index, ax=ax)
        ax.set_title("Feature Importance (Top 10)")
        st.pyplot(fig)
