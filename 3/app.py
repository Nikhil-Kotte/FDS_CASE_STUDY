"""Titanic survival predictor (Case Study 3).

Run from the repository folder:  streamlit run 3/app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "titanic_logreg.joblib"
PORTS = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}
TITLES = {"male": ["Mr", "Master", "Rare"], "female": ["Mrs", "Miss", "Rare"]}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Titanic Survival Predictor")
st.write("Enter the passenger details and click Predict.")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.radio("Sex", ["male", "female"], horizontal=True)
    title = st.selectbox("Title", TITLES[sex])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=28.0)
    port = st.selectbox("Port of embarkation", list(PORTS))
with col2:
    sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=14.45)
    has_cabin = st.checkbox("Cabin number known")

if st.button("Predict"):
    family_size = sibsp + parch + 1
    passenger = pd.DataFrame([{
        "Pclass": pclass, "Age": age, "Fare": fare, "Embarked": PORTS[port], "Title": title,
        "FamilySize": family_size, "IsAlone": int(family_size == 1), "HasCabin": int(has_cabin),
    }])
    prob = model.predict_proba(passenger)[0, 1]
    st.write(f"Survival probability: **{prob:.1%}**")
    if prob >= 0.5:
        st.success("Predicted: Survived")
    else:
        st.error("Predicted: Did not survive")

st.caption("Logistic regression trained on 80% of the Titanic data (tuned with 5-fold cross-validation). "
           "Made for a college case study.")
