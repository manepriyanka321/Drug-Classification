import streamlit as st
import pandas as pd
import joblib 

# Set the webpage title
st.set_page_config(page_title="Drug Classification")
st.title("Drug Classification Project")
st.subheader("~by Priyanka Mane")

# Load the model
model = joblib.load("notebook/model.joblib")
pre = joblib.load("notebook/pre.joblib")

# Function to predict results
def predict_results(model, age, sex, bp, cholesterol, na_to_k):

    d = {
        "Age" : [age],
        "Sex" : [sex],
        "BP" : [bp],
        "Cholesterol" : [cholesterol],
        "Na_to_K" : [na_to_k]
    }

    xnew = pd.DataFrame(d)
    xnew = pre.transform(xnew)

    preds = model.predict(xnew)
    probs = model.predict_proba(xnew)

    classes = model.classes_

    prob_d = {}
    for c, p in zip(classes, probs.flatten()):
        prob_d[c] = float(p)

    return preds[0], prob_d

# Take input from user
age = st.number_input("Age", min_value=1, step=1)
sex = st.text_input("Sex", max_chars=1)
bp = st.text_input("BP", max_chars=4 )
cholesterol = st.text_input("Cholesterol",max_chars=4 )
na_to_k = st.number_input("Na_to_K", min_value=0.001, step= 0.001)

submit = st.button("Submit", type="primary")

if submit:
    pred, probs = predict_results(model, age,sex,bp,cholesterol,na_to_k)
    st.subheader(f"Prediction : {pred}")

    for c, p in probs.items():
        st.subheader(f"{c} : {p:.4f}")
        st.progress(p)