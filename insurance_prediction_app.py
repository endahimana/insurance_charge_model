## We need to import necessary libraries
import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

## Setting the title of the web application
st.title("Insurance Charges Prediction App")

## Loading the pre-trained machine learning model
model = load_model('insurance_charges_model')

## Loading the dataset for prediction input
data = pd.read_csv('insurance_data.csv')

## Input section for user to select for prediction
st.sidebar.header("Input Features for Prediction")

## User input for selecting an index from the dataset
age = st.sidebar.slider('Age', int(data.age.min()), int(data.age.max()), int(data.age.mean()))

bmi = st.sidebar.slider('BMI', float(data.bmi.min()), float(data.bmi.max()), float(data.bmi.mean()))

children = st.sidebar.slider('Number of Children', int(data.children.min()), int(data.children.max()), int(data.children.mean()))

bloodpressure = st.sidebar.slider('Blood Pressure', float(data.bloodpressure.min()), float(data.bloodpressure.max()), float(data.bloodpressure.mean()))

smoker = st.sidebar.selectbox('Smoker', data.smoker.unique())

region = st.sidebar.selectbox('Region', data.region.unique())

diabetic = st.sidebar.selectbox('Diabetic', data.diabetic.unique())

gender = st.sidebar.selectbox('Gender', data.gender.unique())

## Creating a DataFrame for the input features
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'bloodpressure': [bloodpressure],
    'smoker': [smoker],
    'region': [region],
    'diabetic': [diabetic],
    'gender': [gender],
})

## Display prediction button
if st.button("Predict Insurance Charges"):
    ## Making prediction using the loaded model
    prediction = predict_model(model, data=input_data)
    predicted_charge = prediction['prediction_label'][0]
    
    ## Displaying the prediction result
    st.subheader("Predicted Insurance Charge:")
    st.write(f"${predicted_charge:,.2f}")

