import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the dataset to extract unique team names
df = pd.read_csv("data/processed/df_final.csv")
unique_teams = df['batting_team'].unique()

# Load the pre-trained model pipeline
model_path = "models/pipe.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title and Description
st.title("IPL Score Prediction")
st.write("Use the sidebar to enter the match details to predict the score.")

# Sidebar User Input Fields
st.sidebar.header("Input Features")

batting_team = st.sidebar.selectbox('Select Batting Team:', unique_teams)
bowling_team = st.sidebar.selectbox('Select Bowling Team:', unique_teams)
over = st.sidebar.number_input('Enter Over Number (1-20):', min_value=1, max_value=20, step=1)
ball = st.sidebar.number_input('Enter Ball Number (1-6):', min_value=1, max_value=6, step=1)
current_runs = st.sidebar.number_input('Enter Current Runs:', min_value=0, step=1)
rolling_back_30balls_runs = st.sidebar.number_input('Enter Runs in Last 30 Balls:', min_value=0, step=1)
rolling_back_30balls_wkts = st.sidebar.number_input('Enter Wickets in Last 30 Balls:', min_value=0, step=1)

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'over': [over],
    'ball': [ball],
    'current_runs': [current_runs],
    'rolling_back_30balls_runs': [rolling_back_30balls_runs],
    'rolling_back_30balls_wkts': [rolling_back_30balls_wkts]
})

# When 'Predict' is clicked, make a prediction and display SHAP plots
if st.sidebar.button('Predict'):
    try:
        # Transform the input data using the pipeline
        transformed_input = model.named_steps['trf1'].transform(input_data)
        
        # Make prediction
        prediction = model.named_steps['model'].predict(transformed_input)
        st.write(f"Predicted Score: {prediction[0]:.2f}")

       # SHAP explainability
        """ explainer = shap.Explainer(model.named_steps['model'])
        shap_values = explainer(transformed_input)
        print(explainer)
        print(shap_values)

        # Display SHAP summary plot
        st.subheader("SHAP Summary Plot")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, transformed_input, feature_names=model.named_steps['trf1'].get_feature_names_out(), show=False)
        st.pyplot(fig)  # Display the summary plot"""
             
        
    except Exception as e:
        st.error(f"Error generating SHAP plots: {e}")
