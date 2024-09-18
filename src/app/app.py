from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the dataset to extract unique team names
df = pd.read_csv("data/df_final.csv")
unique_teams = df['batting_team'].unique()

# Load the pre-trained model pipeline
model_path = "models/pipe.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', teams=unique_teams)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the form
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    over = int(request.form['over'])
    ball = int(request.form['ball'])
    current_runs = int(request.form['current_runs'])
    rolling_back_30balls_runs = int(request.form['rolling_back_30balls_runs'])
    rolling_back_30balls_wkts = int(request.form['rolling_back_30balls_wkts'])

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'over': [over],
        'ball': [ball],
        'current_runs': [current_runs],
        'rolling_back_30balls_runs': [rolling_back_30balls_runs],
        'rolling_back_30balls_wkts': [rolling_back_30balls_wkts]
    })

    try:
        # Transform the input data using the pipeline
        transformed_input = model.named_steps['trf1'].transform(input_data)

        # Make prediction
        prediction = model.named_steps['model'].predict(transformed_input)[0]

        return render_template('index.html', teams=unique_teams, prediction_text=f"Predicted Score: {prediction:.2f}")
    except Exception as e:
        return render_template('index.html', teams=unique_teams, prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
