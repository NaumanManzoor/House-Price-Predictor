from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor dictionary from model.pkl
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)  # data is expected to be a dict with keys 'model' and 'preprocessor'

model = data['model']
preprocessor = data['preprocessor']

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from form and convert to correct types
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = float(request.form['housing_median_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])
        ocean_proximity = request.form['ocean_proximity']

        # Prepare a single-row DataFrame to pass to preprocessor
        input_df = pd.DataFrame([{
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity': ocean_proximity
        }])

        # Preprocess the input features using the saved preprocessor
        X_processed = preprocessor.transform(input_df)

        # Predict using the loaded model
        prediction = model.predict(X_processed)[0]

        # Format the prediction output nicely
        prediction_text = f'Estimated Median House Value: ${prediction:,.2f}'

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
