from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model and data
model = pickle.load(open('medical_model.pkl', 'rb'))
data = pd.read_csv('dataset.csv')
label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])

# Create symptom map
symptom_map = {symptom: symptom.replace('_', ' ').title() for symptom in data.columns[:-1]}
reverse_symptom_map = {v: k for k, v in symptom_map.items()}

@app.route('/')
def index():
    symptoms = list(symptom_map.values())
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    input_data = np.zeros(len(symptom_map))  # Initialize input data with zeros

    for symptom in selected_symptoms:
        if symptom in reverse_symptom_map:
            index = list(symptom_map.keys()).index(reverse_symptom_map[symptom])
            input_data[index] = 1  # Set the corresponding symptom index to 1
        else:
            print(f"Warning: Symptom '{symptom}' not found in reverse_symptom_map.")

    input_df = pd.DataFrame([input_data], columns=symptom_map.keys())

    try:
        prediction = model.predict(input_df)
        disease = label_encoder.inverse_transform(prediction)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', symptoms=list(symptom_map.values()), error='Prediction failed')

    return render_template('index.html', symptoms=list(symptom_map.values()), disease=disease[0])

@app.route('/data_insights')
def data_insights():
    symptom_counts = data.drop(columns='prognosis').sum().sort_values(ascending=False)
    symptom_names = [symptom_map[symptom] for symptom in symptom_counts.index]

    df_symptoms = pd.DataFrame({
        'Symptom': symptom_names,
        'Count': symptom_counts.values
    })

    insights = df_symptoms.to_dict(orient='records')
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True)
