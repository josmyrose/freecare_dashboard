from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
import os

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/cleaned_data.csv')
# Initialize model as None
model = None

def load_model():
    """Load the trained model"""
    global model
    # Load model and scaler (do this once when app starts)
    MODEL_PATH ='models/policy_model1.pkl'
    
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = joblib.load('models/policy_model2.joblib')
        print("Model loaded successfully!")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
# Load the diarrhea prediction model
try:
    diarrhea_model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))
    print("Diarrhea model loaded successfully")
except Exception as e:
    print(f"Error loading diarrhea model: {str(e)}")
    diarrhea_model = None

@app.route('/')
def overview():
    return render_template('overview.html')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/data')
def data_explorer():
    return render_template('data.html',df=df)
@app.route('/healthcarevisualisation')
def healthcare_explorer():
    return render_template('healthcarevisualisation.html',df=df)
@app.route('/simulation')
def simulation_explorer():
    return render_template('simulation.html',df=df)
@app.route('/diaherreapred')
def diarrhea_dashboard():
    print("Accessed /diaherreapred route")
    return render_template('diaherreapred.html')
@app.route('/predict_diarrhea', methods=['POST'])
def predict_diarrhea():
    global model
    if diarrhea_model is None:
        return jsonify({'error': 'Diarrhea model not available'}), 503
    
    try:
        # Get data from POST request
        data = request.get_json()
        
        
        # Validate all required fields are present
        required_fields = [
            'any_healthcare_visit',
            'Treatment_encoded',
            'Diarrhea_last_week_',
            'Three_Stool_last_week'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Prepare features
        features = [
            int(data['any_healthcare_visit']),
            int(data['Treatment_encoded']),
            float(data['Diarrhea_last_week_']),
            float(data['Three_Stool_last_week'])
        ]
        policy_features = [
                "any_healthcare_visit",
                "Treatment_encoded",
                "Diarrhea_last_week_",
                "Three_Stool_last_week"
]
        # Suppose your model expects 5 features (adjust as needed)
     
        features_df = pd.DataFrame(features, columns=policy_features)

        # Make prediction
        #features_array = np.array(features).reshape(1, -1)
        prediction = diarrhea_model.predict(features_df)
        probability = diarrhea_model.predict_proba(features_df)[0][1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        # Get data from request
        data = request.json
        
        # Prepare features in correct order
        features = [
            data['health_worker_visits_check_'],
            data['free_care_check_'],
            data['Pharmacy_last_week'],
            data['Traditional_last_week'],
            data['Mosquito_net_used_'],
            data['ORT_ingr_correct'],
            data['Treatment_encoded']
        ]
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)[0][1]  # Probability of positive class
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400
# Load the model once at startup
try:
    load_model()
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
@app.route('/treatment-effect')
def treatment_effect():
    """Render the treatment effect analysis page"""
    return render_template('treatment_effect.html') 

if __name__ == '__main__':
    app.run(debug=True)
