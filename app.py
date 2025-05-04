from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import pickle
import os
import dashboard



app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/output.csv')

# Load the diarrhea prediction model
try:
    diarrhea_model = pickle.load(open('models/model.pkl', 'rb'))
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
    return render_template('data.html', df=df)

@app.route('/healthcarevisualisation')
def healthcare_explorer():
    return render_template('healthcarevisualisation.html', df=df)
@app.route('/keyfindings')
def keyfindings():
    return render_template('keyfindings.html', df=df)
@app.route('/healthcare')
def healthcare():
    return render_template('healthcarestatistics.html', df=df)
# Corrected route name
@app.route('/diarrheapred')
def diarrhea_dashboard():
    print("Accessed /diarheapred route")
    return render_template('diarrheapred.html')

# Handle both GET and POST for the prediction route
@app.route('/predict_diarrhea', methods=['GET', 'POST'])
def predict_diarrhea():
    if diarrhea_model is None:
        return jsonify({'error': 'Diarrhea model not available'}), 503
    
    if request.method == 'GET':
        return render_template('diarrheapred.html', prediction_text="Please submit the form to get a prediction")
    
    try:
        # Get data from POST request
        data = request.form if request.form else request.get_json()
        
        # Validate all required fields are present
        required_fields = [
            'any_healthcare_visit',
            'Treatment_encoded',
            'Diarrhea_last_week_',  # Fixed typo in field name
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
        
        features_df = pd.DataFrame([features], columns=required_fields)
        
        # Make prediction
        prediction = diarrhea_model.predict(features_df)
        probability = diarrhea_model.predict_proba(features_df)[0][1]
        
        result = "Person had diarrhea." if prediction[0] == 1 else "Person did not have diarrhea."
        
        return render_template('diarrheapred.html', 
                             prediction_text=result,
                             probability=f"{probability:.2f}")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
