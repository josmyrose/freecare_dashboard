from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/cleaned_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/data')
def data_explorer():
    return render_template('data.html',df=df)
@app.route('/treatment-effect')
def treatment_effect():
    """Render the treatment effect analysis page"""
    return render_template('treatment_effect.html')

@app.route('/api/treatment-effect/predict', methods=['POST'])
def predict_treatment_effect():
    """API endpoint for treatment effect prediction"""
    try:
        # Get parameters from request
        data = request.json
        outcome = data.get('outcome', 'Diarrhea_Had')
        model_type = data.get('model_type', 'random_forest')
        
        # Load your dataset (adjust path as needed)
        df = pd.read_csv('data/cleaned_data.csv')
        
        # Prepare data
        treatment = 'free_care_check_'
        covariates = ['gender', 'MUAC_Unadjusted', 'dist1', 'child_weight_1_']
        
        X = df[covariates]
        y = df[outcome]
        T = df[treatment]
        
        # Train model (or load pre-trained)
        model_path = f'models/{model_type}_{outcome}.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=5, random_state=42)
            
            model.fit(X, y)
            joblib.dump(model, model_path)
        
        # Calculate treatment effects
        treated = df[df[treatment] == 1]
        control = df[df[treatment] == 0]
        
        pred_treated = model.predict(treated[covariates])
        pred_control = model.predict(control[covariates])
        
        ate = pred_treated.mean() - pred_control.mean()
        
        return jsonify({
            'success': True,
            'ate': float(ate),
            'treated_mean': float(pred_treated.mean()),
            'control_mean': float(pred_control.mean()),
            'outcome': outcome,
            'model_type': model_type
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/treatment-effect/feature-importance', methods=['POST'])
def get_feature_importance():
    """API endpoint for feature importance"""
    try:
        data = request.json
        outcome = data.get('outcome', 'Diarrhea_Had')
        model_type = data.get('model_type', 'random_forest')
        
        model_path = f'models/{model_type}_{outcome}.pkl'
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': 'Model not trained yet'})
        
        model = joblib.load(model_path)
        
        if model_type == 'random_forest':
            importances = model.feature_importances_
        else:
            importances = model.feature_importances_
        
        covariates = ['gender', 'MUAC_Unadjusted', 'dist1', 'child_weight_1_']
        
        return jsonify({
            'success': True,
            'features': covariates,
            'importance': [float(x) for x in importances]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
@app.route('/api/treatment_counts')
def treatment_counts():
    treatment_cols = ['Treatment1', 'Treatment2', 'Treatment3', 'Treatment4']
    counts = df[treatment_cols].sum().reset_index()
    counts.columns = ['Treatment', 'Count']
    
    fig = px.bar(counts, x='Treatment', y='Count', 
                 title='Distribution of Treatments',
                 color='Treatment')
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

@app.route('/api/diarrhea_by_gender')
def diarrhea_by_gender():
    gender_diarrhea = df.groupby(['gender', 'Diarrhea_Had']).size().unstack().reset_index()
    gender_diarrhea.columns = ['Gender', 'No_Diarrhea', 'Diarrhea']
    gender_diarrhea['Gender'] = gender_diarrhea['Gender'].map({0: 'Male', 1: 'Female'})
    
    fig = px.bar(gender_diarrhea, x='Gender', y=['No_Diarrhea', 'Diarrhea'],
                 title='Diarrhea Cases by Gender',
                 labels={'value': 'Count', 'variable': 'Diarrhea Status'},
                 barmode='group')
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

@app.route('/api/diarrhea_by_age')
def diarrhea_by_age():
    # Create age groups based on weight (simplified for this example)
    df['age_group'] = pd.cut(df['weight'], 
                            bins=[0, 10, 15, 20, 25, 30],
                            labels=['0-10kg', '10-15kg', '15-20kg', '20-25kg', '25-30kg'])
    
    age_diarrhea = df.groupby(['age_group', 'Diarrhea_Had']).size().unstack().reset_index()
    age_diarrhea.columns = ['Age_Group', 'No_Diarrhea', 'Diarrhea']
    
    fig = px.bar(age_diarrhea, x='Age_Group', y=['No_Diarrhea', 'Diarrhea'],
                 title='Diarrhea Cases by Weight (Age Proxy)',
                 labels={'value': 'Count', 'variable': 'Diarrhea Status'},
                 barmode='group')
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

@app.route('/api/treatment_effectiveness')
def treatment_effectiveness():
    treatment_cols = ['Treatment1', 'Treatment2', 'Treatment3', 'Treatment4']
    results = []
    
    for treatment in treatment_cols:
        treated = df[df[treatment] == 1]
        untreated = df[df[treatment] == 0]
        
        treated_diarrhea_rate = treated['Diarrhea_Had'].mean()
        untreated_diarrhea_rate = untreated['Diarrhea_Had'].mean()
        
        results.append({
            'Treatment': treatment,
            'Treated_Diarrhea_Rate': treated_diarrhea_rate,
            'Untreated_Diarrhea_Rate': untreated_diarrhea_rate,
            'Effectiveness': (untreated_diarrhea_rate - treated_diarrhea_rate) / untreated_diarrhea_rate * 100
        })
    
    results_df = pd.DataFrame(results)
    
    fig = px.bar(results_df, x='Treatment', y='Effectiveness',
                 title='Treatment Effectiveness (Reduction in Diarrhea Rate)',
                 labels={'Effectiveness': 'Effectiveness (%)'})
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

@app.route('/api/filtered_data', methods=['POST'])
def filtered_data():
    filters = request.json
    
    filtered_df = df.copy()
    
    if 'treatment' in filters:
        treatment = filters['treatment']
        filtered_df = filtered_df[filtered_df[treatment] == 1]
    
    if 'gender' in filters:
        gender_map = {'male': 0, 'female': 1}
        filtered_df = filtered_df[filtered_df['gender'] == gender_map[filters['gender']]]
    
    if 'diarrhea' in filters:
        filtered_df = filtered_df[filtered_df['Diarrhea_Had'] == (1 if filters['diarrhea'] == 'yes' else 0)]
    
    return filtered_df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
