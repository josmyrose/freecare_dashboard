
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

def predict_defect(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)[0][1]  # Probability of defect
    return prediction
