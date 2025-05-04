import os
import joblib

def load_model():
    model_path = 'policy_model2.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        print("Model file not found, using dummy model")
        return None  # or return a dummy model

