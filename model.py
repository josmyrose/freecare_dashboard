import os
import joblib

def load_model():
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # Running in GitHub Actions — return a dummy model
        return None
    else:
        # Running locally — load the real model
        return joblib.load('models/policy_model2.joblib')

