from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

app = Flask(__name__)

# -----------------------------
# Load model and scaler C:\Users\samik\Documents\pythonlearn\MLZC2025\model.binmodel.bin

# -----------------------------
with open("/workspaces/repo-project/Wine-Quality-tool/model.bin", "rb") as f:
    model = pickle.load(f)

#with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# List of features the model expects (IMPORTANT: must match training order)
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol"
]

@app.route("/")
def home():
    return jsonify({"message": "Wine Quality Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data], columns=FEATURES)

        # Scale input
        #X_scaled = scaler.transform(df)
        X_scaled =df

        # Predict
        pred = model.predict(X_scaled)[0]
        probability = float(model.predict_proba(X_scaled)[0][1])

        return jsonify({
            "good_quality_prediction": int(pred),
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
