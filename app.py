from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

app = Flask(__name__)

# Load trained artifacts
rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
X_columns = joblib.load("X_columns.pkl")

@app.route("/")
def home():
    return "🎵 Song Stream Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    artist_encoded = label_encoders["Artist"].transform([data["Artist"]])[0]
    album_type_encoded = label_encoders["Album_type"].transform([data["Album_type"]])[0]

    user_input = pd.DataFrame([{
        "Artist": artist_encoded,
        "Album_type": album_type_encoded,
        "Danceability": data["Danceability"],
        "Energy": data["Energy"],
        "Tempo": data["Tempo"]
    }])

    for col in X_columns:
        if col not in user_input.columns:
            user_input[col] = 0

    user_input = user_input[X_columns]
    user_scaled = scaler.transform(user_input)

    pred_stream = (rf.predict(user_scaled) + xgb.predict(user_scaled)) / 2

    return jsonify({"Predicted_Stream": float(pred_stream[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
