from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ------------------------------
# Load trained artifacts
# ------------------------------
rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
X_columns = joblib.load("X_columns.pkl")

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return "🎵 Song Stream Prediction API is Running!"

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        required_fields = ["Artist", "Album_type", "Danceability", "Energy", "Tempo"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Unknown category handling
        if data["Artist"] not in label_encoders["Artist"].classes_:
            return jsonify({"error": "Unknown artist"}), 400

        if data["Album_type"] not in label_encoders["Album_type"].classes_:
            return jsonify({"error": "Unknown album type"}), 400

        artist_encoded = label_encoders["Artist"].transform([data["Artist"]])[0]
        album_type_encoded = label_encoders["Album_type"].transform([data["Album_type"]])[0]

        user_input = pd.DataFrame([{
            "Artist": artist_encoded,
            "Album_type": album_type_encoded,
            "Danceability": float(data["Danceability"]),
            "Energy": float(data["Energy"]),
            "Tempo": float(data["Tempo"])
        }])

        # Fill remaining features with training means
        for col in X_columns:
            if col not in user_input.columns:
                user_input[col] = 0

        user_input = user_input[X_columns]
        user_scaled = scaler.transform(user_input)

        pred_stream = (rf.predict(user_scaled) + xgb.predict(user_scaled)) / 2

        return jsonify({
            "Predicted_Stream": float(pred_stream[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Start Flask (Render-compatible)
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render injects PORT
    app.run(host="0.0.0.0", port=port)
