# ------------------------------
# 1. Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ------------------------------
# 2. Load Dataset
# ------------------------------
df = pd.read_csv("cleaned_dataset.csv")
print("Dataset Loaded Successfully")
print(df.head())
print("Dataset Shape:", df.shape)

# ------------------------------
# 3. Data Cleaning & Missing Values
# ------------------------------
print("\nMissing Values Before:\n", df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing Values After:\n", df.isnull().sum())

# ------------------------------
# 4. Encoding Categorical Variables
# ------------------------------
label_encoders = {}
cat_cols = ["Artist", "Album", "Album_type", "Title", "Channel", "most_playedon"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df["Licensed"] = df["Licensed"].replace({"True": 1, "False": 0}).astype(int)
df["official_video"] = df["official_video"].replace({"True": 1, "False": 0}).astype(int)

# ------------------------------
# 5. Feature Selection & Scaling
# ------------------------------
X = df.drop(["Track", "Stream"], axis=1)
y = df["Stream"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 6. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# 7. Train Random Forest Model
# ------------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Performance")
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# ------------------------------
# 8. Train XGBoost Model
# ------------------------------
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective="reg:squarederror"
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Performance")
print("RMSE:", xgb_rmse)
print("R2 Score:", xgb_r2)

# ------------------------------
# 9. Ensemble Evaluation
# ------------------------------
ensemble_pred = (rf_pred + xgb_pred) / 2

ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_r2 = r2_score(y_test, ensemble_pred)

print("\nEnsemble Model Performance")
print("RMSE:", ensemble_rmse)
print("R2 Score:", ensemble_r2)

# ------------------------------
# 10. Save Trained Artifacts
# ------------------------------
joblib.dump(rf, "rf_model.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(X.columns.tolist(), "X_columns.pkl")

print("\n✅ All artifacts saved successfully:")
print(" - rf_model.pkl")
print(" - xgb_model.pkl")
print(" - scaler.pkl")
print(" - label_encoders.pkl")
print(" - X_columns.pkl")
