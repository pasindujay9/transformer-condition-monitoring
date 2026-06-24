import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def classify_condition(hi):
    if hi >= 85:
        return "Good"
    elif hi >= 70:
        return "Fair"
    elif hi >= 50:
        return "Poor / Warning"
    else:
        return "Critical"


def add_features(df):
    df = df.copy()

    # TDCG calculation
    df["TDCG"] = (
        df["H2"]
        + df["Methane"]
        + df["Ethane"]
        + df["Ethylene"]
        + df["Acetylene"]
        + df["CO"]
    )

    # Estimated DP from Furan / 2-FAL
    # Chendong equation: DP = (1.51 - log10(Furan)) / 0.0035
    df["EstimatedDP"] = np.where(
        df["Furan"] > 0,
        (1.51 - np.log10(df["Furan"])) / 0.0035,
        1000
    )

    # Limit very high DP values to a realistic upper value
    df["EstimatedDP"] = df["EstimatedDP"].clip(lower=0, upper=1200)

    # Paper ageing indicator ratio
    df["CO2_CO_Ratio"] = df["CO2"] / (df["CO"] + 1e-6)

    # Log-transformed gas/furan features
    log_columns = [
        "H2", "CO", "CO2", "Methane", "Acetylene",
        "Ethylene", "Ethane", "Furan", "TDCG"
    ]

    for col in log_columns:
        df["log_" + col] = np.log10(df[col] + 1)

    return df


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_csv("DatasetA.csv")

# --------------------------------------------------
# 2. Add diagnostic features
# --------------------------------------------------

data = add_features(data)

# --------------------------------------------------
# 3. Select input features and output
# --------------------------------------------------

feature_names = [
    "H2", "CO", "CO2", "Methane", "Acetylene", "Ethylene", "Ethane",
    "Furan", "Water", "Acid", "BDV", "DDF1", "DDF2", "Color", "IFT",
    "TDCG", "EstimatedDP", "CO2_CO_Ratio",
    "log_H2", "log_CO", "log_CO2", "log_Methane", "log_Acetylene",
    "log_Ethylene", "log_Ethane", "log_Furan", "log_TDCG"
]

X = data[feature_names]
y = data["HI"]

# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=1
)

# --------------------------------------------------
# 5. Train Random Forest model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=1,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Predict and evaluate
# --------------------------------------------------

y_pred = model.predict(X_test)

# Limit predicted HI between 0 and 100
y_pred = np.clip(y_pred, 0, 100)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

actual_condition = [classify_condition(v) for v in y_test]
predicted_condition = [classify_condition(v) for v in y_pred]

condition_accuracy = np.mean(
    np.array(actual_condition) == np.array(predicted_condition)
) * 100

print("\nRandom Forest Model Performance")
print("--------------------------------")
print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")
print(f"Condition Classification Accuracy = {condition_accuracy:.2f} %")

# --------------------------------------------------
# 7. Show first 10 predictions
# --------------------------------------------------

result_table = pd.DataFrame({
    "Actual_HI": y_test.values[:10],
    "Predicted_HI": y_pred[:10],
    "Actual_Condition": actual_condition[:10],
    "Predicted_Condition": predicted_condition[:10],
    "Absolute_Error": np.abs(y_test.values[:10] - y_pred[:10])
})

print("\nFirst 10 actual vs predicted results:")
print(result_table)

# --------------------------------------------------
# 8. Save trained model
# --------------------------------------------------

model_package = {
    "model": model,
    "feature_names": feature_names
}

joblib.dump(model_package, "transformer_hi_model.pkl")

print("\nModel saved as transformer_hi_model.pkl")
