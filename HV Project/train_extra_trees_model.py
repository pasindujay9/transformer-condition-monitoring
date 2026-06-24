import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score


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

    df["TDCG"] = (
        df["H2"] +
        df["Methane"] +
        df["Ethane"] +
        df["Ethylene"] +
        df["Acetylene"] +
        df["CO"]
    )

    df["EstimatedDP"] = np.where(
        df["Furan"] > 0,
        (1.51 - np.log10(df["Furan"])) / 0.0035,
        1000
    )

    df["EstimatedDP"] = df["EstimatedDP"].clip(lower=0, upper=1200)

    df["CO2_CO_Ratio"] = df["CO2"] / (df["CO"] + 1e-6)

    df["Furan_Acid"] = df["Furan"] * df["Acid"]
    df["Water_Acid"] = df["Water"] * df["Acid"]
    df["Inv_BDV"] = 1 / (df["BDV"] + 1e-6)
    df["Inv_IFT"] = 1 / (df["IFT"] + 1e-6)

    log_columns = [
        "H2", "CO", "CO2", "Methane", "Acetylene",
        "Ethylene", "Ethane", "Furan", "TDCG"
    ]

    for col in log_columns:
        df["log_" + col] = np.log10(df[col] + 1)

    return df


# Load dataset
data = pd.read_csv("DatasetA.csv")

# Add features
data = add_features(data)

# Create condition labels
data["Condition"] = data["HI"].apply(classify_condition)

# Input features
feature_names = [
    "H2", "CO", "CO2", "Methane", "Acetylene", "Ethylene", "Ethane",
    "Furan", "Water", "Acid", "BDV", "DDF1", "DDF2", "Color", "IFT",
    "TDCG", "EstimatedDP", "CO2_CO_Ratio",
    "Furan_Acid", "Water_Acid", "Inv_BDV", "Inv_IFT",
    "log_H2", "log_CO", "log_CO2", "log_Methane", "log_Acetylene",
    "log_Ethylene", "log_Ethane", "log_Furan", "log_TDCG"
]

X = data[feature_names]
y_reg = data["HI"]
y_cls = data["Condition"]

# Train-test split
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X,
    y_reg,
    y_cls,
    test_size=0.20,
    random_state=1,
    stratify=y_cls
)

# --------------------------------------------------
# Model 1: Extra Trees Regressor for HI prediction
# --------------------------------------------------

reg_model = ExtraTreesRegressor(
    n_estimators=800,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=1,
    n_jobs=-1
)

reg_model.fit(X_train, y_reg_train)

y_reg_pred = reg_model.predict(X_test)
y_reg_pred = np.clip(y_reg_pred, 0, 100)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg_test, y_reg_pred)

# HI-threshold condition from predicted HI
hi_based_condition = [classify_condition(v) for v in y_reg_pred]
hi_based_accuracy = accuracy_score(y_cls_test, hi_based_condition) * 100

# --------------------------------------------------
# Model 2: Extra Trees Classifier for direct condition prediction
# --------------------------------------------------

cls_model = ExtraTreesClassifier(
    n_estimators=800,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=1,
    n_jobs=-1,
    class_weight="balanced"
)

cls_model.fit(X_train, y_cls_train)

y_cls_pred = cls_model.predict(X_test)

direct_condition_accuracy = accuracy_score(y_cls_test, y_cls_pred) * 100

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nExtra Trees Model Performance")
print("--------------------------------")
print("HI Regression Model:")
print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")
print(f"HI-based Condition Accuracy = {hi_based_accuracy:.2f} %")

print("\nDirect Condition Classification Model:")
print(f"Direct Condition Accuracy = {direct_condition_accuracy:.2f} %")

# Save both models
model_package = {
    "regression_model": reg_model,
    "classification_model": cls_model,
    "feature_names": feature_names
}

joblib.dump(model_package, "transformer_extra_trees_model.pkl")

print("\nModel saved as transformer_extra_trees_model.pkl")


