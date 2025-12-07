import os
import datetime
import pickle
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Load environment variables from .env (for local/dev)
load_dotenv()

# Paths
BASE_PATH = Path(os.getenv("BASE_PATH", ".")).resolve()
RAW_DATA_PATH = BASE_PATH / os.getenv("RAW_DATA_PATH", "data/raw/historic_demand_2009_2024.csv")
PROCESSED_DIR = BASE_PATH / os.getenv("PROCESSED_DIR", "data/processed")
MODEL_DIR = BASE_PATH / os.getenv("MODEL_DIR", "model")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load and clean raw data
df = pd.read_csv(RAW_DATA_PATH, index_col=0)
df.columns = df.columns.str.lower()
df.drop(columns=["scottish_transfer", "viking_flow", "greenlink_flow"], inplace=True)
df.drop(columns=["nsl_flow", "eleclink_flow"], axis=1, inplace=True)
df.drop(index=df[df["settlement_period"] > 48].index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Holiday feature
uk_holidays = holidays.UK(subdiv="England", years=range(2009, 2024), observed=True)
df["is_holiday"] = df["settlement_date"].apply(
    lambda x: pd.to_datetime(x).date() in uk_holidays
)
df["is_holiday"] = df["is_holiday"].astype(int)

# Remove broken days
null_days = df.loc[df["tsd"] == 0.0, "settlement_date"].unique().tolist()
null_days_index = []
for day in null_days:
    null_days_index.extend(df[df["settlement_date"] == day].index.tolist())
df.drop(index=null_days_index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Build proper datetime index
df["period_hour"] = df["settlement_period"].apply(
    lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5))
)
df.loc[df["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"
period_hour_col = df.pop("period_hour")
df.insert(2, "period_hour", period_hour_col)

df["settlement_date"] = pd.to_datetime(df["settlement_date"] + " " + df["period_hour"])
df.set_index("settlement_date", inplace=True)
df.sort_index(inplace=True)


def create_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["day_of_month"] = frame.index.day
    frame["day_of_week"] = frame.index.day_of_week
    frame["day_of_year"] = frame.index.day_of_year
    frame["quarter"] = frame.index.quarter
    frame["month"] = frame.index.month
    frame["year"] = frame.index.year
    frame["week_of_year"] = frame.index.isocalendar().week.astype("int64")
    return frame


def add_lags(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    target_map = frame["tsd"].to_dict()
    frame["lag1"] = (frame.index - pd.Timedelta("364 days")).map(target_map)
    frame["lag2"] = (frame.index - pd.Timedelta("728 days")).map(target_map)
    frame["lag3"] = (frame.index - pd.Timedelta("1092 days")).map(target_map)
    return frame


df = create_features(df)
df = add_lags(df)
df.dropna(subset=["lag1", "lag2", "lag3"], inplace=True)

# Time-based splits
threshold_date_1 = pd.to_datetime("06-01-2019", dayfirst=True)
threshold_date_2 = pd.to_datetime("06-01-2021", dayfirst=True)

train_data = df.loc[df.index < threshold_date_1]
test_data = df.loc[(df.index >= threshold_date_1) & (df.index < threshold_date_2)]
hold_out_data = df.loc[df.index >= threshold_date_2]

FEATURES = [
    "settlement_period",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "quarter",
    "month",
    "year",
    "week_of_year",
    "lag1",
    "lag2",
    "lag3",
    "is_holiday",
]
TARGET = "tsd"

X_train = train_data[FEATURES]
y_train = train_data[TARGET]

X_test = test_data[FEATURES]
y_test = test_data[TARGET]

X_hold_out = hold_out_data[FEATURES]
y_hold_out = hold_out_data[TARGET]

# Hyperparameter search
n_splits = 5
tss = TimeSeriesSplit(n_splits=n_splits)

base_estimator = xgb.XGBRegressor(
    booster="gbtree",
    tree_method="hist",
    random_state=43,
    objective="reg:squarederror",
    n_estimators=1500,
    learning_rate=0.02,
    early_stopping_rounds=80,
    n_jobs=-1,
    eval_metric="rmse",
)

param_dist = {
    "max_depth": randint(3, 10),
    "min_child_weight": randint(1, 8),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "gamma": uniform(0, 3),
    "reg_alpha": uniform(0, 5),
    "reg_lambda": uniform(0, 5),
}

fit_params = {
    "eval_set": [(X_hold_out, y_hold_out)],
    "verbose": False,
}

xgb_search = RandomizedSearchCV(
    estimator=base_estimator,
    param_distributions=param_dist,
    n_iter=40,
    cv=tss,
    scoring="neg_root_mean_squared_error",
    verbose=2,
    n_jobs=-1,
    random_state=43,
    error_score="raise",
)

xgb_search.fit(X_train, y_train, **fit_params)
best_params = xgb_search.best_params_

# Final model
final_model = xgb.XGBRegressor(
    **best_params,
    booster="gbtree",
    tree_method="hist",
    random_state=43,
    objective="reg:squarederror",
    n_estimators=1500,
    learning_rate=0.02,
    early_stopping_rounds=80,
    n_jobs=-1,
    eval_metric="rmse",
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_hold_out, y_hold_out)],
    verbose=True,
)

# Evaluation
y_pred_test = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("\nTest performance")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")

# Save processed data and model
train_data.to_csv(PROCESSED_DIR / "train_data.csv", index=False)
test_data.to_csv(PROCESSED_DIR / "test_data.csv", index=False)
hold_out_data.to_csv(PROCESSED_DIR / "hold_out_data.csv", index=False)

with open(MODEL_DIR / "final_xgb_model.pkl", "wb") as f:
    pickle.dump(final_model, f)