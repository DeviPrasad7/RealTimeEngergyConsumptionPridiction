import json
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config import METRICS_PATH, MODEL_PATH


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


def build_search(
    n_splits: int = 5,
) -> RandomizedSearchCV:
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

    tss = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
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
    return search


def train_model(
    train_data,
    test_data,
    hold_out_data,
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    X_hold_out = hold_out_data[FEATURES]
    y_hold_out = hold_out_data[TARGET]

    search = build_search()
    fit_params = {"eval_set": [(X_hold_out, y_hold_out)], "verbose": False}
    search.fit(X_train, y_train, **fit_params)

    best_params = search.best_params_

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

    y_pred_test = final_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    return final_model, metrics


def save_model_and_metrics(model: xgb.XGBRegressor, metrics: Dict[str, float]) -> None:
    import pickle

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)