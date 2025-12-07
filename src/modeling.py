import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import randint, uniform

FEATURES = [
    "settlement_period", "day_of_month", "day_of_week", "day_of_year",
    "quarter", "month", "year", "week_of_year",
    "lag1", "lag2", "lag3", "is_holiday"
]

TARGET = "tsd"

def make_splits(df):
    t1 = pd.to_datetime("01-06-2019", dayfirst=True)
    t2 = pd.to_datetime("01-06-2021", dayfirst=True)
    train = df[df.index < t1]
    test = df[(df.index >= t1) & (df.index < t2)]
    hold = df[df.index >= t2]
    return train, test, hold

def train_best(train, test, hold):
    X_train, y_train = train[FEATURES], train[TARGET]
    X_hold, y_hold = hold[FEATURES], hold[TARGET]
    base = xgb.XGBRegressor(
        booster="gbtree",
        tree_method="hist",
        objective="reg:squarederror",
        random_state=43,
        n_estimators=1500,
        learning_rate=0.02,
        early_stopping_rounds=80,
        n_jobs=-1,
        eval_metric="rmse"
    )
    params = {
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 8),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "gamma": uniform(0, 3),
        "reg_alpha": uniform(0, 5),
        "reg_lambda": uniform(0, 5)
    }
    tss = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        base,
        params,
        n_iter=40,
        cv=tss,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train, eval_set=[(X_hold, y_hold)], verbose=False)
    best = xgb.XGBRegressor(
        **search.best_params_,
        booster="gbtree",
        tree_method="hist",
        objective="reg:squarederror",
        random_state=43,
        n_estimators=1500,
        learning_rate=0.02,
        early_stopping_rounds=80,
        n_jobs=-1,
        eval_metric="rmse"
    )
    best.fit(X_train, y_train, eval_set=[(X_hold, y_hold)], verbose=False)
    pred = best.predict(test[FEATURES])
    rmse = root_mean_squared_error(test[TARGET], pred)
    r2 = r2_score(test[TARGET], pred)
    return best, rmse, r2