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

def make_splits(df, train_pct=0.7, test_pct=0.15):
    """Splits the dataframe into training, testing, and holdout sets."""
    n = len(df)
    train_end = int(n * train_pct)
    test_end = int(n * (train_pct + test_pct))
    
    train = df.iloc[:train_end]
    test = df.iloc[train_end:test_end]
    hold = df.iloc[test_end:]
    
    return train, test, hold

def train_best(train, test, hold):
    """Trains the best model using RandomizedSearchCV."""
    X_train, y_train = train[FEATURES], train[TARGET]
    X_hold, y_hold = hold[FEATURES], hold[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

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
    
    best_model = search.best_estimator_
    
    pred = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    return best_model, rmse, r2