import datetime
import json
from pathlib import Path
from typing import Tuple

import holidays
import numpy as np
import pandas as pd

from config import (
    BASE_PATH,
    RAW_DATA_PATH,
    PROCESSED_DIR,
    REQUIRED_COLUMNS,
    DROP_COLUMNS,
    TRAIN_STATE_PATH,
)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    df.drop(index=df[df["settlement_period"] > 48].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["tsd"] = pd.to_numeric(df["tsd"], errors="coerce")
    df.dropna(subset=["tsd"], inplace=True)

    null_days = df.loc[df["tsd"] == 0.0, "settlement_date"].unique().tolist()
    if null_days:
        idx = []
        for day in null_days:
            idx.extend(df[df["settlement_date"] == day].index.tolist())
        df.drop(index=idx, inplace=True)
        df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("Data is empty after cleaning.")

    return df


def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    years = pd.to_datetime(df["settlement_date"]).dt.year.unique().tolist()
    uk_holidays = holidays.UK(subdiv="England", years=years, observed=True)

    df["is_holiday"] = df["settlement_date"].apply(
        lambda x: pd.to_datetime(x).date() in uk_holidays
    )
    df["is_holiday"] = df["is_holiday"].astype(int)
    return df


def build_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_hour"] = df["settlement_period"].apply(
        lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5))
    )
    df.loc[df["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"

    period_hour_col = df.pop("period_hour")
    df.insert(2, "period_hour", period_hour_col)

    df["settlement_date"] = pd.to_datetime(df["settlement_date"] + " " + df["period_hour"])
    df.set_index("settlement_date", inplace=True)
    df.sort_index(inplace=True)

    if df.index.hasnans:
        df = df[~df.index.isna()]

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.day_of_week
    df["day_of_year"] = df.index.day_of_year
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week_of_year"] = df.index.isocalendar().week.astype("int64")
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target_map = df["tsd"].to_dict()
    df["lag1"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag2"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag3"] = (df.index - pd.Timedelta("1092 days")).map(target_map)
    df.dropna(subset=["lag1", "lag2", "lag3"], inplace=True)
    return df


def train_test_holdout_split(
    df: pd.DataFrame,
    threshold_date_1: str = "06-01-2019",
    threshold_date_2: str = "06-01-2021",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    t1 = pd.to_datetime(threshold_date_1, dayfirst=True)
    t2 = pd.to_datetime(threshold_date_2, dayfirst=True)

    train_data = df.loc[df.index < t1]
    test_data = df.loc[(df.index >= t1) & (df.index < t2)]
    hold_out_data = df.loc[df.index >= t2]

    if train_data.empty or test_data.empty or hold_out_data.empty:
        raise ValueError(
            "One of the splits is empty. Check threshold dates and data coverage."
        )

    return train_data, test_data, hold_out_data


def save_processed_splits(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    hold_out_data: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(PROCESSED_DIR / "train_data.csv", index=False)
    test_data.to_csv(PROCESSED_DIR / "test_data.csv", index=False)
    hold_out_data.to_csv(PROCESSED_DIR / "hold_out_data.csv", index=False)


def get_raw_last_timestamp(df: pd.DataFrame) -> pd.Timestamp:
    return df.index.max()


def should_train(df: pd.DataFrame) -> bool:
    current_last_ts = get_raw_last_timestamp(df)

    if TRAIN_STATE_PATH.exists():
        with TRAIN_STATE_PATH.open("r") as f:
            state = json.load(f)
        prev_ts_str = state.get("last_timestamp")
        if prev_ts_str:
            prev_ts = pd.to_datetime(prev_ts_str)
            if current_last_ts <= prev_ts:
                print("No new data since last training. Skipping training.")
                return False

    return True


def update_train_state(df: pd.DataFrame) -> None:
    TRAIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {"last_timestamp": get_raw_last_timestamp(df).isoformat()}
    with TRAIN_STATE_PATH.open("w") as f:
        json.dump(state, f)