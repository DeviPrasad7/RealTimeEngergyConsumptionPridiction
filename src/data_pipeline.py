import datetime
from pathlib import Path
import pandas as pd
import holidays

def clean_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df["settlement_date"] = (
        df["settlement_date"]
        .astype(str)
        .str.replace(r"[^0-9\-\/]", "", regex=True)
        .str.slice(0, 10)
    )
    df["settlement_date"] = pd.to_datetime(
        df["settlement_date"],
        # dayfirst=True,
        format="mixed",
        errors="coerce"
    )
    df = df.dropna(subset=["settlement_date"])
    df = df[df["settlement_period"] <= 48]
    return df

def apply_holidays(df):
    uk = holidays.UK(subdiv="England", years=range(2009, 2026))
    df["is_holiday"] = df["settlement_date"].apply(lambda x: x.date() in uk).astype(int)
    return df

def remove_zero_days(df):
    bad = df[df["tsd"] == 0.0]["settlement_date"].unique()
    df = df[~df["settlement_date"].isin(bad)].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def apply_datetime_index(df):
    df["period_hour"] = df["settlement_period"].apply(
        lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5))
    )
    df.loc[df["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"
    p = df.pop("period_hour")
    df.insert(2, "period_hour", p)
    df["timestamp"] = pd.to_datetime(
        df["settlement_date"].dt.strftime("%Y-%m-%d") + " " + df["period_hour"]
    )
    df = df.set_index("timestamp").sort_index()
    return df

def add_calendar(df):
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.day_of_week
    df["day_of_year"] = df.index.day_of_year
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    return df

def add_lags(df):
    m = df["tsd"].to_dict()
    df["lag1"] = (df.index - pd.Timedelta("364 days")).map(m)
    df["lag2"] = (df.index - pd.Timedelta("728 days")).map(m)
    df["lag3"] = (df.index - pd.Timedelta("1092 days")).map(m)
    df[["lag1", "lag2", "lag3"]] = df[["lag1", "lag2", "lag3"]].bfill()
    df.dropna(subset=["lag1", "lag2", "lag3"], inplace=True)
    return df

def preprocess(df):
    df = apply_holidays(df)
    # df = remove_zero_days(df)
    df = apply_datetime_index(df)
    df = add_calendar(df)
    df = add_lags(df)
    return df