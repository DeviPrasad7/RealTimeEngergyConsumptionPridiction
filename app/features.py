import pandas as pd
import datetime
import holidays

FEATURES = [
    "settlement_period", "day_of_month", "day_of_week", "day_of_year",
    "quarter", "month", "year", "week_of_year",
    "lag1", "lag2", "lag3", "is_holiday"
]

def load_history(path):
    df = pd.read_parquet(path)
    df.sort_index(inplace=True)
    return df

def make_timestamp(date_str: str, period: int) -> pd.Timestamp:
    td = datetime.timedelta(hours=(period - 1) * 0.5)
    s = str(td)
    if s == "1 day, 0:00:00":
        s = "0:00:00"
    return pd.to_datetime(f"{date_str} {s}", format='%Y-%m-%d %H:%M:%S')

def build_features(ts: pd.Timestamp | pd.DatetimeIndex, history: pd.DataFrame) -> pd.DataFrame:
    if isinstance(ts, pd.Timestamp):
        ts = pd.DatetimeIndex([ts])

    df = pd.DataFrame(index=ts)

    df["settlement_period"] = ((df.index - df.index.normalize()).seconds // 1800) + 1
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["day_of_year"] = df.index.dayofyear
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    
    history_map = history["tsd"].to_dict()
    df["lag1"] = (df.index - pd.Timedelta("364 days")).map(history_map)
    df["lag2"] = (df.index - pd.Timedelta("728 days")).map(history_map)
    df["lag3"] = (df.index - pd.Timedelta("1092 days")).map(history_map)

    fallback = history["tsd"].tail(2000).mean()
    
    df[["lag1", "lag2", "lag3"]] = df[["lag1", "lag2", "lag3"]].ffill().bfill()
    df[["lag1", "lag2", "lag3"]] = df[["lag1", "lag2", "lag3"]].fillna(fallback)

    min_year = df.index.year.min()
    max_year = df.index.year.max()
    uk_holidays = holidays.UK(subdiv="England", years=range(min_year, max_year + 1))
    df["is_holiday"] = df.index.to_series().apply(lambda x: x.date() in uk_holidays).astype(int)

    return df[FEATURES]