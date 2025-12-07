import pandas as pd
import datetime
import holidays

FEATURES = [
    "settlement_period", "day_of_month", "day_of_week", "day_of_year",
    "quarter", "month", "year", "week_of_year",
    "lag1", "lag2", "lag3", "is_holiday"
]

uk_holidays = holidays.UK(subdiv="England", years=range(2009, 2026))

def load_history(path):
    df = pd.read_parquet(path)
    df.sort_index(inplace=True)
    return df

def make_timestamp(date_str: str, period: int) -> pd.Timestamp:
    td = datetime.timedelta(hours=(period - 1) * 0.5)
    s = str(td)
    if s == "1 day, 0:00:00":
        s = "0:00:00"
    return pd.to_datetime(f"{date_str} {s}", dayfirst=True)

def build_features(ts: pd.Timestamp, history: pd.DataFrame) -> pd.DataFrame:
    row = {}
    row["settlement_period"] = int(((ts - ts.normalize()).seconds // 1800) + 1)
    row["day_of_month"] = ts.day
    row["day_of_week"] = ts.day_of_week
    row["day_of_year"] = ts.day_of_year
    row["quarter"] = ts.quarter
    row["month"] = ts.month
    row["year"] = ts.year
    row["week_of_year"] = int(ts.isocalendar().week)
    row["lag1"] = history["tsd"].get(ts - pd.Timedelta("364 days"))
    row["lag2"] = history["tsd"].get(ts - pd.Timedelta("728 days"))
    row["lag3"] = history["tsd"].get(ts - pd.Timedelta("1092 days"))
    fallback = history["tsd"].tail(2000).mean()
    for k in ("lag1", "lag2", "lag3"):
        if row[k] is None:
            row[k] = fallback
    d = ts.date()
    row["is_holiday"] = int(d in uk_holidays)
    df = pd.DataFrame([row], index=[ts])
    return df[FEATURES]