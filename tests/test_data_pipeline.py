import pandas as pd
from src.data_pipeline import apply_holidays

def test_apply_holidays_future_year():
    dates = pd.to_datetime(["2027-12-24", "2027-12-25", "2027-12-26", "2027-12-27", "2027-12-28"])
    df = pd.DataFrame({"settlement_date": dates})
    
    df_with_holidays = apply_holidays(df.copy())
    
    assert "is_holiday" in df_with_holidays.columns
    assert df_with_holidays.loc[0, "is_holiday"] == 0
    assert df_with_holidays.loc[1, "is_holiday"] == 1
    assert df_with_holidays.loc[2, "is_holiday"] == 1
    assert df_with_holidays.loc[3, "is_holiday"] == 1
    assert df_with_holidays.loc[4, "is_holiday"] == 1
