from data_pipeline import (
    add_holiday_feature,
    add_lags,
    basic_clean,
    build_datetime_index,
    create_features,
    load_raw_data,
    save_processed_splits,
    should_train,
    train_test_holdout_split,
    update_train_state,
)
from modeling import save_model_and_metrics, train_model


def main() -> None:
    df_raw = load_raw_data()
    df_clean = basic_clean(df_raw)
    df_holiday = add_holiday_feature(df_clean)
    df_ts = build_datetime_index(df_holiday)
    df_feat = create_features(df_ts)
    df_feat = add_lags(df_feat)

    if not should_train(df_feat):
        return

    train_data, test_data, hold_out_data = train_test_holdout_split(df_feat)
    save_processed_splits(train_data, test_data, hold_out_data)

    model, metrics = train_model(train_data, test_data, hold_out_data)
    save_model_and_metrics(model, metrics)
    update_train_state(df_feat)

    print("Training complete.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()