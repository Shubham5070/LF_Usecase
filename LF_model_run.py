import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
DB_PATH = Path("data/load_forecasting.db")
RUN_DATE = "2026-01-15"
MODEL_ID = 202
HORIZON_TYPE = "day_ahead"
MODEL_PATH = "amazon/chronos-t5-large"

FREQ = "15min"
BLOCKS_PER_DAY = 96
MAX_ALLOWED_GAP_HOURS = 36

# Lookback parameters
LBY = 0  # lookback years
LBM = 4  # lookback months

# ------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------
def celsius_to_fahrenheit(temp_celsius):
    return temp_celsius * 9/5 + 32

def fahrenheit_to_celsius(temp_fahrenheit):
    return (temp_fahrenheit - 32) * 5 / 9

def heat_index_from_celsius(t_celsius, rh):
    t = celsius_to_fahrenheit(t_celsius)
    simple_hi = 0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094))
    hi = 0.5 * (simple_hi + t)
    
    if hi >= 80:
        hi = (
            -42.379 + (2.04901523 * t) + (10.14333127 * rh)
            - (0.22475541 * t * rh) - (0.00683783 * t * t)
            - (0.05481717 * rh * rh) + (0.00122874 * t * t * rh)
            + (0.00085282 * t * rh * rh) - (0.00000199 * t * t * rh * rh)
        )
        
        if rh < 13 and 80 <= t <= 112:
            adjustment = ((13 - rh) / 4) * np.sqrt((17 - abs(t - 95)) / 17)
            hi -= adjustment
        elif rh > 85 and 80 <= t <= 87:
            adjustment = ((rh - 85) / 10) * ((87 - t) / 5)
            hi += adjustment
    
    return fahrenheit_to_celsius(hi)

def create_cyclic_features(df):
    df["minute"] = (df.index.hour + 1) + df.index.minute / 60
    df["hour"] = df.index.hour + 1
    df["day_of_week"] = df.index.dayofweek + 1
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 1440)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    min_year = df["year"].min()
    max_year = df["year"].max()
    year_range = max_year - min_year + 1
    df["year_sin"] = np.sin(2 * np.pi * (df["year"] - min_year) / year_range)
    df["year_cos"] = np.cos(2 * np.pi * (df["year"] - min_year) / year_range)
    df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
    df['is_day_of_week'] = df['day_of_week'].isin([1, 2, 3, 4, 5]).astype(int)
    return df

def dedupe_index(df, keep="first"):
    df = df.sort_index()
    return df[~df.index.duplicated(keep=keep)]

# ------------------------------------------------
# DATA PREPARATION FUNCTIONS
# ------------------------------------------------
def prepare_train_data(conn, input_date, lby=0, lbm=4, hrs_end=5):
    ip_date = pd.to_datetime(input_date)
    train_start = (ip_date + pd.Timedelta(hours=hrs_end, minutes=45) 
                   - pd.DateOffset(years=lby, months=lbm, days=1))
    train_end = ip_date + pd.Timedelta(hours=hrs_end, minutes=45) - pd.DateOffset(days=1)
    
    # Load demand data
    query_load = f"""
        SELECT datetime as ds, demand as y
        FROM lf.t_actual_demand
        WHERE datetime >= '{train_start}'
        AND datetime <= '{train_end}' 
        ORDER BY datetime
    """
    dfl = pd.read_sql(query_load, conn, parse_dates=["ds"])
    
    # Load weather data
    query_weather = f"""
        SELECT datetime as ds, humidity, temp
        FROM t_actual_weather
        WHERE datetime >= '{train_start}' 
        AND datetime <= '{train_end}'
        ORDER BY datetime
    """
    dfw = pd.read_sql(query_weather, conn, parse_dates=["ds"])
    
    # Merge
    dflw = dfl.merge(dfw, on="ds", how="left")
    dflw["date"] = pd.to_datetime(dflw["ds"].dt.date)
    
    # Load holidays - removed day_of_week column
    query_holidays = f"""
        SELECT date, name, normal_holiday, special_day
        FROM t_holidays
        WHERE date >= '{train_start.date()}'  
        AND date <= '{train_end.date()}'
        ORDER BY date
    """
    dfh = pd.read_sql(query_holidays, conn, parse_dates=["date"])
    dfh['normal_holiday'] = dfh['normal_holiday'].astype('int64')
    dfh['special_day'] = dfh['special_day'].astype('int64')
    
    # Merge holidays
    dflwh = dflw.merge(dfh, on=["date"], how="left")
    dflwh.fillna(0, inplace=True)
    
    # Set index before creating cyclic features (to get day_of_week from index)
    dflwh.set_index("ds", inplace=True)
    dflwh.sort_index(inplace=True)
    
    # Create cyclic features (this will create day_of_week from index)
    df = create_cyclic_features(dflwh)
    
    # Feature engineering (now day_of_week exists)
    df["hour"] = df.index.hour + 1
    
    # Convert to proper pandas Series for comparison
    df_dates = pd.Series(df.index.date)
    holiday_dates = dfh["date"].dt.date.tolist()
    
    df["is_day_before_holiday"] = df_dates.shift(-1).isin(holiday_dates).astype(int).values
    df["is_day_after_holiday"] = df_dates.shift(1).isin(holiday_dates).astype(int).values
    
    df["nh_dow_interaction"] = df["normal_holiday"] * df["day_of_week"]
    df["sd_dow_interaction"] = df["special_day"] * df["day_of_week"]
    df["hi"] = df.apply(
        lambda row: heat_index_from_celsius(row["temp"], row["humidity"]), axis=1
    )
    
    df[['normal_holiday', 'special_day', 'is_weekend',
        'is_day_before_holiday', 'is_day_after_holiday',
        'nh_dow_interaction', 'sd_dow_interaction']] = df[[
        'normal_holiday', 'special_day', 'is_weekend',
        'is_day_before_holiday', 'is_day_after_holiday',
        'nh_dow_interaction', 'sd_dow_interaction']].astype(int)
    
    return df

def prepare_test_data(conn, input_date, hrs_start=6):
    ip_date = pd.to_datetime(input_date)
    test_start = ip_date + pd.Timedelta(hours=hrs_start) - pd.DateOffset(days=1)
    test_end = ip_date + pd.Timedelta(hours=23, minutes=45)
    
    # Load actual demand
    query_load = f"""
        SELECT datetime as ds, demand as y
        FROM t_actual_demand
        WHERE datetime >= '{test_start}'
        AND datetime <= '{test_end}' 
        ORDER BY datetime
    """
    dfl = pd.read_sql(query_load, conn, parse_dates=["ds"])
    
    # Load forecasted weather
    query_weather = f"""
        SELECT datetime as ds, date, humidity, temp
        FROM t_forecasted_weather
        WHERE datetime >= '{test_start}' 
        AND datetime <= '{test_end}'
        ORDER BY datetime
    """
    dfw = pd.read_sql(query_weather, conn, parse_dates=["ds", "date"])
    
    # Merge
    dflw = dfl.merge(dfw, on="ds", how="left")
    dflw["date"] = pd.to_datetime(dflw["ds"].dt.date)
    
    # Load holidays - removed day_of_week column
    query_holidays = f"""
        SELECT date, name, normal_holiday, special_day
        FROM t_holidays
        WHERE date >= '{test_start.date()}'  
        AND date <= '{test_end.date()}'
        ORDER BY date
    """
    dfh = pd.read_sql(query_holidays, conn, parse_dates=["date"])
    dfh['normal_holiday'] = dfh['normal_holiday'].astype('int64')
    dfh['special_day'] = dfh['special_day'].astype('int64')
    
    # Merge holidays
    dflwh = dflw.merge(dfh, on=["date"], how="left")
    dflwh.fillna(0, inplace=True)
    
    # Set index before creating cyclic features
    dflwh.set_index("ds", inplace=True)
    dflwh.sort_index(inplace=True)
    
    # Create cyclic features (this will create day_of_week from index)
    df = create_cyclic_features(dflwh)
    
    # Feature engineering
    df["hour"] = df.index.hour + 1
    
    # Convert to proper pandas Series for comparison
    df_dates = pd.Series(df.index.date)
    holiday_dates = dfh["date"].dt.date.tolist()
    
    df["is_day_before_holiday"] = df_dates.shift(-1).isin(holiday_dates).astype(int).values
    df["is_day_after_holiday"] = df_dates.shift(1).isin(holiday_dates).astype(int).values
    
    df["nh_dow_interaction"] = df["normal_holiday"] * df["day_of_week"]
    df["sd_dow_interaction"] = df["special_day"] * df["day_of_week"]
    df["hi"] = df.apply(
        lambda row: heat_index_from_celsius(row["temp"], row["humidity"]), axis=1
    )
    
    df[['normal_holiday', 'special_day', 'is_day_before_holiday', 'is_weekend',
        'is_day_after_holiday', 'nh_dow_interaction', 'sd_dow_interaction']] = df[[
        'normal_holiday', 'special_day', 'is_weekend',
        'is_day_before_holiday', 'is_day_after_holiday',
        'nh_dow_interaction', 'sd_dow_interaction']].astype(int)
    
    return df

def run_and_store_forecast(run_date: str, *, db_path: str | None = None, model_id: int | None = None, model_path: str | None = None, horizon_type: str | None = None) -> dict:
    """Run forecast for `run_date` and store results in DB.

    Returns:
        dict: {ok: bool, rows_written: int, prediction_date: str, model_id: int, metrics: {mape, rmse} | None, error: str | None}
    """
    # Use module defaults when values are not provided
    run_date = str(run_date)
    model_id = model_id or MODEL_ID
    model_path = model_path or MODEL_PATH
    horizon_type = horizon_type or HORIZON_TYPE

    # Use DatabaseFactory for portability
    from db_factory import DatabaseFactory

    cfg = DatabaseFactory.get_config()
    conn = DatabaseFactory.get_connection()

    # Create table in a DB-compatible way
    if cfg.get("type") == "postgresql":
        cur = conn.cursor()
        # prefer the `lf` schema (other code uses lf.*); create it if missing
        target_schema = "lf"
        cur.execute('CREATE SCHEMA IF NOT EXISTS "lf"')
        # check if table exists in the target schema
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
            (target_schema, "t_predicted_demand_chatbot"),
        )
        if not cur.fetchone():
            cur.execute(
                f'''
                CREATE TABLE "{target_schema}"."t_predicted_demand_chatbot" (
                    id BIGSERIAL PRIMARY KEY,
                    model_id INTEGER,
                    prediction_date DATE,
                    generated_at TIMESTAMP WITH TIME ZONE,
                    datetime TIMESTAMP WITH TIME ZONE,
                    block INTEGER,
                    predicted_demand DOUBLE PRECISION,
                    horizon_type TEXT,
                    version TEXT
                )
                '''
            )
            conn.commit()
            print(f"[DB] Created table {target_schema}.t_predicted_demand_chatbot")
        else:
            # ensure any pending transactional DDL is visible
            conn.commit()
    else:
        # sqlite
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS t_predicted_demand_chatbot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                prediction_date TEXT,
                generated_at TEXT,
                datetime TEXT,
                block INTEGER,
                predicted_demand REAL,
                horizon_type TEXT,
                version TEXT
            )
            """
        )
        conn.commit()

    # Initialize Chronos pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16
    )

    # Prepare data
    df_train_raw = prepare_train_data(conn, run_date, lby=LBY, lbm=LBM, hrs_end=5)
    df_test_raw = prepare_test_data(conn, run_date, hrs_start=6)

    df_train_in = dedupe_index(df_train_raw).asfreq(FREQ).round(2)
    df_test_in = dedupe_index(df_test_raw).asfreq(FREQ).round(2)

    # Prepare training series (just demand values)
    context = torch.tensor(df_train_in['y'].values, dtype=torch.float32).unsqueeze(0)

    # Run prediction
    with torch.no_grad():
        forecast = pipeline.predict(
            context,
            prediction_length=BLOCKS_PER_DAY,
            num_samples=20
        )

    predictions = forecast[0].numpy()[10, :]

    # Create forecast timestamps
    forecast_index = pd.date_range(
        start=f"{run_date} 00:00:00",
        periods=BLOCKS_PER_DAY,
        freq=FREQ
    )

    # Get actual values if available
    y_true = df_test_in['y'].values[-BLOCKS_PER_DAY:] if len(df_test_in) >= BLOCKS_PER_DAY else None

    metrics = None
    if y_true is not None and len(y_true) == len(predictions):
        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=predictions) * 100
        rmse = root_mean_squared_error(y_true=y_true, y_pred=predictions)
        metrics = {"mape": float(mape), "rmse": float(rmse)}

    # Format result dataframe
    result_df = pd.DataFrame({
        "prediction_date": pd.to_datetime(run_date).date(),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "datetime": forecast_index.astype(str),
        "block": forecast_index.hour * (int(BLOCKS_PER_DAY / 24)) + forecast_index.minute // (int(1440 / BLOCKS_PER_DAY)) + 1,
        "predicted_demand": predictions,
        "model_id": model_id,
        "horizon_type": horizon_type,
        "version": Path(model_path).stem if model_path else "unknown"
    })

    # Around line 376, after table creation, verify it's queryable:
    if cfg.get("type") == "postgresql":
        # Verify table exists in correct schema
        cur.execute(
            "SELECT COUNT(*) FROM lf.t_predicted_demand_chatbot WHERE 1=0"
        )
        print(f"[DB] Verified table lf.t_predicted_demand_chatbot is accessible")

    # Delete existing predictions for the same date/model/horizon (idempotent)
    if cfg.get("type") == "postgresql":
        # use schema-qualified name (other queries reference lf.*)
        schema = "lf"
        fq_table = f'"{schema}"."t_predicted_demand_chatbot"'

        # Ensure the table exists (defensive — covers cases where table was dropped externally)
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
            (schema, "t_predicted_demand_chatbot"),
        )
        if not cur.fetchone():
            # create table from the DataFrame-like schema (safe fallback)
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            cur.execute(
                f'''
                CREATE TABLE "{schema}"."t_predicted_demand_chatbot" (
                    id BIGSERIAL PRIMARY KEY,
                    model_id INTEGER,
                    prediction_date DATE,
                    generated_at TIMESTAMP WITH TIME ZONE,
                    datetime TIMESTAMP WITH TIME ZONE,
                    block INTEGER,
                    predicted_demand DOUBLE PRECISION,
                    horizon_type TEXT,
                    version TEXT
                )
                '''
            )
            conn.commit()
            print(f"[DB] Created table {schema}.t_predicted_demand_chatbot (fallback)")

        # delete existing rows from the correct schema-qualified table
        cur.execute(
            f"DELETE FROM {fq_table} WHERE prediction_date = %s AND model_id = %s AND horizon_type = %s",
            (run_date, model_id, horizon_type),
        )
        conn.commit()

        # Bulk insert using execute_batch for performance
        from psycopg2.extras import execute_batch
        insert_sql = (
            f"INSERT INTO {fq_table}"
            "(model_id, prediction_date, generated_at, datetime, block, predicted_demand, horizon_type, version)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        )
        rows = [
            (
                int(r.model_id),
                pd.to_datetime(r.prediction_date).date(),
                pd.to_datetime(r.generated_at).to_pydatetime(),
                pd.to_datetime(r.datetime).to_pydatetime(),
                int(r.block),
                float(r.predicted_demand),
                r.horizon_type,
                r.version,
            )
            for r in result_df.itertuples(index=False)
        ]
        if rows:
            execute_batch(cur, insert_sql, rows, page_size=200)
            conn.commit()

    else:
        # sqlite path: use pandas.to_sql (sqlite3 connection supported)
        # Use parameterized DELETE for sqlite as well
        cur.execute(
            "DELETE FROM t_predicted_demand_chatbot WHERE prediction_date = ? AND model_id = ? AND horizon_type = ?",
            (str(run_date), model_id, horizon_type),
        )
        conn.commit()

        result_df.to_sql(
            "t_predicted_demand_chatbot",
            conn,
            if_exists="append",
            index=False,
        )
    rows_written = len(result_df)

    # Close DB connection
    try:
        conn.close()
    except Exception:
        pass

    return {
        "ok": True,
        "rows_written": rows_written,
        "prediction_date": run_date,
        "model_id": model_id,
        "metrics": metrics,
    }


# Backwards-compatible CLI behavior
if __name__ == "__main__":
    res = run_and_store_forecast(RUN_DATE)
    if res.get("ok"):
        print(f"✅ Forecast stored for {res['prediction_date']} (rows={res['rows_written']})")
        if res.get("metrics"):
            print(f"   MAPE: {res['metrics'].get('mape'):.2f}%  RMSE: {res['metrics'].get('rmse'):.2f}")
    else:
        print("❌ Forecast run failed:", res.get("error"))