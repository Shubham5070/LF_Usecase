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
MODEL_ID = 202
HORIZON_TYPE = "day_ahead"
MODEL_PATH = "amazon/chronos-t5-small"

FREQ = "15min"
BLOCKS_PER_DAY = 96
MAX_ALLOWED_GAP_HOURS = 36

# Lookback parameters
LBY = 0  # lookback years
LBM = 4  # lookback months

# Feature adjustment parameters
HOLIDAY_DAMPENING = 0.3  # How much to apply holiday adjustment (0-1)
HI_DAMPENING = 0.3  # How much to apply heat index adjustment (0-1)

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
    from db_factory import DatabaseFactory
    cfg = DatabaseFactory.get_config()
    is_postgres = cfg.get("type") == "postgresql"
    
    ip_date = pd.to_datetime(input_date)
    train_start = (ip_date + pd.Timedelta(hours=hrs_end, minutes=45) 
                   - pd.DateOffset(years=lby, months=lbm, days=1))
    train_end = ip_date + pd.Timedelta(hours=hrs_end, minutes=45) - pd.DateOffset(days=1)
    
    # Schema-qualified table names for PostgreSQL
    actual_demand_table = "lf.t_actual_demand" if is_postgres else "t_actual_demand"
    actual_weather_table = "lf.t_actual_weather" if is_postgres else "t_actual_weather"
    holidays_table = "lf.t_holidays" if is_postgres else "t_holidays"
    
    # Load demand data
    query_load = f"""
        SELECT datetime as ds, demand as y
        FROM {actual_demand_table}
        WHERE datetime >= '{train_start}'
        AND datetime <= '{train_end}' 
        ORDER BY datetime
    """
    dfl = pd.read_sql(query_load, conn, parse_dates=["ds"])
    
    # Load weather data
    query_weather = f"""
        SELECT datetime as ds, humidity, temp
        FROM {actual_weather_table}
        WHERE datetime >= '{train_start}' 
        AND datetime <= '{train_end}'
        ORDER BY datetime
    """
    dfw = pd.read_sql(query_weather, conn, parse_dates=["ds"])
    
    # Merge
    dflw = dfl.merge(dfw, on="ds", how="left")
    dflw["date"] = pd.to_datetime(dflw["ds"].dt.date)
    
    # Load holidays
    query_holidays = f"""
        SELECT date, name, normal_holiday, special_day
        FROM {holidays_table}
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
    
    # Set index before creating cyclic features
    dflwh.set_index("ds", inplace=True)
    dflwh.sort_index(inplace=True)
    
    # Create cyclic features
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
    
    df[['normal_holiday', 'special_day', 'is_weekend',
        'is_day_before_holiday', 'is_day_after_holiday',
        'nh_dow_interaction', 'sd_dow_interaction']] = df[[
        'normal_holiday', 'special_day', 'is_weekend',
        'is_day_before_holiday', 'is_day_after_holiday',
        'nh_dow_interaction', 'sd_dow_interaction']].astype(int)
    
    return df

def prepare_test_data(conn, input_date, hrs_start=6):
    from db_factory import DatabaseFactory
    cfg = DatabaseFactory.get_config()
    is_postgres = cfg.get("type") == "postgresql"
    
    ip_date = pd.to_datetime(input_date)
    test_start = ip_date + pd.Timedelta(hours=hrs_start) - pd.DateOffset(days=1)
    test_end = ip_date + pd.Timedelta(hours=23, minutes=45)
    
    # Schema-qualified table names for PostgreSQL
    actual_demand_table = "lf.t_actual_demand" if is_postgres else "t_actual_demand"
    forecasted_weather_table = "lf.t_forecasted_weather" if is_postgres else "t_forecasted_weather"
    holidays_table = "lf.t_holidays" if is_postgres else "t_holidays"
    
    # Load actual demand
    query_load = f"""
        SELECT datetime as ds, demand as y
        FROM {actual_demand_table}
        WHERE datetime >= '{test_start}'
        AND datetime <= '{test_end}' 
        ORDER BY datetime
    """
    dfl = pd.read_sql(query_load, conn, parse_dates=["ds"])
    
    # Load forecasted weather
    query_weather = f"""
        SELECT datetime as ds, date, humidity, temp
        FROM {forecasted_weather_table}
        WHERE datetime >= '{test_start}' 
        AND datetime <= '{test_end}'
        ORDER BY datetime
    """
    dfw = pd.read_sql(query_weather, conn, parse_dates=["ds", "date"])
    
    # Merge
    dflw = dfl.merge(dfw, on="ds", how="left")
    dflw["date"] = pd.to_datetime(dflw["ds"].dt.date)
    
    # Load holidays
    query_holidays = f"""
        SELECT date, name, normal_holiday, special_day
        FROM {holidays_table}
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
    
    # Create cyclic features
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

def apply_feature_adjustments(predictions, df_train, df_test_features, 
                              holiday_dampening=0.3, hi_dampening=0.3):
    """
    Apply feature-based adjustments to Chronos predictions.
    
    Args:
        predictions: Raw predictions from Chronos
        df_train: Training dataframe with historical data
        df_test_features: Test features for the forecast period (normal_holiday, hi)
        holiday_dampening: Weight for holiday adjustment (0-1)
        hi_dampening: Weight for heat index adjustment (0-1)
    
    Returns:
        Adjusted predictions
    """
    adjusted_predictions = predictions.copy()
    
    # 1. Calculate holiday impact
    train_holiday = df_train[df_train['normal_holiday'] == 1]['y']
    train_non_holiday = df_train[df_train['normal_holiday'] == 0]['y']
    
    if len(train_holiday) > 0 and len(train_non_holiday) > 0:
        holiday_ratio = train_holiday.mean() / train_non_holiday.mean()
        print(f"   Holiday ratio: {holiday_ratio:.3f}")
    else:
        holiday_ratio = 1.0
        print("   No holiday data for adjustment")
    
    # 2. Calculate heat index impact using quantile-based approach
    hi_adjustment = np.ones(len(predictions))
    
    if 'hi' in df_train.columns and df_train['hi'].notna().sum() > 0:
        try:
            # Create heat index quintiles from training data
            df_train_copy = df_train.copy()
            df_train_copy['hi_quintile'] = pd.qcut(
                df_train_copy['hi'], 
                q=5, 
                labels=False, 
                duplicates='drop'
            )
            
            # Calculate mean demand by quintile
            hi_impact = df_train_copy.groupby('hi_quintile')['y'].mean()
            overall_mean = df_train_copy['y'].mean()
            
            # Map forecast hi to quintiles
            df_test_copy = df_test_features.copy()
            df_test_copy['hi_quintile'] = pd.qcut(
                df_test_copy['hi'], 
                q=5, 
                labels=False, 
                duplicates='drop'
            )
            
            # Calculate adjustment factor for each forecast point
            for i in range(len(predictions)):
                if i < len(df_test_copy):
                    quintile = df_test_copy.iloc[i]['hi_quintile']
                    if quintile is not None and quintile in hi_impact.index:
                        hi_adjustment[i] = hi_impact[quintile] / overall_mean
            
            print(f"   HI adjustment range: {hi_adjustment.min():.3f} - {hi_adjustment.max():.3f}")
        except Exception as e:
            print(f"   Warning: Could not calculate HI adjustment: {e}")
            hi_adjustment = np.ones(len(predictions))
    else:
        print("   No HI data for adjustment")
    
    # 3. Apply combined adjustments
    for i in range(len(predictions)):
        adjustment_factor = 1.0
        
        # Apply holiday adjustment
        if i < len(df_test_features) and df_test_features.iloc[i]['normal_holiday'] == 1:
            adjustment_factor *= (1 + holiday_dampening * (holiday_ratio - 1))
        
        # Apply heat index adjustment
        adjustment_factor *= (1 + hi_dampening * (hi_adjustment[i] - 1))
        
        adjusted_predictions[i] = predictions[i] * adjustment_factor
    
    return adjusted_predictions

def run_and_store_forecast(run_date: str, *, db_path: str | None = None, model_id: int | None = None, 
                           model_path: str | None = None, horizon_type: str | None = None,
                           use_feature_adjustment: bool = True) -> dict:
    """Run forecast for `run_date` and store results in DB.

    Returns:
        dict: {ok: bool, rows_written: int, prediction_date: str, model_id: int, 
               metrics: {mape, rmse, mape_adjusted, rmse_adjusted} | None, error: str | None}
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
        target_schema = "lf"
        cur.execute('CREATE SCHEMA IF NOT EXISTS "lf"')
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
            conn.commit()
    else:
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
    print(f"\n[Model] Loading Chronos from {model_path} on {device}...")

    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32
    )

    # Prepare data
    print(f"[Data] Preparing training data for {run_date}...")
    df_train_raw = prepare_train_data(conn, run_date, lby=LBY, lbm=LBM, hrs_end=5)
    df_test_raw = prepare_test_data(conn, run_date, hrs_start=6)

    df_train_in = dedupe_index(df_train_raw).asfreq(FREQ).round(2)
    df_test_in = dedupe_index(df_test_raw).asfreq(FREQ).round(2)
    
    print(f"   Train size: {len(df_train_in)} records")
    print(f"   Test size: {len(df_test_in)} records")

    # Prepare training series (just demand values for Chronos)
    context = torch.tensor(df_train_in['y'].values, dtype=torch.float32).unsqueeze(0)

    # Run prediction
    print(f"[Forecast] Running Chronos prediction for {BLOCKS_PER_DAY} blocks...")
    with torch.no_grad():
        forecast = pipeline.predict(
            context,
            prediction_length=BLOCKS_PER_DAY,
            num_samples=5
        )

    predictions_raw = forecast[0].median(dim=0).values.numpy()
    
    # Apply feature-based adjustments if enabled
    if use_feature_adjustment:
        print(f"[Adjustment] Applying feature adjustments (normal_holiday, hi)...")
        
        # Get test features for the forecast period
        forecast_features = df_test_in.iloc[-BLOCKS_PER_DAY:][['normal_holiday', 'hi']].copy()
        
        predictions = apply_feature_adjustments(
            predictions_raw, 
            df_train_in, 
            forecast_features,
            holiday_dampening=HOLIDAY_DAMPENING,
            hi_dampening=HI_DAMPENING
        )
    else:
        predictions = predictions_raw
        print(f"[Adjustment] Feature adjustment disabled, using raw predictions")

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
        # Calculate metrics for adjusted predictions
        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=predictions) * 100
        rmse = root_mean_squared_error(y_true=y_true, y_pred=predictions)
        
        # Also calculate metrics for raw predictions for comparison
        mape_raw = mean_absolute_percentage_error(y_true=y_true, y_pred=predictions_raw) * 100
        rmse_raw = root_mean_squared_error(y_true=y_true, y_pred=predictions_raw)
        
        metrics = {
            "mape": float(mape), 
            "rmse": float(rmse),
            "mape_raw": float(mape_raw),
            "rmse_raw": float(rmse_raw),
            "improvement_mape": float(mape_raw - mape),
            "improvement_rmse": float(rmse_raw - rmse)
        }

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

    # Verify and insert data
    if cfg.get("type") == "postgresql":
        cur.execute(
            "SELECT COUNT(*) FROM lf.t_predicted_demand_chatbot WHERE 1=0"
        )
        
        schema = "lf"
        fq_table = f'"{schema}"."t_predicted_demand_chatbot"'

        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
            (schema, "t_predicted_demand_chatbot"),
        )
        if not cur.fetchone():
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

        cur.execute(
            f"DELETE FROM {fq_table} WHERE prediction_date = %s AND model_id = %s AND horizon_type = %s",
            (run_date, model_id, horizon_type),
        )
        conn.commit()

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


def backtest_last_5_days():
    """
    Backtest forecast for last 5 days of December 2025 (Dec 26-31).
    For each day, use data up to 05:45 AM of the previous day.
    """
    forecast_dates = [
        "2025-12-26",
        "2025-12-27", 
        "2025-12-28",
        "2025-12-29",
        "2025-12-30",
        "2025-12-31"
    ]
    
    print("=" * 80)
    print("BACKTESTING: Last 5 Days of December 2025")
    print("=" * 80)
    print(f"Forecast dates: {', '.join(forecast_dates)}")
    print(f"Features: normal_holiday, hi (heat index)")
    print(f"Model: {MODEL_PATH}")
    print("=" * 80)
    
    all_results = []
    
    for date in forecast_dates:
        print(f"\n{'='*80}")
        print(f"FORECASTING FOR: {date}")
        print(f"{'='*80}")
        
        try:
            result = run_and_store_forecast(
                date,
                model_id=MODEL_ID,
                model_path=MODEL_PATH,
                horizon_type=HORIZON_TYPE,
                use_feature_adjustment=True
            )
            
            if result.get("ok"):
                print(f"\n✅ Forecast completed for {date}")
                print(f"   Rows written: {result['rows_written']}")
                
                if result.get("metrics"):
                    m = result['metrics']
                    print(f"\n   📊 METRICS:")
                    print(f"      Raw Chronos  -> MAPE: {m.get('mape_raw', 0):.2f}%  RMSE: {m.get('rmse_raw', 0):.2f}")
                    print(f"      With Features-> MAPE: {m.get('mape', 0):.2f}%  RMSE: {m.get('rmse', 0):.2f}")
                    print(f"      Improvement  -> MAPE: {m.get('improvement_mape', 0):.2f}%  RMSE: {m.get('improvement_rmse', 0):.2f}")
                
                all_results.append({
                    'date': date,
                    'success': True,
                    **result
                })
            else:
                print(f"❌ Forecast failed for {date}: {result.get('error')}")
                all_results.append({
                    'date': date,
                    'success': False,
                    'error': result.get('error')
                })
                
        except Exception as e:
            print(f"❌ Exception during forecast for {date}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'date': date,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in all_results if r.get('success')]
    failed = [r for r in all_results if not r.get('success')]
    
    print(f"Total forecasts: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n📈 AVERAGE METRICS (across successful forecasts):")
        avg_mape = np.mean([r['metrics']['mape'] for r in successful if r.get('metrics')])
        avg_rmse = np.mean([r['metrics']['rmse'] for r in successful if r.get('metrics')])
        avg_mape_raw = np.mean([r['metrics']['mape_raw'] for r in successful if r.get('metrics')])
        avg_rmse_raw = np.mean([r['metrics']['rmse_raw'] for r in successful if r.get('metrics')])
        
        print(f"   Raw Chronos  -> MAPE: {avg_mape_raw:.2f}%  RMSE: {avg_rmse_raw:.2f}")
        print(f"   With Features-> MAPE: {avg_mape:.2f}%  RMSE: {avg_rmse:.2f}")
        print(f"   Improvement  -> MAPE: {avg_mape_raw - avg_mape:.2f}%  RMSE: {avg_rmse_raw - avg_rmse:.2f}")
    
    if failed:
        print(f"\n❌ Failed forecasts:")
        for r in failed:
            print(f"   {r['date']}: {r.get('error', 'Unknown error')}")
    
    print("=" * 80)
    
    return all_results


# Backwards-compatible CLI behavior
if __name__ == "__main__":
    # Run backtest for last 5 days
    results = backtest_last_5_days()