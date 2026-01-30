"""functions.py used to define regularlly used functions"""
import os
from datetime import time
import math
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

def db_connection():
    """Function to Establish DB Connection and Return SQLAlchemy Engine"""
    load_dotenv()
    host = os.getenv("host")
    database = os.getenv("database")
    user = os.getenv("user")
    password = os.getenv("password")
    port = os.getenv("port")
    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    )
    with engine.connect() as c:
        # If we reach here, the connection was successful!
        print("Successfully connected to the database!")
    return engine

def get_time_from_block(block_number: int) -> time:
    """
    Given a 1-based 15-minute block number (1..96), return the corresponding time of day.
    Block 1 -> 00:00, Block 2 -> 00:15, ..., Block 96 -> 23:45.
    """
    if not isinstance(block_number, int):
        raise TypeError("block_number must be an integer")
    if not 1 <= block_number <= 96:
        raise ValueError("block_number must be in [1, 96] for 15-minute blocks in a day")

    total_minutes = (block_number - 1) * 15
    hour = total_minutes // 60
    minute = total_minutes % 60
    return time(hour, minute)

def get_block_number(t:time):
    """Returns Time taking Block Number as Input"""
    hour, minute = t.hour, t.minute
    return (hour * 4 + minute // 15) + 1

def calculate_daily_metrics(date, model_id, engine):
    """
    Calculate daily MAPE and RMSE for the specified date and model_id,
    and insert the results into the t_metrics table.

    Parameters:
    date (str): Date in 'YYYY-MM-DD' format for which to calculate metrics.
    model_id (int): The ID of the model to evaluate.
    engine: SQLAlchemy engine for the database connection.
    """
    date = pd.to_datetime(date).strftime("%Y-%m-%d")
    # Query actual and forecasted data for the given date and model_id
    query_actual = f"""
        SELECT block, demand
        FROM "AEML".t_actual_demand
        WHERE date = '{date}' order by block
    """

    query_forecasted = f"""
        SELECT block, forecasted_demand
        FROM "AEML".t_forecasted_demand
        WHERE date = '{date}' AND model_id = {model_id} order by block
    """

    # Load data into DataFrames
    df_actual = pd.read_sql(query_actual, con=engine)
    df_forecasted = pd.read_sql(query_forecasted, con=engine)

    # Merge actual and forecasted data on datetime
    df_merged = pd.merge(df_actual, df_forecasted, on="block", how="inner")

    # Ensure there are no missing values after the merge
    if df_merged.empty:
        raise ValueError(
            f"No matching data found for date {date} and model_id {model_id}."
        )

    # Extract actual and predicted values
    y_actual = df_merged["demand"].values
    y_predicted = df_merged["forecasted_demand"].values

    # Calculate MAPE
    mape = (
        mean_absolute_percentage_error(y_actual, y_predicted) * 100
    )  # Convert to percentage

    # Calculate RMSE
    rmse = root_mean_squared_error(y_actual, y_predicted)
    # Calculate SMAPE
    smape = 100 * np.mean(
        np.abs(y_predicted - y_actual)
        / (
            (np.abs(y_actual) + np.abs(y_predicted)) / 2
        )  # Add small epsilon to avoid division by zero
    )

    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame(
        [
            {
                "date": date,
                "mape": np.round(mape, 2),
                "rmse": np.round(rmse, 2),
                "smape": np.round(smape, 2),
                "model_id": model_id,
            }
        ]
    )

    # Write the DataFrame to the database
    metrics_df.to_sql(
        "t_metrics", con=engine, schema="AEML", if_exists="append", index=False
    )

    print(f"Metrics for {date} and model_id {model_id} inserted into t_metrics.")
    return metrics_df

def fahrenheit_to_celsius(temp_fahrenheit):
    """converst temp from to fahrenheit celsius"""
    return (temp_fahrenheit - 32) * 5 / 9

def celsius_to_fahrenheit(temp_celsius):
    """converst temp from celsius to fahrenheit """
    return temp_celsius * 9/5 + 32

def heat_index_from_celsius(t_celsius, rh):
    """Functions retursn Heat Index as per NWS Heat Index calculation Formula"""
    t = celsius_to_fahrenheit(t_celsius)

    # Simplified formula for heat index when hi < 80°F
    simple_hi = 0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094))

    # Average with the temperature itself
    hi = 0.5 * (simple_hi + t)

    # Use the full Rothfusz regression equation if hi >= 80°F
    if hi >= 80:
        hi = (
            -42.379
            + (2.04901523 * t)
            + (10.14333127 * rh)
            - (0.22475541 * t * rh)
            - (0.00683783 * t * t)
            - (0.05481717 * rh * rh)
            + (0.00122874 * t * t * rh)
            + (0.00085282 * t * rh * rh)
            - (0.00000199 * t * t * rh * rh)
        )

        # Low humidity adjustment (rh < 13% and 80°F <= t <= 112°F)
        if rh < 13 and 80 <= t <= 112:
            adjustment = ((13 - rh) / 4) * math.sqrt((17 - abs(t - 95)) / 17)
            hi -= adjustment

        # High humidity adjustment (rh > 85% and 80°F <= t <= 87°F)
        elif rh > 85 and 80 <= t <= 87:
            adjustment = ((rh - 85) / 10) * ((87 - t) / 5)
            hi += adjustment

    # Convert the result back to Celsius
    return fahrenheit_to_celsius(hi)

def create_cyclic_features(df):
    """Function to create cyclic features"""
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
    df['is_weekend'] = df['day_of_week'].isin([6,7]).astype(int)
    df['is_day_of_week'] = df['day_of_week'].isin([1,2,3,4,5]).astype(int)
    return df

def dedupe_index(df, keep="first"):
    df = df.sort_index()
    return df[~df.index.duplicated(keep=keep)]
