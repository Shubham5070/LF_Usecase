"""Data Pipeline Functions"""
import pandas as pd
from functions import create_cyclic_features, heat_index_from_celsius
def data_prep_train(input_date,engine,lby=3, lbm=0, sp_id=2,hrs_end=5):
    """function to prepare training data"""
    ip_date = pd.to_datetime(input_date)
    # Calculate train_start: 3 years and 1 day earlier from 8:00 AM
    train_start = (
        ip_date
        + pd.Timedelta(hours=hrs_end, minutes=45)
        - pd.DateOffset(years=lby, months=lbm, days=1)
    )
    # Calculate train_end: 1 day earlier from 8:00 AM
    train_end = ip_date + pd.Timedelta(hours=hrs_end, minutes=45) - pd.DateOffset(days=1)
    query_load = f"""
        SELECT datetime as ds, demand as y
        FROM "AEML".t_actual_demand
        WHERE datetime >= '{train_start}'
        AND datetime <= '{train_end}' order by datetime
    """
    dfl = pd.read_sql(query_load, con=engine)
    dfl["ds"] = pd.to_datetime(dfl["ds"])
    query_weather = f"""
        SELECT datetime as ds,humidity, temp
	    FROM "AEML".t_actual_weather
        WHERE datetime >= '{train_start}' 
        AND datetime <= '{train_end}'
        AND sp_id = '{sp_id}' order by datetime
    """
    dfw = pd.read_sql(query_weather, con=engine)
    dfw["ds"] = pd.to_datetime(dfw["ds"])
    dfl["ds"] = pd.to_datetime(dfl["ds"])
    dfw["ds"] = pd.to_datetime(dfw["ds"])
    # dfw['time'] = pd.to_timedelta(dfw['time'])
    dflw = dfl.merge(dfw, on="ds", how="left")
    dflw["date"] = dflw["ds"].dt.date
    dflw["date"] = pd.to_datetime(dflw["date"])
    query_holidays = f"""
        SELECT date, name, normal_holiday, special_day,day_of_week
        FROM "AEML".t_holidays
        WHERE date >= '{train_start.date()}'  
        AND date <= '{train_end.date()}'
        order by date
    """
    dfh = pd.read_sql(query_holidays, con=engine)
    dfh['normal_holiday'] = dfh['normal_holiday'].astype('int64')
    dfh['special_day'] = dfh['special_day'].astype('int64')
    dfh["date"] = pd.to_datetime(dfh["date"])
    dflwh = dflw.merge(dfh, on=["date"], how="left")
    dflwh.fillna(0, inplace=True)
    dflwh["ds"] = pd.to_datetime(dflwh["ds"])
    dflwh["hour"] = dflwh["ds"].dt.hour + 1
    dflwh["is_day_before_holiday"] = (
        (dflwh["date"] + pd.Timedelta(days=1)).isin(dfh["date"]).astype(int)
    )
    dflwh["is_day_after_holiday"] = (
        (dflwh["date"] - pd.Timedelta(days=1)).isin(dfh["date"]).astype(int)
    )
    dflwh["nh_dow_interaction"] = dflwh["normal_holiday"] * dflwh["day_of_week"]
    dflwh["sd_dow_interaction"] = dflwh["special_day"] * dflwh["day_of_week"]
    dflwh["hi"] = dflwh.apply(
        lambda row: heat_index_from_celsius(row["temp"], row["humidity"]), axis=1
    )
    dflwh.set_index("ds", inplace=True)
    dflwh.sort_index(inplace=True)
    df = create_cyclic_features(dflwh)
    df[['normal_holiday','special_day','is_weekend',
        'is_day_before_holiday','is_day_after_holiday','nh_dow_interaction', 'sd_dow_interaction']] = df[['normal_holiday','special_day','is_weekend',
    'is_day_before_holiday','is_day_after_holiday',
    'nh_dow_interaction', 'sd_dow_interaction']].astype(int)
    return df
def data_prep_test(input_date,engine, fcwt='t_forecasted_weather', sp_id=4,hrs_start=6):
    """Fuction to Prepare Test Data"""
    ip_date = pd.to_datetime(input_date)
    test_start = ip_date + pd.Timedelta(hours=hrs_start) - pd.DateOffset(days=1)
    test_end = ip_date + pd.Timedelta(hours=23,minutes=45) 
    dfl = pd.DataFrame()
    dfl['ds'] = pd.date_range(start=test_start,end=test_end,freq='15min')
    query_load = f"""
        SELECT datetime as ds, demand as y
        FROM "AEML".t_actual_demand
        WHERE datetime >= '{test_start}'
        AND datetime <= '{test_end}' order by datetime
    """
    dfl = pd.read_sql(query_load, con=engine)
    dfl["ds"] = pd.to_datetime(dfl["ds"])
    query_weather = f"""
        SELECT datetime as ds,date,humidity, temp
        FROM "AEML".{fcwt}
        WHERE datetime >= '{test_start}' 
        AND datetime <= '{test_end}'
        AND sp_id = '{sp_id}'
        order by datetime
        """
    dfw = pd.read_sql(query_weather, con=engine)
    dfw["date"] = pd.to_datetime(dfw["date"])
    dfw["ds"] = pd.to_datetime(dfw["ds"])
    dflw = dfl.merge(dfw, on="ds", how="left")
    dflw["date"] = dflw["ds"].dt.date
    dflw["date"] = pd.to_datetime(dflw["date"])
    query_holidays = f"""
    SELECT date, name, normal_holiday, special_day,day_of_week
    FROM "AEML".t_holidays
    WHERE date >= '{test_start.date()}'  
    AND date <= '{test_end.date()}'
    order by date
        """
    dfh = pd.read_sql(query_holidays, con=engine)
    dfh['normal_holiday'] = dfh['normal_holiday'].astype('int64')
    dfh['special_day'] = dfh['special_day'].astype('int64')
    dfh["date"] = pd.to_datetime(dfh["date"])
    dflwh = dflw.merge(dfh, on=["date"], how="left")
    dflwh.fillna(0, inplace=True)
    dflwh["ds"] = pd.to_datetime(dflwh["ds"])
    dflwh["hour"] = dflwh["ds"].dt.hour + 1
    dflwh["is_day_before_holiday"] = (
        (dflwh["date"] + pd.Timedelta(days=1)).isin(dfh["date"]).astype(int)
    )
    dflwh["is_day_after_holiday"] = (
        (dflwh["date"] - pd.Timedelta(days=1)).isin(dfh["date"]).astype(int)
    )
    dflwh["nh_dow_interaction"] = dflwh["normal_holiday"] * dflwh["day_of_week"]
    dflwh["sd_dow_interaction"] = dflwh["special_day"] * dflwh["day_of_week"]

    dflwh["hi"] = dflwh.apply(
        lambda row: heat_index_from_celsius(row["temp"], row["humidity"]), axis=1
    )
    dflwh.set_index("ds", inplace=True)
    dflwh.sort_index(inplace=True)
    df = create_cyclic_features(dflwh)
    df[['normal_holiday','special_day','is_day_before_holiday','is_weekend',
        'is_day_after_holiday','nh_dow_interaction', 'sd_dow_interaction']] = df[['normal_holiday','special_day','is_weekend',
    'is_day_before_holiday','is_day_after_holiday','nh_dow_interaction', 'sd_dow_interaction']].astype(int)
    return df
def data_prep_test_id(input_date,engine, fcwt='t_forecasted_weather', sp_id=4,hrs_start=6):
    """Fuction to Prepare Test Data"""
    ip_date = pd.to_datetime(input_date)
    test_start = ip_date + pd.Timedelta(hours=hrs_start) - pd.DateOffset(days=0)
    test_end = ip_date + pd.Timedelta(hours=23,minutes=45) 
    dfl = pd.DataFrame()
    # dfl['ds'] = pd.date_range(start=test_start,end=test_end,freq='15min')
    query_load = f"""
        SELECT datetime as ds, demand as y
        FROM "AEML".t_actual_demand
        WHERE datetime >= '{test_start}'
        AND datetime <= '{test_end}' order by datetime
    """
    dfl = pd.read_sql(query_load, con=engine)
    dfl["ds"] = pd.to_datetime(dfl["ds"])
    query_weather = f"""
        SELECT datetime as ds,date,humidity, temp
        FROM "AEML".{fcwt}
        WHERE datetime >= '{test_start}' 
        AND datetime <= '{test_end}'
        AND sp_id = '{sp_id}'
        order by datetime
        """
    dfw = pd.read_sql(query_weather, con=engine)
    dfw["date"] = pd.to_datetime(dfw["date"])
    dfw["ds"] = pd.to_datetime(dfw["ds"])
    dflw = dfl.merge(dfw, on="ds", how="left")
    dflw["date"] = dflw["ds"].dt.date
    dflw["date"] = pd.to_datetime(dflw["date"])
    query_holidays = f"""
    SELECT date, name, normal_holiday, special_day,day_of_week
    FROM "AEML".t_holidays
    WHERE date >= '{test_start.date()}'  
    AND date <= '{test_end.date()}'
    order by date
        """
    dfh = pd.read_sql(query_holidays, con=engine)
    dfh['normal_holiday'] = dfh['normal_holiday'].astype('int64')
    dfh['special_day'] = dfh['special_day'].astype('int64')
    dfh["date"] = pd.to_datetime(dfh["date"])
    dflwh = dflw.merge(dfh, on=["date"], how="left")
    dflwh.fillna(0, inplace=True)
    dflwh["ds"] = pd.to_datetime(dflwh["ds"])
    dflwh["hour"] = dflwh["ds"].dt.hour + 1
    dflwh["is_day_before_holiday"] = (
        (dflwh["date"] + pd.Timedelta(days=1)).isin(dfh["date"]).astype(int)
    )
    dflwh["is_day_after_holiday"] = (
        (dflwh["date"] - pd.Timedelta(days=1)).isin(dfh["date"]).astype(int)
    )
    dflwh["nh_dow_interaction"] = dflwh["normal_holiday"] * dflwh["day_of_week"]
    dflwh["sd_dow_interaction"] = dflwh["special_day"] * dflwh["day_of_week"]

    dflwh["hi"] = dflwh.apply(
        lambda row: heat_index_from_celsius(row["temp"], row["humidity"]), axis=1
    )
    dflwh.set_index("ds", inplace=True)
    dflwh.sort_index(inplace=True)
    df = create_cyclic_features(dflwh)
    df[['normal_holiday','special_day','is_day_before_holiday','is_weekend',
        'is_day_after_holiday','nh_dow_interaction', 'sd_dow_interaction']] = df[['normal_holiday','special_day','is_weekend',
    'is_day_before_holiday','is_day_after_holiday','nh_dow_interaction', 'sd_dow_interaction']].astype(int)
    return df

    ''' Test Data Preparation for Generated Forecast Model'''
    ip_date = pd.to_datetime(input_date)
    test_start = ip_date + pd.Timedelta(hours=hrs_start) - pd.DateOffset(days=1)
    test_end = ip_date + pd.Timedelta(hours=23,minutes=45) 
    dfw = pd.DataFrame()
    dfw['datetime'] = pd.date_range(start=test_start,end=test_end,freq='15min')
    dfw['date'] = dfw['datetime'].dt.date
    dfw['humidity'] = 0
    dfw['temp'] =0
    dfw['date'] = pd.to_datetime(dfw['date'])
    query_holidays = f'''
    SELECT date, name, normal_holiday, special_day,day_of_week
    FROM "AEML".t_holidays
    WHERE date >= '{test_start.date()}'  
    AND date <= '{test_end.date()}'
    order by date
        '''
    dfh = pd.read_sql(query_holidays,con=engine)
    dfh['date'] = pd.to_datetime(dfh['date'])
    dfwh = dfw.merge(dfh,on=['date'],how='left')
    dfwh.fillna(0,inplace=True)
    dfwh['datetime'] = pd.to_datetime(dfwh['datetime'])
    dfwh['hour'] = dfwh['datetime'].dt.hour + 1
    dfwh['is_day_before_holiday'] = (dfwh['date'] + pd.Timedelta(days=1)).isin(dfh['date']).astype(int)
    dfwh['is_day_after_holiday'] = (dfwh['date'] - pd.Timedelta(days=1)).isin(dfh['date']).astype(int)
    dfwh['nh_dow_interaction'] = dfwh['normal_holiday'] * dfwh['day_of_week']
    dfwh['sd_dow_interaction'] = dfwh['special_day'] * dfwh['day_of_week']
    dfwh['hi'] = dfwh.apply(lambda row: heat_index_from_celsius(row['temp'], row['humidity']), axis=1)
    dfwh.set_index('datetime', inplace=True)
    dfwh.sort_index(inplace=True)
    df = create_cyclic_features(dfwh)
    df['holiday_hour'] = df['normal_holiday'] * df['hour']
    df['special_day_hour'] = df['special_day'] * df['hour']
    df['minute_normal_holiday'] = df['minute'] * df['normal_holiday']
    df['minute_special_day'] = df['minute'] * df['special_day']
    df['dow_sin_normal_holiday'] = df['day_of_week_sin'] * df['normal_holiday']
    df['dow_cos_normal_holiday'] = df['day_of_week_cos'] * df['normal_holiday']
    df['dow_sin_special_day'] = df['day_of_week_sin'] * df['special_day']
    df['dow_cos_special_day'] = df['day_of_week_cos'] * df['special_day']
    df['minute_sin_normal_holiday'] = df['minute_sin'] * df['normal_holiday']
    df['minute_cos_normal_holiday'] = df['minute_cos'] * df['normal_holiday']
    df['minute_sin_special_day'] = df['minute_sin'] * df['special_day']
    df['minute_cos_special_day'] = df['minute_cos'] * df['special_day']
    return df