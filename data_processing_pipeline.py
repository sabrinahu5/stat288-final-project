import os, time, datetime as dt
import requests, numpy as np, pandas as pd

# --------------------------------------------------------------------- #
# 0. CONFIGURATION –– EDIT THESE FOUR LINES
# --------------------------------------------------------------------- #
BBOX       = [76.75, 28.35, 77.45, 28.95]      # xmin,ymin,xmax,ymax   (Delhi)
DATE_FROM  = "2018-01-01"
DATE_TO    = "2024-12-31"
API_KEY    = os.getenv("OPENAQ_KEY")           # export OPENAQ_KEY first
# --------------------------------------------------------------------- #

OAQ  = "https://api.openaq.org/v3"
HEAD = {"X-API-Key": API_KEY} if API_KEY else {}
LIM  = 100                          # page cap

# --------------------------------------------------------------------- #
# 1. DISCOVER SENSORS INSIDE BOUNDING BOX
# --------------------------------------------------------------------- #
def get_pm25_sensors(bbox):
    rows, page = [], 1
    while True:
        params = {
            "parameters_id": 2,                       # PM2.5
            "bbox"       : ",".join(map(str, bbox)),
            "limit"      : LIM,
            "page"       : page }
        r = requests.get(f"{OAQ}/sensors", params=params, headers=HEAD).json()
        rows.extend(r["results"])
        if page * LIM >= r["meta"]["found"]:
            break
        page += 1
        time.sleep(0.3)
    df = pd.json_normalize(rows)
    return df[["id", "locationId", "coordinates.latitude", "coordinates.longitude"]].rename(
        columns={"id": "sid", "coordinates.latitude": "latitude",
                 "coordinates.longitude": "longitude"})

# --------------------------------------------------------------------- #
# 2. DOWNLOAD RAW MEASUREMENTS PER SENSOR (PAGINATED)
# --------------------------------------------------------------------- #
def measurements_for_sensor(sid, start, stop):
    rows, page = [], 1
    while True:
        params = {
            "date_from": start,
            "date_to"  : stop,
            "limit"    : LIM,
            "page"     : page }
        url = f"{OAQ}/sensors/{sid}/measurements"
        r   = requests.get(url, params=params, headers=HEAD).json()
        rows.extend(r["results"])
        if page * LIM >= r["meta"]["found"]:
            break
        page += 1
        time.sleep(0.2)
    return rows

def pull_city_measurements(sensor_df, start, stop):
    all_rows = []
    for sid in sensor_df["sid"]:
        all_rows.extend(measurements_for_sensor(sid, start, stop))
    df = pd.json_normalize(all_rows)
    # keep only the essentials
    keep = ["sensorId", "value", "date.utc", "coordinates.latitude",
            "coordinates.longitude"]
    return df[keep].rename(columns={"sensorId": "sid", "value": "pm25"})

# --------------------------------------------------------------------- #
# 3. OPTIONAL: LIGHT OUTLIER CLEANING (24‑h Hampel FILTER)
# --------------------------------------------------------------------- #
def hampel_24h(series, k=5, window=24):
    med = series.rolling(window, center=True).median()
    mad = series.rolling(window, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    rzs = 0.6745 * np.abs(series - med) / mad
    return series[(rzs < k) | (mad == 0)]

def clean_df(df):
    df["dt"] = pd.to_datetime(df["date.utc"])
    df = df.dropna(subset=["pm25"])
    df = (df.sort_values(["sid", "dt"])
            .groupby("sid", group_keys=False)
            .apply(lambda g: g.assign(pm25=hampel_24h(g.pm25))))
    return df.dropna(subset=["pm25"])

# --------------------------------------------------------------------- #
# 4. AGGREGATE TO WEEKLY MEANS (RETAIN LAT/LON)
# --------------------------------------------------------------------- #
def weekly_means(df):
    df["week"] = df["dt"].dt.to_period("W-MON").dt.start_time
    wk = (df.groupby(["sid", "week"], as_index=False)
            .agg(pm25=("pm25", "mean"),
                 latitude=("coordinates.latitude", "first"),
                 longitude=("coordinates.longitude", "first"),
                 n_raw=("pm25", "size")))
    # require ≥24 hourly readings in that week
    wk = wk[wk.n_raw >= 24].drop(columns="n_raw")
    return wk[["sid", "week", "pm25", "latitude", "longitude"]]

# --------------------------------------------------------------------- #
# 5. MAIN ORCHESTRATION
# --------------------------------------------------------------------- #
def build_weekly_dataframe(bbox=BBOX, date_from=DATE_FROM, date_to=DATE_TO):
    sensors = get_pm25_sensors(bbox)
    raw_df  = pull_city_measurements(sensors, date_from, date_to)
    clean   = clean_df(raw_df)
    weekly  = weekly_means(clean)
    return weekly.reset_index(drop=True)

if __name__ == "__main__":
    df_weekly = build_weekly_dataframe()
    print(df_weekly.head())
    # df_weekly.to_csv("pm25_weekly.csv", index=False)