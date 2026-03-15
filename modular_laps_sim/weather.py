from datetime import timedelta
from typing import Tuple
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache


class WeatherService:
    def __init__(self, track):
        self.track = track
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = openmeteo_requests.Client(
                session=requests_cache.CachedSession(cache_name="openmeteo_cache", backend="memory")
            )
        return self._client

    def fetch_weather_data(self, race_config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        url = "https://api.open-meteo.com/v1/forecast"
        local_today = pd.Timestamp.now(tz=self.track.timezone).date()
        params = {
            "latitude": self.track.latitude,
            "longitude": self.track.longitude,
            "hourly": ["shortwave_radiation_instant", "cloud_cover"],
            "timezone": self.track.timezone,
            "start_date": str(local_today),
            "end_date": str(local_today + timedelta(days=15)),
        }

        responses = self.client.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()

        ghi_values = hourly.Variables(0).ValuesAsNumpy()
        cloud_values = hourly.Variables(1).ValuesAsNumpy()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "ghi": ghi_values,
            "cloud_cover": cloud_values,
        }

        df = pd.DataFrame(data=hourly_data).set_index("date").tz_convert(self.track.timezone)

        interval = pd.Timedelta(seconds=hourly.Interval())
        selected = None
        for race_date in pd.Index(df.index.date).unique():
            start_dt = pd.Timestamp(race_date, tz=self.track.timezone) + pd.Timedelta(
                hours=race_config.start_time_hour
            )
            end_dt = pd.Timestamp(race_date, tz=self.track.timezone) + pd.Timedelta(
                hours=race_config.end_time_hour
            )
            race_window = df.loc[(df.index >= start_dt) & (df.index < end_dt)].copy()
            has_full_coverage = (
                not race_window.empty
                and race_window.index.min() <= start_dt
                and race_window.index.max() >= (end_dt - interval)
            )
            if has_full_coverage:
                selected = (start_dt, end_dt, race_window)
                break

        if selected is None:
            raise ValueError("No weather data found for a complete race time window")

        start_dt, end_dt, race_window = selected
        minute_index = pd.date_range(start=start_dt, end=end_dt, freq="min", inclusive="left")
        minute_df = race_window.reindex(minute_index)
        minute_df["ghi"] = minute_df["ghi"].interpolate(method="linear", limit_direction="both")
        minute_df["cloud_cover"] = minute_df["cloud_cover"].interpolate(
            method="linear", limit_direction="both"
        )

        n_points = len(minute_df)
        time_minutes = np.arange(n_points) + race_config.start_time_hour * 60
        ghi_data = np.clip(minute_df["ghi"].to_numpy(), 0, None)
        cloud_data = minute_df["cloud_cover"].to_numpy() / 100.0

        return time_minutes, ghi_data, cloud_data

    def generate_synthetic_data(self, race_config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start_min = int(race_config.start_time_hour * 60)
        end_min = int(race_config.end_time_hour * 60)
        time_minutes = np.arange(start_min, end_min)

        t_peak = 13 * 60
        peak_ghi = 900
        a = peak_ghi / ((t_peak - start_min) ** 2)
        ghi_data = np.maximum(0, -a * (time_minutes - t_peak) ** 2 + peak_ghi)

        noise = np.clip(np.random.normal(1.0, 0.1, len(ghi_data)), 0.7, 1.2)
        smooth_noise = np.convolve(noise, np.ones(15) / 15, mode="same")
        ghi_data = ghi_data * smooth_noise

        cloud_data = np.clip(0.3 + 0.3 * np.random.randn(len(time_minutes)), 0, 1)
        cloud_data = np.convolve(cloud_data, np.ones(30) / 30, mode="same")

        return time_minutes, ghi_data, cloud_data
