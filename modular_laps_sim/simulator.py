from config import SimulationResults
from physics import PhysicsEngine
from weather import WeatherService
from controllers import build_strategy
import numpy as np


class LapsRaceSimulator:
    def __init__(self, track, car, race, use_api_weather=True):
        self.track = track
        self.car = car
        self.race = race
        self.use_api_weather = use_api_weather

        self.physics = PhysicsEngine(self.car)
        self.weather = WeatherService(self.track)
        self.strategy = build_strategy(self.race.strategy, self.race)

    def run(self) -> SimulationResults:
        if self.use_api_weather:
            try:
                time_minutes, ghi_data, cloud_data = self.weather.fetch_weather_data(self.race)
            except Exception as e:
                print(f"Warning: API fetch failed ({e}), using synthetic data")
                time_minutes, ghi_data, cloud_data = self.weather.generate_synthetic_data(self.race)
        else:
            time_minutes, ghi_data, cloud_data = self.weather.generate_synthetic_data(self.race)

        dt_minutes = self.race.time_step_minutes
        dt_hours = dt_minutes / 60.0

        self.strategy.prepare(self.physics, ghi_data, cloud_data, dt_hours)
        ideal_soc = self.strategy.ideal_soc
        print(f"   Optimal constant speed baseline: {self.strategy.optimal_speed:.2f} m/s ({self.strategy.optimal_speed*2.237:.1f} mph)")

        n_steps = len(time_minutes)
        soc = self.race.start_soc
        speed = self.strategy.optimal_speed
        total_distance = 0.0
        current_lap_distance = 0.0
        total_laps = 0

        soc_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        bdr_arr = np.zeros(n_steps)
        ghi_arr = np.zeros(n_steps)
        cloud_arr = np.zeros(n_steps)
        lap_times = []
        lap_start_time = time_minutes[0]

        for i in range(n_steps):
            ghi = max(ghi_data[i], 0)
            cloud = cloud_data[i]

            power_in = self.physics.solar_power(ghi)
            power_out = self.physics.power_drained(speed)
            bdr = self.physics.battery_drain_rate(speed, ghi)

            energy_delta = (power_in - power_out) * dt_hours
            soc += energy_delta / self.car.battery_capacity
            soc = float(np.clip(soc, self.race.min_soc, 1.0))

            distance_step = speed * (dt_minutes * 60)
            total_distance += distance_step
            current_lap_distance += distance_step

            if current_lap_distance >= self.track.lap_distance_m:
                total_laps += 1
                lap_time = time_minutes[i] - lap_start_time
                lap_times.append(lap_time)
                current_lap_distance -= self.track.lap_distance_m
                lap_start_time = time_minutes[i]

            soc_error = soc - ideal_soc[i]
            prev_speed = speed
            speed = self.strategy.next_speed(i, speed, soc, soc_error, bdr, self.physics, ghi_data, dt_hours)

            regen_wh = self.physics.regen_energy(prev_speed, speed)
            if regen_wh > 0:
                soc += regen_wh / self.car.battery_capacity
                soc = min(soc, 1.0)

            soc_arr[i] = soc
            speed_arr[i] = speed
            bdr_arr[i] = bdr
            ghi_arr[i] = ghi
            cloud_arr[i] = cloud

        return SimulationResults(
            time_minutes=time_minutes,
            soc=soc_arr,
            speed=speed_arr,
            ghi=ghi_arr,
            cloud_cover=cloud_arr,
            lap_times=lap_times,
            total_laps=total_laps,
            total_distance_m=total_distance,
            bdr=bdr_arr,
            ideal_soc=ideal_soc,
        )
