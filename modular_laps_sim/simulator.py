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
        completed_full_window = True
        reached_min_soc_time_minutes = None
        laps_at_min_soc = None

        time_arr = []
        soc_arr = []
        speed_arr = []
        bdr_arr = []
        ghi_arr = []
        cloud_arr = []
        ideal_soc_arr = []
        lap_times = []
        lap_start_time = time_minutes[0]

        for i in range(n_steps):
            ghi = max(ghi_data[i], 0)
            cloud = cloud_data[i]

            power_in = self.physics.solar_power(ghi)
            power_out = self.physics.power_drained(speed)
            bdr = self.physics.battery_drain_rate(speed, ghi)

            prev_soc = soc
            energy_delta = (power_in - power_out) * dt_hours
            next_soc = prev_soc + energy_delta / self.car.battery_capacity

            step_fraction = 1.0
            hit_min_soc = prev_soc > self.race.min_soc and next_soc <= self.race.min_soc
            if hit_min_soc:
                denom = prev_soc - next_soc
                if denom > 0:
                    step_fraction = float(np.clip((prev_soc - self.race.min_soc) / denom, 0.0, 1.0))
                soc = self.race.min_soc
                completed_full_window = False
                reached_min_soc_time_minutes = time_minutes[i] + dt_minutes * step_fraction
            else:
                soc = float(np.clip(next_soc, self.race.min_soc, 1.0))

            distance_step = speed * (dt_minutes * 60) * step_fraction
            total_distance += distance_step
            current_lap_distance += distance_step

            if current_lap_distance >= self.track.lap_distance_m and speed > 0:
                total_laps += 1
                lap_time = (time_minutes[i] + dt_minutes * step_fraction) - lap_start_time
                lap_times.append(lap_time)
                current_lap_distance -= self.track.lap_distance_m
                lap_start_time = time_minutes[i] + dt_minutes * step_fraction

            soc_error = soc - ideal_soc[i]
            prev_speed = speed
            if not hit_min_soc:
                speed = self.strategy.next_speed(i, speed, soc, soc_error, bdr, self.physics, ghi_data, dt_hours)

                regen_wh = self.physics.regen_energy(prev_speed, speed)
                if regen_wh > 0:
                    soc += regen_wh / self.car.battery_capacity
                    soc = min(soc, 1.0)

            time_arr.append(time_minutes[i] + dt_minutes * step_fraction)
            soc_arr.append(soc)
            speed_arr.append(prev_speed)
            bdr_arr.append(bdr)
            ghi_arr.append(ghi)
            cloud_arr.append(cloud)
            ideal_soc_arr.append(ideal_soc[i])

            if hit_min_soc:
                laps_at_min_soc = total_laps
                break

        return SimulationResults(
            time_minutes=np.array(time_arr),
            soc=np.array(soc_arr),
            speed=np.array(speed_arr),
            ghi=np.array(ghi_arr),
            cloud_cover=np.array(cloud_arr),
            lap_times=lap_times,
            total_laps=total_laps,
            total_distance_m=total_distance,
            bdr=np.array(bdr_arr),
            ideal_soc=np.array(ideal_soc_arr),
            completed_full_window=completed_full_window,
            reached_min_soc_time_minutes=reached_min_soc_time_minutes,
            laps_at_min_soc=laps_at_min_soc,
        )
