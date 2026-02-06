"""
Interactive Solar Car Laps Race Simulator - Brainerd International Raceway

Run this during a race. After each lap, input the car's actual state
(SoC, lap time) and get the optimal target speed for the next lap.

The simulator uses real weather forecasts and a predictive speed controller
to recommend the speed that maximizes laps completed while hitting
the target SoC at the end of the race.

Usage:
    python interactive_laps_sim.py
"""

from laps_sim import (
    TrackConfig, CarConfig, RaceConfig,
    PhysicsEngine, WeatherService, SpeedController,
)
import numpy as np


class InteractiveSimulator:
    """Lap-by-lap interactive race simulator with strategy recommendations."""

    def __init__(self, track, car, race, use_api_weather=True):
        self.track = track
        self.car = car
        self.race = race
        self.physics = PhysicsEngine(car)
        self.weather_service = WeatherService(track)
        self.controller = SpeedController(race)

        # Fetch weather for the entire race window
        print("Fetching weather data...")
        if use_api_weather:
            try:
                self.time_minutes, self.ghi_data, self.cloud_data = \
                    self.weather_service.fetch_weather_data(race)
                print("  Using live weather forecast from Open-Meteo API")
            except Exception as e:
                print(f"  API failed ({e}), using synthetic weather data")
                self.time_minutes, self.ghi_data, self.cloud_data = \
                    self.weather_service.generate_synthetic_data(race)
        else:
            self.time_minutes, self.ghi_data, self.cloud_data = \
                self.weather_service.generate_synthetic_data(race)
            print("  Using synthetic weather data")

        # Mutable race state
        self.current_soc = race.start_soc
        self.current_speed = race.initial_speed_mps
        self.current_step = 0
        self.total_laps = 0
        self.total_distance_m = 0.0
        self.current_lap_distance = 0.0
        self.lap_history = []

    @property
    def n_total_steps(self):
        return len(self.time_minutes)

    @property
    def time_remaining_hours(self):
        steps_left = self.n_total_steps - self.current_step
        return max(0, steps_left * self.race.time_step_minutes / 60)

    @property
    def current_time_str(self):
        if self.current_step < self.n_total_steps:
            total_min = self.time_minutes[self.current_step]
        else:
            total_min = self.time_minutes[-1]
        h = int(total_min // 60)
        m = int(total_min % 60)
        period = "AM" if h < 12 else "PM"
        dh = h if h <= 12 else h - 12
        if dh == 0:
            dh = 12
        return f"{dh}:{m:02d} {period}"

    @property
    def is_race_over(self):
        return self.current_step >= self.n_total_steps

    def _compute_ideal_soc_remaining(self, cloud_data):
        """Ideal SoC curve from current_soc to target_soc over remaining steps."""
        n = len(cloud_data)
        if n == 0:
            return np.array([])

        soc_drop = self.current_soc - self.race.target_soc
        if soc_drop <= 0:
            return np.full(n, self.current_soc)

        ideal = np.zeros(n)
        ideal[0] = self.current_soc

        cloud_factors = (1 - cloud_data) * 0.5 + 0.5
        total_weight = np.sum(cloud_factors)
        if total_weight == 0:
            total_weight = 1.0
        drops = cloud_factors * (soc_drop / total_weight)

        for i in range(1, n):
            ideal[i] = ideal[i - 1] - drops[i]
            ideal[i] = max(ideal[i], self.race.target_soc)

        return ideal

    def project_remaining(self):
        """Simulate rest of race from current state. Returns projection dict."""
        if self.is_race_over:
            return None

        rem_ghi = self.ghi_data[self.current_step:]
        rem_cloud = self.cloud_data[self.current_step:]
        n = len(rem_ghi)
        ideal_soc = self._compute_ideal_soc_remaining(rem_cloud)

        dt_hours = self.race.time_step_minutes / 60
        soc = self.current_soc
        speed = self.current_speed
        lap_dist = self.current_lap_distance
        proj_laps = 0
        proj_distance = 0.0

        speeds = np.zeros(n)
        socs = np.zeros(n)

        for i in range(n):
            ghi = max(rem_ghi[i], 0)
            bdr = self.physics.battery_drain_rate(speed, ghi)
            power_in = self.physics.solar_power(ghi)
            power_out = self.physics.power_drained(speed)

            energy_delta = (power_in - power_out) * dt_hours
            soc += energy_delta / self.car.battery_capacity
            soc = max(soc, self.race.min_soc)

            dist_step = speed * (self.race.time_step_minutes * 60)
            proj_distance += dist_step
            lap_dist += dist_step

            if lap_dist >= self.track.lap_distance_m:
                proj_laps += 1
                lap_dist -= self.track.lap_distance_m

            soc_error = soc - ideal_soc[i]
            speed = self.controller.adjust_speed(speed, soc_error, bdr)

            speeds[i] = speed
            socs[i] = soc

        return {
            'final_soc': soc,
            'additional_laps': proj_laps,
            'total_laps': self.total_laps + proj_laps,
            'additional_distance_km': proj_distance / 1000,
            'speeds': speeds,
            'socs': socs,
        }

    def get_recommendation(self):
        """Recommended target speed for the next lap and projected outcomes."""
        proj = self.project_remaining()
        if proj is None:
            return None

        # Estimate time steps for the next lap
        remaining_lap_m = self.track.lap_distance_m - self.current_lap_distance
        speed_est = self.current_speed if self.current_speed > 0 else self.race.initial_speed_mps
        est_lap_seconds = remaining_lap_m / speed_est
        est_steps = max(1, int(est_lap_seconds / (self.race.time_step_minutes * 60)))
        est_steps = min(est_steps, len(proj['speeds']))

        rec_speed = float(np.mean(proj['speeds'][:est_steps]))

        soc_after = float(proj['socs'][min(est_steps - 1, len(proj['socs']) - 1)])

        # Weather context for the next lap period
        end_idx = min(self.current_step + est_steps, self.n_total_steps)
        next_ghi = self.ghi_data[self.current_step:end_idx]
        next_cloud = self.cloud_data[self.current_step:end_idx]
        avg_ghi = float(np.mean(next_ghi)) if len(next_ghi) > 0 else 0
        avg_cloud = float(np.mean(next_cloud)) if len(next_cloud) > 0 else 0

        # Power balance at recommended speed and average GHI
        power_in = self.physics.solar_power(avg_ghi)
        power_out = self.physics.power_drained(rec_speed)
        net_power = power_in - power_out

        return {
            'speed_mps': rec_speed,
            'speed_mph': rec_speed * 2.237,
            'speed_kmh': rec_speed * 3.6,
            'soc_after_lap': soc_after,
            'est_lap_time_min': est_lap_seconds / 60,
            'final_soc': proj['final_soc'],
            'total_laps': proj['total_laps'],
            'avg_ghi': avg_ghi,
            'avg_cloud_pct': avg_cloud * 100,
            'power_in_w': power_in,
            'power_out_w': power_out,
            'net_power_w': net_power,
        }

    def record_lap(self, actual_soc_pct, lap_time_min, avg_speed_mph=None):
        """Record a completed lap with actual data. SoC is in percent (0-100)."""
        self.total_laps += 1
        self.total_distance_m += self.track.lap_distance_m
        self.current_lap_distance = 0.0

        # Advance clock
        steps = int(lap_time_min / self.race.time_step_minutes)
        self.current_step = min(self.current_step + steps, self.n_total_steps)

        # Update state from actuals
        self.current_soc = actual_soc_pct / 100.0
        if avg_speed_mph is not None:
            self.current_speed = avg_speed_mph / 2.237
        elif lap_time_min > 0:
            self.current_speed = self.track.lap_distance_m / (lap_time_min * 60)

        self.lap_history.append({
            'lap': self.total_laps,
            'soc': actual_soc_pct,
            'lap_time': lap_time_min,
            'time': self.current_time_str,
        })


# =============================================================================
# Display helpers
# =============================================================================

def prompt_float(message, default=None, min_val=None, max_val=None):
    """Prompt for a float value. Returns 'quit' on 'q', None on empty with no default."""
    while True:
        if default is not None:
            raw = input(f"  {message} [{default}]: ").strip()
            if raw == '':
                return float(default)
        else:
            raw = input(f"  {message}: ").strip()
            if raw == '':
                return None

        if raw.lower() == 'q':
            return 'quit'

        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"    Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"    Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("    Please enter a number.")


def print_banner():
    print()
    print("=" * 60)
    print("   INTERACTIVE SOLAR CAR LAPS RACE SIMULATOR")
    print("   Brainerd International Raceway")
    print("=" * 60)
    print()
    print("  After each lap, enter your car's actual state.")
    print("  The simulator will tell you the optimal speed")
    print("  for the next lap.")
    print()
    print("  Enter 'q' at any prompt to quit.")
    print()


def print_state(sim):
    print()
    print(f"  Time:           {sim.current_time_str}")
    print(f"  Time remaining: {sim.time_remaining_hours:.1f} hours")
    print(f"  Current SoC:    {sim.current_soc * 100:.1f}%")
    print(f"  Laps completed: {sim.total_laps}")
    print(f"  Distance:       {sim.total_distance_m / 1000:.1f} km "
          f"({sim.total_distance_m / 1609.34:.1f} mi)")


def print_recommendation(rec, sim):
    print()
    print("-" * 60)
    print("   STRATEGY RECOMMENDATION")
    print("-" * 60)
    print(f"   Target speed:    {rec['speed_mph']:.1f} mph"
          f"   ({rec['speed_mps']:.1f} m/s  /  {rec['speed_kmh']:.1f} km/h)")
    print(f"   Est. lap time:   {rec['est_lap_time_min']:.1f} min")
    print(f"   SoC after lap:   {rec['soc_after_lap'] * 100:.1f}%")
    print()
    print(f"   Solar input:     {rec['power_in_w']:.0f} W"
          f"   (GHI: {rec['avg_ghi']:.0f} W/m2, cloud: {rec['avg_cloud_pct']:.0f}%)")
    print(f"   Motor drain:     {rec['power_out_w']:.0f} W")
    net = rec['net_power_w']
    label = "CHARGING" if net > 0 else "DRAINING"
    print(f"   Net power:       {abs(net):.0f} W {label}")
    print()
    print(f"   If you follow this strategy for the rest of the race:")
    print(f"     Final SoC:     {rec['final_soc'] * 100:.1f}%"
          f"   (target: {sim.race.target_soc * 100:.0f}%)")
    print(f"     Total laps:    {rec['total_laps']}")
    print("-" * 60)


def print_lap_history(sim):
    if not sim.lap_history:
        return
    print()
    print("  LAP HISTORY")
    print(f"  {'Lap':>4}  {'Clock':>10}  {'SoC':>7}  {'Lap Time':>10}")
    print(f"  {'----':>4}  {'----------':>10}  {'-------':>7}  {'----------':>10}")
    for lap in sim.lap_history:
        print(f"  {lap['lap']:>4}  {lap['time']:>10}  {lap['soc']:>5.1f}%"
              f"  {lap['lap_time']:>8.1f} min")


def print_final_summary(sim):
    print()
    print("=" * 60)
    print("   RACE COMPLETE")
    print("=" * 60)
    print(f"   Total laps:     {sim.total_laps}")
    print(f"   Total distance: {sim.total_distance_m / 1000:.1f} km"
          f"  ({sim.total_distance_m / 1609.34:.1f} mi)")
    print(f"   Final SoC:      {sim.current_soc * 100:.1f}%")
    print(f"   Target SoC:     {sim.race.target_soc * 100:.0f}%")
    print(f"   SoC error:      {abs(sim.current_soc - sim.race.target_soc) * 100:.1f}%")
    print_lap_history(sim)
    print()
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print_banner()

    # Load car parameters
    try:
        car = CarConfig.from_json("car_params.json")
        print("  Loaded car parameters from car_params.json")
    except FileNotFoundError:
        car = CarConfig()
        print("  Using default car parameters")

    track = TrackConfig()
    print(f"  Track: {track.name}")
    print(f"  Lap:   {track.lap_distance_km:.3f} km"
          f" ({track.lap_distance_km / 1.609:.2f} mi)")

    # Race configuration
    print("\n--- Race Configuration ---")

    target_soc = prompt_float("Target end-of-race SoC %",
                              default=10, min_val=0, max_val=100)
    if target_soc == 'quit':
        return

    aggressiveness = prompt_float("Aggressiveness (1.0=normal, 1.6=aggressive)",
                                  default=1.6, min_val=0.1, max_val=5.0)
    if aggressiveness == 'quit':
        return

    start_soc = prompt_float("Starting SoC %",
                             default=100, min_val=0, max_val=100)
    if start_soc == 'quit':
        return

    race = RaceConfig(
        start_soc=start_soc / 100.0,
        target_soc=target_soc / 100.0,
        aggressiveness=aggressiveness,
        initial_speed_mps=20.0,
        time_step_minutes=1.0,
    )

    # Create simulator (fetches weather)
    sim = InteractiveSimulator(track, car, race, use_api_weather=True)

    print()
    print("=" * 60)
    print("   RACE STARTED")
    print("=" * 60)

    # Main interactive loop
    while not sim.is_race_over:
        print_state(sim)

        rec = sim.get_recommendation()
        if rec is None:
            print("\n  No time remaining for another lap.")
            break

        print_recommendation(rec, sim)

        # Wait for the lap to be completed
        print()
        user_in = input("  Press Enter when lap is complete"
                        " (or 'q' to quit): ").strip()
        if user_in.lower() == 'q':
            break

        # Collect actual lap data
        print("\n  --- Enter Actual Lap Data ---")

        actual_soc = prompt_float("Actual SoC %", min_val=0, max_val=100)
        if actual_soc == 'quit':
            break
        if actual_soc is None:
            print("    SoC is required.")
            continue

        lap_time = prompt_float("Lap time (minutes)",
                                min_val=0.1, max_val=120)
        if lap_time == 'quit':
            break
        if lap_time is None:
            print("    Lap time is required.")
            continue

        avg_speed = prompt_float(
            "Avg speed this lap in mph (Enter to calc from lap time)"
        )
        if avg_speed == 'quit':
            break

        sim.record_lap(actual_soc, lap_time, avg_speed)
        print(f"\n  Lap {sim.total_laps} recorded.")

    # Final summary
    print_final_summary(sim)


if __name__ == "__main__":
    main()
