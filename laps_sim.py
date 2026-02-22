"""
Solar Car Laps Race Simulator - Brainerd International Raceway, Minnesota

This simulator models a solar car's performance during a laps-based race
at Brainerd International Raceway. The race format typically involves
completing as many laps as possible within a time window while managing
battery state of charge (SoC).

Track Info:
- Location: Alton, Virginia (36.5666°N, 79.2058°W)
- Patriot Course Length: 1.10 miles (1.77 km)
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openmeteo_requests
import requests_cache


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class TrackConfig:
    """Configuration for the race track."""
    name: str = "Virginia International Raceway (Patriot Course)"
    location: str = "Alton, Virginia"
    latitude: float = 36.5666
    longitude: float = -79.2058
    lap_distance_km: float = 1.77  # 1.10 miles Patriot Course
    lap_distance_m: float = field(init=False)
    timezone: str = "America/New_York"
    
    def __post_init__(self):
        self.lap_distance_m = self.lap_distance_km * 1000


@dataclass
class CarConfig:
    """Solar car physical parameters."""
    mass: float = 337  # kg
    battery_capacity: float = 5000  # Wh
    solar_panel_area: float = 4  # m^2
    solar_panel_efficiency: float = 0.23
    electrical_efficiency: float = 0.99
    C_dA: float = 0.73809  # drag coefficient * frontal area
    tire_pressure: float = 5  # bar
    regen_efficiency: float = 0.5  # regenerative braking efficiency coefficient (μ)
    rho: float = 1.192  # air density kg/m^3
    g: float = 9.80665  # gravity m/s^2
    
    @classmethod
    def from_json(cls, filepath: str) -> 'CarConfig':
        """Load car configuration from JSON file."""
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        # Handle 'inf' string values
        for key, value in params.items():
            if value == "inf":
                params[key] = math.inf
        
        return cls(
            mass=params.get("mass", 337),
            battery_capacity=params.get("battery_capacity", 5000),
            solar_panel_area=params.get("solar_panel_area", 4),
            solar_panel_efficiency=params.get("solar_panel_efficiency", 0.23),
            electrical_efficiency=params.get("electrical_efficiency", 0.99),
            C_dA=params.get("C_dA", 0.73809),
            tire_pressure=params.get("tire_pressure", 5),
            regen_efficiency=params.get("regen_efficiency", 0.5),
            rho=params.get("rho", 1.192),
            g=params.get("g", 9.80665)
        )


@dataclass
class RaceConfig:
    """Race simulation parameters."""
    start_time_hour: float = 10.0  # 10:00 AM
    end_time_hour: float = 18.0    # 6:00 PM
    start_soc: float = 1.0         # Starting state of charge (100%)
    target_soc: float = 0.10       # Target SoC at end (10%)
    min_soc: float = 0.10          # Minimum allowed SoC (10%)
    aggressiveness: float = 1.2   # Speed adjustment aggressiveness
    initial_speed_mps: float = 20.0  # Initial speed in m/s (~45 mph)
    max_speed_mps: float = 35.0    # Max speed m/s (~78 mph)
    min_speed_mps: float = 8.0     # Min speed ms (~18 mph)
    time_step_minutes: float = 1.0 # Simulation time step


@dataclass
class SimulationResults:
    """Container for simulation results."""
    time_minutes: np.ndarray
    soc: np.ndarray
    speed: np.ndarray
    ghi: np.ndarray
    cloud_cover: np.ndarray
    lap_times: List[float]
    total_laps: int
    total_distance_m: float
    bdr: np.ndarray  # Battery drain rate
    ideal_soc: np.ndarray
    
    @property
    def total_distance_km(self) -> float:
        return self.total_distance_m / 1000
    
    @property
    def total_distance_miles(self) -> float:
        return self.total_distance_m / 1609.34
    
    @property
    def avg_speed_mps(self) -> float:
        return np.mean(self.speed)
    
    @property
    def avg_speed_mph(self) -> float:
        return self.avg_speed_mps * 2.237
    
    @property
    def final_soc(self) -> float:
        return self.soc[-1]


# =============================================================================
# Physics Engine
# =============================================================================

class PhysicsEngine:
    """Handles all physics calculations for the solar car."""
    
    def __init__(self, car: CarConfig):
        self.car = car
    
    def get_crr(self, v: float) -> float:
        """Calculate coefficient of rolling resistance based on velocity."""
        v_kmh = v * 3.6  # convert to km/h
        return 0.005 + (1 / self.car.tire_pressure) * (0.01 + 0.0095 * (v_kmh / 100) ** 2)
    
    def force_drag(self, v: float) -> float:
        """Calculate aerodynamic drag force (N)."""
        return 0.5 * self.car.C_dA * self.car.rho * v ** 2
    
    def force_rolling_resistance(self, v: float) -> float:
        """Calculate rolling resistance force (N)."""
        return self.car.mass * self.car.g * self.get_crr(v)
    
    def power_drained(self, v: float) -> float:
        """Calculate power consumption at given velocity (W)."""
        F_d = self.force_drag(v)
        F_r = self.force_rolling_resistance(v)
        return (F_d + F_r) * v * self.car.electrical_efficiency
    
    def solar_power(self, ghi: float) -> float:
        """Calculate solar power input (W)."""
        return self.car.solar_panel_area * ghi * self.car.solar_panel_efficiency
    
    def regen_energy(self, v_old: float, v_new: float) -> float:
        """
        Calculate energy recovered from regenerative braking (Wh).
        
        E_regen = (1/2 * m * (v_old² - v_new²)) / 3600 * μ
        
        Only returns positive energy when decelerating (v_new < v_old).
        """
        if v_new >= v_old:
            return 0.0
        delta_ke = 0.5 * self.car.mass * (v_old ** 2 - v_new ** 2)
        return (delta_ke / 3600) * self.car.regen_efficiency
    
    def battery_drain_rate(self, v: float, ghi: float) -> float:
        """Calculate battery drain rate (fraction per hour)."""
        power_out = self.power_drained(v)
        power_in = self.solar_power(ghi) 
        return (power_out - power_in) / self.car.battery_capacity


# =============================================================================
# Weather Service
# =============================================================================

class WeatherService:
    """Fetches and processes weather data from Open-Meteo API."""
    
    def __init__(self, track: TrackConfig):
        self.track = track
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = openmeteo_requests.Client(
                session=requests_cache.CachedSession(
                    cache_name='openmeteo_cache',
                    backend='memory'
                )
            )
        return self._client
    
    def fetch_weather_data(self, race_config: RaceConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetch GHI and cloud cover data from Open-Meteo API.
        
        Returns:
            Tuple of (time_minutes, ghi_data, cloud_data)
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.track.latitude,
            "longitude": self.track.longitude,
            "hourly": ["shortwave_radiation_instant", "cloud_cover"],
        }
        
        responses = self.client.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        ghi_values = hourly.Variables(0).ValuesAsNumpy()
        cloud_values = hourly.Variables(1).ValuesAsNumpy()
        
        # Create DataFrame with timestamps
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "ghi": ghi_values,
            "cloud_cover": cloud_values
        }
        
        df = pd.DataFrame(data=hourly_data)
        df = df.set_index("date").tz_convert(self.track.timezone)
        
        # Filter to race hours
        start_time = f"{int(race_config.start_time_hour):02d}:00"
        end_time = f"{int(race_config.end_time_hour):02d}:00"
        df = df.between_time(start_time, end_time)
        df = df[1:10]  # First day only (skip first incomplete hour)
        df = df.reset_index()
        
        # Resample to 1-minute intervals
        df = df.resample("min", on="date").mean()
        df["ghi"] = df["ghi"].interpolate(method="spline", order=3)
        df["cloud_cover"] = df["cloud_cover"].interpolate(method="linear")
        
        # Convert to arrays
        n_points = len(df)
        time_minutes = np.arange(n_points) + race_config.start_time_hour * 60
        ghi_data = np.clip(df["ghi"].to_numpy(), 0, None)
        cloud_data = df["cloud_cover"].to_numpy() / 100  # Convert to 0-1 range
        
        return time_minutes, ghi_data, cloud_data
    
    def generate_synthetic_data(self, race_config: RaceConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic weather data for testing."""
        start_min = int(race_config.start_time_hour * 60)
        end_min = int(race_config.end_time_hour * 60)
        time_minutes = np.arange(start_min, end_min + 1)
        
        # Parabolic GHI curve peaking at solar noon
        t_peak = 13 * 60  # 1:00 PM
        peak_ghi = 900  # W/m² (typical for Minnesota summer)
        a = peak_ghi / ((t_peak - start_min) ** 2)
        ghi_data = np.maximum(0, -a * (time_minutes - t_peak) ** 2 + peak_ghi)
        
        # Add smooth random variation
        noise = np.clip(np.random.normal(1.0, 0.1, len(ghi_data)), 0.7, 1.2)
        smooth_noise = np.convolve(noise, np.ones(15)/15, mode='same')
        ghi_data = ghi_data * smooth_noise
        
        # Synthetic cloud cover (inverse correlation with GHI variation)
        cloud_data = np.clip(0.3 + 0.3 * np.random.randn(len(time_minutes)), 0, 1)
        cloud_data = np.convolve(cloud_data, np.ones(30)/30, mode='same')
        
        return time_minutes, ghi_data, cloud_data


# =============================================================================
# Speed Controller
# =============================================================================

class SpeedController:
    """
    Manages speed using an energy-budget optimal speed + gentle PI corrections.
    
    Strategy:
      1. Pre-compute the optimal *constant* speed that exactly uses the
         available energy budget (battery + total solar input) over the race.
         Constant speed is provably optimal because drag power ~ v³, so by
         Jensen's inequality any fluctuation wastes energy.
      2. During the race, apply a small PI (proportional-integral) correction
         to handle forecast errors, keeping speed nearly constant while
         tracking the ideal SoC curve.
    """
    
    def __init__(self, race_config: RaceConfig):
        self.config = race_config
        self._integral_error: float = 0.0  # accumulated SoC error for I-term
        self._optimal_speed: Optional[float] = None
    
    def compute_optimal_speed(
        self, physics: 'PhysicsEngine', ghi_data: np.ndarray, dt_hours: float
    ) -> float:
        """
        Compute the constant speed that exactly depletes the energy budget.
        
        Energy budget:
            E_available = battery_usable + sum(solar_power_i * dt)
        
        Energy cost at constant speed v for N steps:
            E_cost = N * power_drained(v) * dt
        
        Solve for v such that E_cost = E_available using bisection.
        """
        cap = physics.car.battery_capacity
        usable_energy = (self.config.start_soc - self.config.target_soc) * cap  # Wh
        total_solar = np.sum(physics.solar_power(ghi_data)) * dt_hours  # Wh
        total_available = usable_energy + total_solar  # Wh
        n_steps = len(ghi_data)
        
        # Bisection: find v where total drain == total_available
        v_lo, v_hi = self.config.min_speed_mps, self.config.max_speed_mps
        for _ in range(60):  # converges in ~60 iterations to <0.001 m/s
            v_mid = (v_lo + v_hi) / 2.0
            cost = n_steps * physics.power_drained(v_mid) * dt_hours
            if cost < total_available:
                v_lo = v_mid
            else:
                v_hi = v_mid
        
        self._optimal_speed = (v_lo + v_hi) / 2.0
        return self._optimal_speed
    
    def compute_ideal_soc_curve(
        self, cloud_data: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Compute ideal SoC curve that adapts to cloud conditions.
        
        - Sunny (low cloud): Can use more battery (faster SoC decline)
        - Cloudy (high cloud): Conserve battery (slower SoC decline)
        """
        n_steps = len(cloud_data)
        soc = np.zeros(n_steps)
        soc[0] = self.config.start_soc
        soc_drop_total = self.config.start_soc - self.config.target_soc
        
        cloud_factors = (1 - cloud_data) * alpha + (1 - alpha)
        total_weighted = np.sum(cloud_factors)
        normalized_drops = cloud_factors * (soc_drop_total / total_weighted)
        
        for i in range(1, n_steps):
            soc[i] = soc[i-1] - normalized_drops[i]
            soc[i] = max(soc[i], self.config.target_soc)
        
        return soc
    
    def adjust_speed(
        self, current_speed: float, soc_error: float, bdr: float,
        current_soc: float = 1.0
    ) -> float:
        """
        Apply a gentle PI correction around the pre-computed optimal speed,
        with an emergency slowdown layer that forces the car toward min_speed
        when SoC approaches the minimum threshold.
        
        Args:
            current_speed: Current velocity (m/s) — used as fallback only
            soc_error: Current SoC - Ideal SoC (positive = above target)
            bdr: Battery drain rate (unused, kept for interface compatibility)
            current_soc: Actual SoC right now (used for emergency guard)
        
        Returns:
            Adjusted speed (m/s)
        """
        base_speed = self._optimal_speed if self._optimal_speed else current_speed
        agg = self.config.aggressiveness
        
        # PI gains (scaled by aggressiveness)
        Kp = 0.08 * agg   # proportional gain
        Ki = 0.002 * agg   # integral gain
        
        # Update integral with anti-windup clamp
        self._integral_error += soc_error
        self._integral_error = np.clip(self._integral_error, -5.0, 5.0)
        
        # Compute correction (positive error → speed up to use excess SoC)
        correction = Kp * soc_error + Ki * self._integral_error
        
        # Limit total correction to ±5 % of base speed for stability
        correction = np.clip(correction, -0.15, 0.15)
        
        speed = base_speed * (1.0 + correction)
        
        # --- Emergency SoC guard ---
        # When SoC is within a 5% margin above min_soc, progressively
        # blend speed toward min_speed to prevent breaching the floor.
        margin = 0.05  # start slowing 5% above min_soc
        soc_above_min = current_soc - self.config.min_soc
        if soc_above_min < margin:
            # urgency goes from 0 (at min_soc + margin) to 1 (at min_soc)
            urgency = 1.0 - max(soc_above_min, 0.0) / margin
            speed = speed * (1 - urgency) + self.config.min_speed_mps * urgency
        
        return np.clip(speed, self.config.min_speed_mps, self.config.max_speed_mps)
    
    def reset(self):
        """Reset controller state (call before a new simulation run)."""
        self._integral_error = 0.0
        self._optimal_speed = None


# =============================================================================
# Laps Race Simulator
# =============================================================================

class LapsRaceSimulator:
    """Main simulator for laps-based solar car races."""
    
    def __init__(
        self,
        track: Optional[TrackConfig] = None,
        car: Optional[CarConfig] = None,
        race: Optional[RaceConfig] = None,
        use_api_weather: bool = True
    ):
        self.track = track or TrackConfig()
        self.car = car or CarConfig()
        self.race = race or RaceConfig()
        self.use_api_weather = use_api_weather
        
        self.physics = PhysicsEngine(self.car)
        self.weather = WeatherService(self.track)
        self.controller = SpeedController(self.race)
    
    def run(self) -> SimulationResults:
        """
        Run the laps race simulation.
        
        Returns:
            SimulationResults containing all simulation data
        """
        # Fetch weather data
        if self.use_api_weather:
            try:
                time_minutes, ghi_data, cloud_data = self.weather.fetch_weather_data(self.race)
            except Exception as e:
                print(f"Warning: API fetch failed ({e}), using synthetic data")
                time_minutes, ghi_data, cloud_data = self.weather.generate_synthetic_data(self.race)
        else:
            time_minutes, ghi_data, cloud_data = self.weather.generate_synthetic_data(self.race)
        
        # Compute ideal SoC curve and optimal constant speed
        ideal_soc = self.controller.compute_ideal_soc_curve(cloud_data)
        dt_hours = self.race.time_step_minutes / 60
        optimal_v = self.controller.compute_optimal_speed(
            self.physics, ghi_data, dt_hours
        )
        print(f"   Optimal constant speed: {optimal_v:.2f} m/s ({optimal_v*2.237:.1f} mph)")
        
        # Initialize simulation state
        n_steps = len(time_minutes)
        dt_minutes = self.race.time_step_minutes
        
        soc = self.race.start_soc
        speed = optimal_v  # Start at the pre-computed optimal speed
        total_distance = 0.0
        current_lap_distance = 0.0
        total_laps = 0
        
        # Result arrays
        soc_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        bdr_arr = np.zeros(n_steps)
        ghi_arr = np.zeros(n_steps)
        cloud_arr = np.zeros(n_steps)
        lap_times = []
        lap_start_time = time_minutes[0]
        
        # Main simulation loop
        for i in range(n_steps):
            ghi = max(ghi_data[i], 0)
            cloud = cloud_data[i]
            
            # Calculate power balance
            power_in = self.physics.solar_power(ghi)
            power_out = self.physics.power_drained(speed)
            bdr = self.physics.battery_drain_rate(speed, ghi)
            
            # Update SoC
            energy_delta = (power_in - power_out) * dt_hours  # Wh
            soc += energy_delta / self.car.battery_capacity
            soc = max(soc, self.race.min_soc)
            
            # Calculate distance for this time step
            distance_step = speed * (dt_minutes * 60)  # meters
            total_distance += distance_step
            current_lap_distance += distance_step
            
            # Check for lap completion
            if current_lap_distance >= self.track.lap_distance_m:
                total_laps += 1
                lap_time = time_minutes[i] - lap_start_time
                lap_times.append(lap_time)
                current_lap_distance -= self.track.lap_distance_m
                lap_start_time = time_minutes[i]
            
            # Adjust speed based on SoC tracking
            soc_error = soc - ideal_soc[i]
            prev_speed = speed
            speed = self.controller.adjust_speed(speed, soc_error, bdr, current_soc=soc)
            
            # Apply regenerative braking energy if decelerating
            regen_wh = self.physics.regen_energy(prev_speed, speed)
            if regen_wh > 0:
                soc += regen_wh / self.car.battery_capacity
                soc = min(soc, 1.0)  # Cap at 100%
            
            # Store results
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
            ideal_soc=ideal_soc
        )


# =============================================================================
# Visualization
# =============================================================================

class SimulationPlotter:
    """Generates plots from simulation results."""
    
    def __init__(self, results: SimulationResults, track: TrackConfig):
        self.results = results
        self.track = track
    
    def _time_to_hours(self) -> np.ndarray:
        """Convert time_minutes to hours from midnight."""
        return self.results.time_minutes / 60
    
    def plot_soc(self, save_path: Optional[str] = None, show: bool = False):
        """Plot state of charge over time."""
        fig, ax = plt.subplots(figsize=(10, 5))
        hours = self._time_to_hours()
        
        ax.plot(hours, self.results.soc * 100, label="Actual SoC", linewidth=2)
        ax.plot(hours, self.results.ideal_soc * 100, '--', label="Ideal SoC", 
                alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("State of Charge (%)")
        ax.set_title(f"Battery SoC - {self.track.name}")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close()
    
    def plot_speed(self, save_path: Optional[str] = None, show: bool = False):
        """Plot speed over time."""
        fig, ax = plt.subplots(figsize=(10, 5))
        hours = self._time_to_hours()
        
        # Convert to mph for display
        speed_mph = self.results.speed * 2.237
        
        ax.plot(hours, speed_mph, color='green', linewidth=1.5)
        ax.axhline(y=np.mean(speed_mph), color='red', linestyle='--', 
                   label=f'Average: {np.mean(speed_mph):.1f} mph', alpha=0.7)
        
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Speed (mph)")
        ax.set_title(f"Speed Profile - {self.track.name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close()
    
    def plot_weather(self, save_path: Optional[str] = None, show: bool = False):
        """Plot GHI and cloud cover."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        hours = self._time_to_hours()
        
        # GHI plot
        ax1.plot(hours, self.results.ghi, color='orange', linewidth=1.5)
        ax1.set_ylabel("GHI (W/m²)")
        ax1.set_title(f"Solar Irradiance - {self.track.name}")
        ax1.grid(True, alpha=0.3)
        
        # Cloud cover plot
        ax2.plot(hours, self.results.cloud_cover * 100, color='gray', linewidth=1.5)
        ax2.fill_between(hours, 0, self.results.cloud_cover * 100, alpha=0.3, color='gray')
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Cloud Cover (%)")
        ax2.set_title("Cloud Cover")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close()
    
    def plot_dashboard(self, save_path: Optional[str] = None, show: bool = False):
        """Generate a comprehensive dashboard with all metrics."""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        hours = self._time_to_hours()
        
        # SoC plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(hours, self.results.soc * 100, label="Actual", linewidth=2)
        ax1.plot(hours, self.results.ideal_soc * 100, '--', label="Target", alpha=0.7)
        ax1.set_ylabel("SoC (%)")
        ax1.set_title("Battery State of Charge")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Speed plot
        ax2 = fig.add_subplot(gs[0, 1])
        speed_mph = self.results.speed * 2.237
        ax2.plot(hours, speed_mph, color='green', linewidth=1.5)
        ax2.axhline(y=np.mean(speed_mph), color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel("Speed (mph)")
        ax2.set_title(f"Speed (Avg: {np.mean(speed_mph):.1f} mph)")
        ax2.grid(True, alpha=0.3)
        
        # GHI plot
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(hours, self.results.ghi, color='orange', linewidth=1.5)
        ax3.set_ylabel("GHI (W/m²)")
        ax3.set_title("Solar Irradiance")
        ax3.grid(True, alpha=0.3)
        
        # Cloud cover plot
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.fill_between(hours, 0, self.results.cloud_cover * 100, alpha=0.5, color='gray')
        ax4.set_ylabel("Cloud Cover (%)")
        ax4.set_title("Cloud Cover")
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # Battery drain rate
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(hours, self.results.bdr, color='red', linewidth=1.5)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_xlabel("Time (hours)")
        ax5.set_ylabel("BDR (1/hr)")
        ax5.set_title("Battery Drain Rate")
        ax5.grid(True, alpha=0.3)
        
        # Race summary text
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        summary_text = f"""
        RACE SUMMARY - {self.track.name}
        {'='*40}
        
        Total Laps: {self.results.total_laps}
        Total Distance: {self.results.total_distance_km:.2f} km 
                       ({self.results.total_distance_miles:.2f} miles)
        
        Final SoC: {self.results.final_soc*100:.1f}%
        
        Speed Statistics:
          • Average: {self.results.avg_speed_mph:.1f} mph
          • Max: {np.max(self.results.speed)*2.237:.1f} mph
          • Min: {np.min(self.results.speed)*2.237:.1f} mph
        
        Lap Times (minutes):
          • Average: {np.mean(self.results.lap_times):.1f}
          • Fastest: {np.min(self.results.lap_times):.1f}
          • Slowest: {np.max(self.results.lap_times):.1f}
        """
        
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f"Solar Car Laps Race Simulation", fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


# =============================================================================
# Results Reporter
# =============================================================================

class ResultsReporter:
    """Generates text reports from simulation results."""
    
    def __init__(self, results: SimulationResults, track: TrackConfig, race: RaceConfig):
        self.results = results
        self.track = track
        self.race = race
    
    def print_summary(self):
        """Print a summary of the race simulation."""
        r = self.results
        
        print("\n" + "=" * 80)
        print(f"  LAPS RACE SIMULATION RESULTS - {self.track.name}")
        print("=" * 80)
        
        print(f"\n📍 Track: {self.track.name}")
        print(f"   Location: {self.track.location}")
        print(f"   Lap Distance: {self.track.lap_distance_km:.3f} km ({self.track.lap_distance_km/1.609:.2f} miles)")
        
        print(f"\n🏁 Race Results:")
        print(f"   Total Laps Completed: {r.total_laps}")
        print(f"   Total Distance: {r.total_distance_km:.2f} km ({r.total_distance_miles:.2f} miles)")
        
        print(f"\n🔋 Battery:")
        print(f"   Starting SoC: {self.race.start_soc*100:.0f}%")
        print(f"   Final SoC: {r.final_soc*100:.1f}%")
        print(f"   Target SoC: {self.race.target_soc*100:.0f}%")
        print(f"   SoC Error: {abs(r.final_soc - self.race.target_soc)*100:.1f}%")
        
        print(f"\n🚗 Speed:")
        print(f"   Average: {r.avg_speed_mps:.2f} m/s ({r.avg_speed_mph:.1f} mph)")
        print(f"   Maximum: {np.max(r.speed):.2f} m/s ({np.max(r.speed)*2.237:.1f} mph)")
        print(f"   Minimum: {np.min(r.speed):.2f} m/s ({np.min(r.speed)*2.237:.1f} mph)")
        
        if r.lap_times:
            print(f"\n⏱️  Lap Times (minutes):")
            print(f"   Average: {np.mean(r.lap_times):.2f}")
            print(f"   Fastest: {np.min(r.lap_times):.2f}")
            print(f"   Slowest: {np.max(r.lap_times):.2f}")
            print(f"\n   Individual Laps:")
            for i, lap_time in enumerate(r.lap_times, 1):
                print(f"     Lap {i}: {lap_time:.2f} min")
        
        print("\n" + "=" * 80 + "\n")
    
    def print_detailed_log(self, interval: int = 10):
        """Print detailed simulation log at specified minute intervals."""
        print("\nDETAILED SIMULATION LOG:")
        print("-" * 100)
        print(f"{'Time':>8} | {'SoC':>8} | {'Speed':>12} | {'GHI':>10} | {'Cloud':>8} | {'BDR':>8}")
        print(f"{'(min)':>8} | {'(%)':>8} | {'(m/s | mph)':>12} | {'(W/m²)':>10} | {'(%)':>8} | {'(/hr)':>8}")
        print("-" * 100)
        
        r = self.results
        for i in range(0, len(r.time_minutes), interval):
            t = r.time_minutes[i]
            print(f"{t:>8.0f} | {r.soc[i]*100:>7.1f}% | "
                  f"{r.speed[i]:>5.1f} | {r.speed[i]*2.237:>4.1f} | "
                  f"{r.ghi[i]:>10.1f} | {r.cloud_cover[i]*100:>7.1f}% | "
                  f"{r.bdr[i]:>+7.3f}")
        
        print("-" * 100)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run the laps race simulation with default configuration."""
    
    # Load car parameters from JSON or use defaults
    try:
        car = CarConfig.from_json("car_params.json")
        print("✓ Loaded car parameters from car_params.json")
    except FileNotFoundError:
        car = CarConfig()
        print("⚠ Using default car parameters")
    
    # Configure race
    track = TrackConfig()
    race = RaceConfig(
        start_soc=1.0,
        target_soc=0.10,
        aggressiveness=1.5,     # moderate PI correction
        initial_speed_mps=20.0,  # fallback only; optimal speed is computed
        time_step_minutes=1.0
    )
    
    print(f"\n🏎️  Starting Laps Race Simulation")
    print(f"   Track: {track.name}")
    print(f"   Lap Distance: {track.lap_distance_km:.3f} km")
    
    # Run simulation
    simulator = LapsRaceSimulator(
        track=track,
        car=car,
        race=race,
        use_api_weather=True
    )
    
    results = simulator.run()
    
    # Generate reports
    reporter = ResultsReporter(results, track, race)
    reporter.print_summary()
    
    # Generate plots
    plotter = SimulationPlotter(results, track)
    plotter.plot_dashboard(save_path="plots/laps/laps_race_dashboard.png")
    plotter.plot_soc(save_path="plots/laps/laps_soc_plot.png")
    plotter.plot_speed(save_path="plots/laps/laps_speed_plot.png")
    plotter.plot_weather(save_path="plots/laps/laps_weather_plot.png")
    
    print("📊 Plots saved:")
    print("   - plots/laps/laps_race_dashboard.png")
    print("   - plots/laps/laps_soc_plot.png")
    print("   - plots/laps/laps_speed_plot.png")
    print("   - plots/laps/laps_weather_plot.png")

    
    return results


if __name__ == "__main__":
    results = main()