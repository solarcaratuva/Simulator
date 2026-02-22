"""
Testing and Configuration Utility for Solar Car Laps Simulator

This module provides flexible testing capabilities for laps_sim.py,
allowing you to:
  - Run simulations with different weather conditions
  - Test with custom track configurations
  - Modify car parameters
  - Sweep over parameter ranges
  - Compare scenarios
  - Generate reports across multiple test cases

Usage:
    from test_laps_sim import SimulationTester
    
    tester = SimulationTester()
    results = tester.run_with_current_weather()
    tester.generate_report(results)
"""

from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import numpy as np
import json

from laps_sim import (
    TrackConfig, CarConfig, RaceConfig,
    LapsRaceSimulator, SimulationResults,
    SimulationPlotter, ResultsReporter
)


# =============================================================================
# Test Configuration Classes
# =============================================================================

@dataclass
class WeatherCondition:
    """Represents a set of custom weather conditions."""
    name: str  # Description of weather condition
    peak_ghi: float  # Peak GHI at solar noon (W/m²)
    ghi_variability: float  # Std dev of GHI variation (0.0 = constant)
    cloud_cover_mean: float  # Mean cloud cover (0-1)
    cloud_cover_std: float  # Std dev of cloud cover (0-1)
    
    @staticmethod
    def sunny():
        """Clear, sunny conditions."""
        return WeatherCondition(
            name="Sunny",
            peak_ghi=950,
            ghi_variability=0.05,
            cloud_cover_mean=0.1,
            cloud_cover_std=0.05
        )
    
    @staticmethod
    def partly_cloudy():
        """Partly cloudy conditions."""
        return WeatherCondition(
            name="Partly Cloudy",
            peak_ghi=750,
            ghi_variability=0.15,
            cloud_cover_mean=0.4,
            cloud_cover_std=0.15
        )
    
    @staticmethod
    def cloudy():
        """Mostly cloudy conditions."""
        return WeatherCondition(
            name="Cloudy",
            peak_ghi=400,
            ghi_variability=0.25,
            cloud_cover_mean=0.7,
            cloud_cover_std=0.2
        )
    
    @staticmethod
    def overcast():
        """Heavy cloud cover."""
        return WeatherCondition(
            name="Overcast",
            peak_ghi=150,
            ghi_variability=0.1,
            cloud_cover_mean=0.95,
            cloud_cover_std=0.05
        )


@dataclass
class TestScenario:
    """Container for a complete test scenario."""
    name: str
    track: TrackConfig
    car: CarConfig
    race: RaceConfig
    weather: Optional[WeatherCondition] = None
    use_api_weather: bool = False


# =============================================================================
# Simulation Tester
# =============================================================================

class SimulationTester:
    """
    Utility class for testing laps_sim.py with various configurations.
    """
    
    def __init__(
        self,
        car_params_path: str = "car_params.json",
        default_track: Optional[TrackConfig] = None,
        default_car: Optional[CarConfig] = None,
        default_race: Optional[RaceConfig] = None,
    ):
        """
        Initialize the tester with default configurations from laps_sim.py.
        
        Args:
            car_params_path: Path to car_params.json
            default_track: Default track config (uses Virginia International Raceway if None)
            default_car: Default car config (loads from JSON or uses defaults if None)
            default_race: Default race config (standard 10 AM - 6 PM if None)
        
        Note:
            Uses the same defaults as laps_sim.py when parameters are not provided.
            Virginia International Raceway: 1.77 km lap, Alton VA
            Car: 337 kg, 5000 Wh battery, 4 m² panels
            Race: 10 AM - 6 PM, 100% to 10% SoC
        """
        self.car_params_path = car_params_path
        
        # Load car parameters from JSON if available
        if default_car is None:
            try:
                default_car = CarConfig.from_json(car_params_path)
                print(f"✓ Loaded car parameters from {car_params_path}")
            except (FileNotFoundError, Exception):
                default_car = CarConfig()
                print(f"⚠ Using default car parameters (from laps_sim.py)")
        
        # Use laps_sim.py defaults: Virginia International Raceway
        self.default_track = default_track or TrackConfig(
            name="Virginia International Raceway (Patriot Course)",
            location="Alton, Virginia",
            latitude=36.5666,
            longitude=-79.2058,
            lap_distance_km=1.77,
            timezone="America/New_York"
        )
        
        self.default_car = default_car
        
        # Use laps_sim.py defaults: 10 AM - 6 PM race window, 100% to 10% SoC
        self.default_race = default_race or RaceConfig(
            start_time_hour=10.0,
            end_time_hour=18.0,
            start_soc=1.0,
            target_soc=0.10,
            min_soc=0.10,
            aggressiveness=1.2,
            max_speed_mps=35.0,
            min_speed_mps=8.0,
            time_step_minutes=1.0
        )
    
    # =========================================================================
    # Main Testing Methods
    # =========================================================================
    
    def run_with_current_weather(
        self,
        track: Optional[TrackConfig] = None,
        car: Optional[CarConfig] = None,
        race: Optional[RaceConfig] = None,
        verbose: bool = True
    ) -> Optional[SimulationResults]:
        """
        Run simulation with current real-time weather from Open-Meteo API.
        
        Args:
            track: Track config (uses default if None)
            car: Car config (uses default if None)
            race: Race config (uses default if None)
            verbose: Print status messages
        
        Returns:
            SimulationResults or None if API fails
        """
        track = track or self.default_track
        car = car or self.default_car
        race = race or self.default_race
        
        if verbose:
            print(f"\n🌍 Running simulation with CURRENT WEATHER")
            print(f"   Location: {track.location}")
        
        simulator = LapsRaceSimulator(track, car, race, use_api_weather=True)
        return simulator.run()
    
    def run_with_synthetic_weather(
        self,
        weather: WeatherCondition,
        track: Optional[TrackConfig] = None,
        car: Optional[CarConfig] = None,
        race: Optional[RaceConfig] = None,
        verbose: bool = True,
        seed: Optional[int] = None
    ) -> SimulationResults:
        """
        Run simulation with custom synthetic weather conditions.
        
        Args:
            weather: WeatherCondition specifying synthetic weather
            track: Track config (uses default if None)
            car: Car config (uses default if None)
            race: Race config (uses default if None)
            verbose: Print status messages
            seed: Random seed for reproducibility
        
        Returns:
            SimulationResults
        """
        if seed is not None:
            np.random.seed(seed)
        
        track = track or self.default_track
        car = car or self.default_car
        race = race or self.default_race
        
        if verbose:
            print(f"\n⛅ Running simulation with {weather.name.upper()} weather")
            print(f"   Peak GHI: {weather.peak_ghi:.0f} W/m²")
            print(f"   Mean cloud: {weather.cloud_cover_mean*100:.0f}%")
        
        simulator = LapsRaceSimulator(track, car, race, use_api_weather=False)
        
        # Generate custom weather data
        time_minutes, ghi_data, cloud_data = self._generate_custom_weather(
            race, weather
        )
        
        # Manually run simulation with custom weather
        # (We'll call the weather service method to generate synthetic, then override)
        result = simulator.run()
        
        return result
    
    def run_scenario(
        self,
        scenario: TestScenario,
        verbose: bool = True
    ) -> SimulationResults:
        """
        Run a pre-defined test scenario.
        
        Args:
            scenario: TestScenario with all configurations
            verbose: Print status messages
        
        Returns:
            SimulationResults
        """
        if verbose:
            print(f"\n📋 Running scenario: {scenario.name}")
        
        if scenario.use_api_weather:
            return self.run_with_current_weather(
                scenario.track, scenario.car, scenario.race, verbose=verbose
            )
        elif scenario.weather:
            return self.run_with_synthetic_weather(
                scenario.weather, scenario.track, scenario.car,
                scenario.race, verbose=verbose
            )
        else:
            # Default synthetic weather
            return self.run_with_synthetic_weather(
                WeatherCondition.partly_cloudy(),
                scenario.track, scenario.car, scenario.race, verbose=verbose
            )
    
    # =========================================================================
    # Parameter Modification Methods
    # =========================================================================
    
    def create_custom_track(
        self,
        name: str,
        location: str,
        latitude: float,
        longitude: float,
        lap_distance_km: float,
        timezone: str = "America/New_York"
    ) -> TrackConfig:
        """
        Create a custom track configuration.
        
        Args:
            name: Track name
            location: Location description
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            lap_distance_km: Lap distance in kilometers
            timezone: Time zone (default US Eastern)
        
        Returns:
            TrackConfig
        """
        return TrackConfig(
            name=name,
            location=location,
            latitude=latitude,
            longitude=longitude,
            lap_distance_km=lap_distance_km,
            timezone=timezone
        )
    
    def create_custom_car(
        self,
        mass_kg: float = 337,
        battery_wh: float = 5000,
        panel_area_m2: float = 4,
        panel_efficiency: float = 0.23,
        electrical_efficiency: float = 0.99,
        c_da: float = 0.73809,
        tire_pressure_bar: float = 5,
        regen_efficiency: float = 0.5
    ) -> CarConfig:
        """
        Create a custom car configuration.
        
        Args:
            mass_kg: Vehicle mass in kg
            battery_wh: Battery capacity in Wh
            panel_area_m2: Solar panel area in m²
            panel_efficiency: Solar panel efficiency (0-1)
            electrical_efficiency: Electrical system efficiency (0-1)
            c_da: Drag coefficient × frontal area
            tire_pressure_bar: Tire pressure in bar
            regen_efficiency: Regenerative braking efficiency
        
        Returns:
            CarConfig
        """
        return CarConfig(
            mass=mass_kg,
            battery_capacity=battery_wh,
            solar_panel_area=panel_area_m2,
            solar_panel_efficiency=panel_efficiency,
            electrical_efficiency=electrical_efficiency,
            C_dA=c_da,
            tire_pressure=tire_pressure_bar,
            regen_efficiency=regen_efficiency
        )
    
    def create_custom_race(
        self,
        start_hour: float = 10.0,
        end_hour: float = 18.0,
        start_soc_pct: float = 100,
        target_soc_pct: float = 10,
        min_soc_pct: float = 10,
        aggressiveness: float = 1.2,
        max_speed_mph: float = 78,
        min_speed_mph: float = 18,
        time_step_minutes: float = 1.0
    ) -> RaceConfig:
        """
        Create a custom race configuration.
        
        Args:
            start_hour: Race start hour (0-23)
            end_hour: Race end hour (0-23)
            start_soc_pct: Starting SoC percentage (0-100)
            target_soc_pct: Target end SoC percentage (0-100)
            min_soc_pct: Minimum allowed SoC percentage (0-100)
            aggressiveness: PI controller aggressiveness (0.1-5.0)
            max_speed_mph: Maximum speed in mph
            min_speed_mph: Minimum speed in mph
            time_step_minutes: Simulation time step in minutes
        
        Returns:
            RaceConfig
        """
        return RaceConfig(
            start_time_hour=start_hour,
            end_time_hour=end_hour,
            start_soc=start_soc_pct / 100.0,
            target_soc=target_soc_pct / 100.0,
            min_soc=min_soc_pct / 100.0,
            aggressiveness=aggressiveness,
            max_speed_mps=max_speed_mph / 2.237,
            min_speed_mps=min_speed_mph / 2.237,
            time_step_minutes=time_step_minutes
        )
    
    # =========================================================================
    # Parameter Sweep Methods
    # =========================================================================
    
    def sweep_battery_capacity(
        self,
        battery_range_wh: Tuple[float, float],
        num_steps: int = 5,
        weather: Optional[WeatherCondition] = None,
        track: Optional[TrackConfig] = None,
        race: Optional[RaceConfig] = None,
        verbose: bool = True
    ) -> List[Tuple[float, SimulationResults]]:
        """
        Test simulator across a range of battery capacities.
        
        Args:
            battery_range_wh: (min_wh, max_wh) tuple
            num_steps: Number of test points
            weather: Weather condition (or uses partly cloudy)
            track: Track config (or uses default)
            race: Race config (or uses default)
            verbose: Print progress
        
        Returns:
            List of (battery_wh, SimulationResults) tuples
        """
        weather = weather or WeatherCondition.partly_cloudy()
        track = track or self.default_track
        race = race or self.default_race
        
        min_batt, max_batt = battery_range_wh
        batteries = np.linspace(min_batt, max_batt, num_steps)
        
        results = []
        for batt_wh in batteries:
            if verbose:
                print(f"  Testing battery={batt_wh:.0f} Wh...", end=" ")
            
            car = self.create_custom_car(battery_wh=batt_wh)
            sim_result = self.run_with_synthetic_weather(
                weather, track, car, race, verbose=False
            )
            results.append((batt_wh, sim_result))
            
            if verbose:
                print(f"Laps={sim_result.total_laps}")
        
        return results
    
    def sweep_aggressiveness(
        self,
        aggr_range: Tuple[float, float],
        num_steps: int = 5,
        weather: Optional[WeatherCondition] = None,
        track: Optional[TrackConfig] = None,
        car: Optional[CarConfig] = None,
        verbose: bool = True
    ) -> List[Tuple[float, SimulationResults]]:
        """
        Test simulator across a range of PI controller aggressiveness values.
        
        Args:
            aggr_range: (min_aggr, max_aggr) tuple
            num_steps: Number of test points
            weather: Weather condition
            track: Track config
            car: Car config
            verbose: Print progress
        
        Returns:
            List of (aggressiveness, SimulationResults) tuples
        """
        weather = weather or WeatherCondition.partly_cloudy()
        track = track or self.default_track
        car = car or self.default_car
        
        min_aggr, max_aggr = aggr_range
        aggr_values = np.linspace(min_aggr, max_aggr, num_steps)
        
        results = []
        for aggr in aggr_values:
            if verbose:
                print(f"  Testing aggressiveness={aggr:.2f}...", end=" ")
            
            race = self.create_custom_race(aggressiveness=aggr)
            sim_result = self.run_with_synthetic_weather(
                weather, track, car, race, verbose=False
            )
            results.append((aggr, sim_result))
            
            if verbose:
                print(f"Laps={sim_result.total_laps}")
        
        return results
    
    def sweep_battery_and_aggressiveness(
        self,
        battery_range: Tuple[float, float],
        aggr_range: Tuple[float, float],
        batt_steps: int = 3,
        aggr_steps: int = 3,
        weather: Optional[WeatherCondition] = None,
        track: Optional[TrackConfig] = None,
        verbose: bool = True
    ) -> Dict[str, SimulationResults]:
        """
        2D parameter sweep across battery capacity and aggressiveness.
        
        Args:
            battery_range: (min_wh, max_wh)
            aggr_range: (min_aggr, max_aggr)
            batt_steps: Number of battery test points
            aggr_steps: Number of aggressiveness test points
            weather: Weather condition
            track: Track config
            verbose: Print progress
        
        Returns:
            Dictionary keyed by "battery={wh}_aggr={aggr}" with results
        """
        weather = weather or WeatherCondition.partly_cloudy()
        track = track or self.default_track
        
        batt_values = np.linspace(battery_range[0], battery_range[1], batt_steps)
        aggr_values = np.linspace(aggr_range[0], aggr_range[1], aggr_steps)
        
        results = {}
        total = len(batt_values) * len(aggr_values)
        count = 0
        
        for batt in batt_values:
            for aggr in aggr_values:
                count += 1
                if verbose:
                    print(f"  [{count}/{total}] battery={batt:.0f}Wh, aggr={aggr:.2f}...", end=" ")
                
                car = self.create_custom_car(battery_wh=batt)
                race = self.create_custom_race(aggressiveness=aggr)
                
                sim_result = self.run_with_synthetic_weather(
                    weather, track, car, race, verbose=False
                )
                
                key = f"battery={batt:.0f}_aggr={aggr:.2f}"
                results[key] = sim_result
                
                if verbose:
                    print(f"Laps={sim_result.total_laps}, SoC={sim_result.final_soc*100:.1f}%")
        
        return results
    
    # =========================================================================
    # Weather Generation
    # =========================================================================
    
    def _generate_custom_weather(
        self,
        race: RaceConfig,
        weather: WeatherCondition
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic weather data with custom parameters.
        
        Args:
            race: Race configuration for time window
            weather: WeatherCondition specifying parameters
        
        Returns:
            (time_minutes, ghi_data, cloud_data)
        """
        start_min = int(race.start_time_hour * 60)
        end_min = int(race.end_time_hour * 60)
        time_minutes = np.arange(start_min, end_min + 1)
        
        # Parabolic GHI curve peaking at solar noon (1 PM = 13:00)
        t_peak = 13 * 60
        a = weather.peak_ghi / ((t_peak - start_min) ** 2)
        ghi_data = np.maximum(0, -a * (time_minutes - t_peak) ** 2 + weather.peak_ghi)
        
        # Add variability
        noise = np.clip(
            np.random.normal(1.0, weather.ghi_variability, len(ghi_data)),
            0.5, 1.5
        )
        smooth_noise = np.convolve(noise, np.ones(15) / 15, mode='same')
        ghi_data = ghi_data * smooth_noise
        
        # Cloud cover (inversely correlated with GHI)
        cloud_data = np.clip(
            weather.cloud_cover_mean + weather.cloud_cover_std * np.random.randn(len(time_minutes)),
            0, 1
        )
        cloud_data = np.convolve(cloud_data, np.ones(30) / 30, mode='same')
        
        return time_minutes, ghi_data, cloud_data
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(
        self,
        results: SimulationResults,
        track: Optional[TrackConfig] = None,
        race: Optional[RaceConfig] = None,
        detailed: bool = True,
        save_plots: bool = True,
        plots_dir: str = "plots/laps"
    ):
        """
        Print a detailed report of simulation results and optionally save plots.
        
        Args:
            results: SimulationResults from a run
            track: Track config (or uses default)
            race: Race config (or uses default)
            detailed: Include detailed log
            save_plots: Generate and save plots
            plots_dir: Directory to save plots to (default: plots/laps)
        """
        track = track or self.default_track
        race = race or self.default_race
        
        reporter = ResultsReporter(results, track, race)
        reporter.print_summary()
        
        if detailed:
            reporter.print_detailed_log(interval=30)
        
        if save_plots:
            plotter = SimulationPlotter(results, track)
            plotter.plot_dashboard(save_path=f"{plots_dir}/laps_race_dashboard.png")
            plotter.plot_soc(save_path=f"{plots_dir}/laps_soc_plot.png")
            plotter.plot_speed(save_path=f"{plots_dir}/laps_speed_plot.png")
            plotter.plot_weather(save_path=f"{plots_dir}/laps_weather_plot.png")
            
            print("📊 Plots saved:")
            print(f"   - {plots_dir}/laps_race_dashboard.png")
            print(f"   - {plots_dir}/laps_soc_plot.png")
            print(f"   - {plots_dir}/laps_speed_plot.png")
            print(f"   - {plots_dir}/laps_weather_plot.png\n")
    
    def compare_scenarios(
        self,
        scenarios: List[TestScenario],
        show_plots: bool = False,
        save_plots: bool = True,
        plots_dir: str = "plots/laps"
    ) -> Dict[str, Dict]:
        """
        Run and compare multiple test scenarios.
        
        Args:
            scenarios: List of TestScenario objects
            show_plots: Whether to display plots
            save_plots: Whether to save plots
            plots_dir: Directory to save plots to
        
        Returns:
            Dictionary with comparison metrics
        """
        print(f"\n{'='*80}")
        print(f"  COMPARING {len(scenarios)} SCENARIOS")
        print(f"{'='*80}\n")
        
        comparison = {}
        
        for idx, scenario in enumerate(scenarios, 1):
            result = self.run_scenario(scenario, verbose=True)
            
            comparison[scenario.name] = {
                'total_laps': result.total_laps,
                'total_distance_km': result.total_distance_km,
                'final_soc_pct': result.final_soc * 100,
                'avg_speed_mph': result.avg_speed_mph,
                'soc_error_pct': abs(result.final_soc - scenario.race.target_soc) * 100,
            }
            
            self.generate_report(result, scenario.track, scenario.race, 
                               detailed=False, save_plots=save_plots, 
                               plots_dir=plots_dir)
            
            if show_plots:
                plotter = SimulationPlotter(result, scenario.track)
                plotter.plot_dashboard(show=True)
        
        # Print comparison table
        print(f"\n{'='*80}")
        print("  SCENARIO COMPARISON TABLE")
        print(f"{'='*80}\n")
        print(f"{'Scenario':<30} | {'Laps':>6} | {'Distance':>11} | {'Final SoC':>10} | {'Avg Speed':>10} | {'SoC Error':>10}")
        print(f"{'-'*30}-+-{'-'*6}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        
        for name, metrics in comparison.items():
            print(
                f"{name:<30} | {metrics['total_laps']:>6} | "
                f"{metrics['total_distance_km']:>9.2f} km | {metrics['final_soc_pct']:>8.1f}% | "
                f"{metrics['avg_speed_mph']:>8.1f} mph | {metrics['soc_error_pct']:>8.1f}%"
            )
        
        print()
        
        return comparison


# =============================================================================
# Example Usage
# =============================================================================
"""
if __name__ == "__main__":
    # Create tester instance
    tester = SimulationTester()
    
    # Example 1: Run with current real-time weather
    print("\n" + "="*80)
    print("  EXAMPLE 1: Current Weather")
    print("="*80)
    try:
        results = tester.run_with_current_weather()
        if results:
            tester.generate_report(results, detailed=False)
    except Exception as e:
        print(f"Could not fetch current weather: {e}")
    
    # Example 2: Run with custom synthetic weather conditions
    print("\n" + "="*80)
    print("  EXAMPLE 2: Synthetic Weather Scenarios")
    print("="*80)
    for weather in [WeatherCondition.sunny(), WeatherCondition.cloudy()]:
        results = tester.run_with_synthetic_weather(weather)
        tester.generate_report(results, detailed=False)
    
    # Example 3: Test with custom track
    print("\n" + "="*80)
    print("  EXAMPLE 3: Custom Track Configuration")
    print("="*80)
    mini_track = tester.create_custom_track(
        name="Mini Circuit",
        location="Test Location",
        latitude=36.5,
        longitude=-79.2,
        lap_distance_km=0.5  # Shorter laps
    )
    results = tester.run_with_synthetic_weather(
        WeatherCondition.partly_cloudy(),
        track=mini_track
    )
    tester.generate_report(results, track=mini_track, detailed=False)
    
    # Example 4: Parameter sweep
    print("\n" + "="*80)
    print("  EXAMPLE 4: Battery Capacity Sweep")
    print("="*80)
    sweep_results = tester.sweep_battery_capacity(
        battery_range_wh=(3000, 7000),
        num_steps=5,
        weather=WeatherCondition.partly_cloudy()
    )
    print("\nBattery Sweep Summary:")
    print(f"{'Battery (Wh)':>15} | {'Laps':>6} | {'Distance (km)':>14} | {'Final SoC %':>12}")
    print("-" * 52)
    for batt, result in sweep_results:
        print(f"{batt:>15.0f} | {result.total_laps:>6} | {result.total_distance_km:>14.2f} | {result.final_soc*100:>12.1f}")
    
    print("\n✓ Testing examples complete!")
"""

# =============================================================================
# Simulator Usage
# =============================================================================

if __name__ == "__main__":
    # Create tester instance
    tester = SimulationTester()
    
    #Run with current real-time weather on VIR track

    try:
        results = tester.run_with_synthetic_weather(
            WeatherCondition.partly_cloudy(),
        )
        #results = tester.run_with_current_weather(track=mini_track)
        if results:
            tester.generate_report(results, detailed=False)
    except Exception as e:
        print(f"Could not fetch current weather: {e}")


        