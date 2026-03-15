from dataclasses import dataclass, field
from typing import Dict, List
import json
import math
import numpy as np


@dataclass
class TrackConfig:
    name: str
    location: str
    latitude: float
    longitude: float
    lap_distance_km: float
    timezone: str
    lap_distance_m: float = field(init=False)

    def __post_init__(self):
        self.lap_distance_m = self.lap_distance_km * 1000.0


@dataclass
class CarConfig:
    mass: float = 337
    battery_capacity: float = 5000
    solar_panel_area: float = 4
    solar_panel_efficiency: float = 0.23
    electrical_efficiency: float = 0.99
    C_dA: float = 0.73809
    tire_pressure: float = 5
    regen_efficiency: float = 0.5
    rho: float = 1.192
    g: float = 9.80665

    @classmethod
    def from_json(cls, filepath: str) -> "CarConfig":
        with open(filepath, "r") as f:
            params = json.load(f)

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
            g=params.get("g", 9.80665),
        )


@dataclass
class RaceConfig:
    start_time_hour: float = 10.0
    end_time_hour: float = 18.0
    start_soc: float = 1.0
    target_soc: float = 0.10
    min_soc: float = 0.10
    aggressiveness: float = 1.2
    initial_speed_mps: float = 20.0
    max_speed_mps: float = 35.0
    min_speed_mps: float = 8.0
    time_step_minutes: float = 1.0
    strategy: str = "pi"


@dataclass
class SimulationResults:
    time_minutes: np.ndarray
    soc: np.ndarray
    speed: np.ndarray
    ghi: np.ndarray
    cloud_cover: np.ndarray
    lap_times: List[float]
    total_laps: int
    total_distance_m: float
    bdr: np.ndarray
    ideal_soc: np.ndarray

    @property
    def total_distance_km(self) -> float:
        return self.total_distance_m / 1000.0

    @property
    def total_distance_miles(self) -> float:
        return self.total_distance_m / 1609.34

    @property
    def avg_speed_mps(self) -> float:
        return float(np.mean(self.speed))

    @property
    def avg_speed_mph(self) -> float:
        return self.avg_speed_mps * 2.237

    @property
    def final_soc(self) -> float:
        return float(self.soc[-1])


def get_available_tracks() -> Dict[str, TrackConfig]:
    return {
        "shenandoah": TrackConfig(
            name="Shenandpah Speedway",
            location="Page County, Virginia",
            latitude=38.5110,
            longitude=-78.6359,
            lap_distance_km=0.604,
            timezone="America/New_York",
        )
    }
