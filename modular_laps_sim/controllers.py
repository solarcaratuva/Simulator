from abc import ABC, abstractmethod
import numpy as np


def compute_ideal_soc_curve(cloud_data: np.ndarray, race_config, alpha: float = 0.5) -> np.ndarray:
    n_steps = len(cloud_data)
    soc = np.zeros(n_steps)
    soc[0] = race_config.start_soc
    soc_drop_total = race_config.start_soc - race_config.target_soc

    cloud_factors = (1 - cloud_data) * alpha + (1 - alpha)
    total_weighted = np.sum(cloud_factors)
    if total_weighted <= 0:
        total_weighted = 1.0
    normalized_drops = cloud_factors * (soc_drop_total / total_weighted)

    for i in range(1, n_steps):
        soc[i] = soc[i - 1] - normalized_drops[i]
        soc[i] = max(soc[i], race_config.target_soc)

    return soc


def solve_optimal_constant_speed(physics, race_config, ghi_data: np.ndarray, dt_hours: float) -> float:
    cap = physics.car.battery_capacity
    usable_energy = (race_config.start_soc - race_config.target_soc) * cap
    total_solar = np.sum(physics.solar_power(ghi_data)) * dt_hours
    total_available = usable_energy + total_solar
    n_steps = len(ghi_data)

    v_lo, v_hi = race_config.min_speed_mps, race_config.max_speed_mps
    for _ in range(60):
        v_mid = (v_lo + v_hi) / 2.0
        cost = n_steps * physics.power_drained(v_mid) * dt_hours
        if cost < total_available:
            v_lo = v_mid
        else:
            v_hi = v_mid
    return (v_lo + v_hi) / 2.0


class BaseStrategy(ABC):
    def __init__(self, race_config):
        self.config = race_config
        self.optimal_speed = None

    def prepare(self, physics, ghi_data: np.ndarray, cloud_data: np.ndarray, dt_hours: float):
        self.optimal_speed = solve_optimal_constant_speed(physics, self.config, ghi_data, dt_hours)
        self.ideal_soc = compute_ideal_soc_curve(cloud_data, self.config)

    @abstractmethod
    def next_speed(self, step: int, current_speed: float, current_soc: float, soc_error: float, bdr: float, physics, ghi_data: np.ndarray, dt_hours: float) -> float:
        pass

    def clamp(self, speed: float) -> float:
        return float(np.clip(speed, self.config.min_speed_mps, self.config.max_speed_mps))


class PIControllerStrategy(BaseStrategy):
    name = "pi"

    def __init__(self, race_config):
        super().__init__(race_config)
        self.integral_error = 0.0

    def next_speed(self, step: int, current_speed: float, current_soc: float, soc_error: float, bdr: float, physics, ghi_data: np.ndarray, dt_hours: float) -> float:
        base_speed = self.optimal_speed if self.optimal_speed is not None else current_speed
        agg = self.config.aggressiveness
        kp = 0.08 * agg
        ki = 0.002 * agg

        self.integral_error += soc_error
        self.integral_error = float(np.clip(self.integral_error, -5.0, 5.0))

        correction = kp * soc_error + ki * self.integral_error
        correction = float(np.clip(correction, -0.15, 0.15))
        speed = base_speed * (1.0 + correction)

        margin = 0.05
        soc_above_min = current_soc - self.config.min_soc
        if soc_above_min < margin:
            urgency = 1.0 - max(soc_above_min, 0.0) / margin
            speed = speed * (1 - urgency) + self.config.min_speed_mps * urgency

        return self.clamp(speed)


class SteppedIfElseStrategy(BaseStrategy):
    name = "stepped"

    def next_speed(self, step: int, current_speed: float, current_soc: float, soc_error: float, bdr: float, physics, ghi_data: np.ndarray, dt_hours: float) -> float:
        agg = self.config.aggressiveness
        step_small = 0.25 * agg
        step_large = 0.60 * agg
        speed = current_speed

        if soc_error > 0.03:
            speed += step_large
        elif soc_error > 0.01:
            speed += step_small
        elif soc_error < -0.03:
            speed -= step_large
        elif soc_error < -0.01:
            speed -= step_small

        if current_soc < self.config.min_soc + 0.03:
            speed -= step_large

        return self.clamp(speed)


class IntervalHoldStrategy(BaseStrategy):
    name = "interval-hold"

    def __init__(self, race_config, interval_minutes: int = 5):
        super().__init__(race_config)
        self.interval_steps = max(1, int(interval_minutes / race_config.time_step_minutes))
        self.held_speed = None

    def _solve_remaining_speed(self, current_soc: float, physics, ghi_remaining: np.ndarray, dt_hours: float) -> float:
        cap = physics.car.battery_capacity
        usable_energy = max(0.0, (current_soc - self.config.target_soc) * cap)
        solar_energy = np.sum(physics.solar_power(ghi_remaining)) * dt_hours
        available = usable_energy + solar_energy

        n_steps = len(ghi_remaining)
        if n_steps == 0:
            return self.config.min_speed_mps

        v_lo, v_hi = self.config.min_speed_mps, self.config.max_speed_mps
        for _ in range(60):
            v_mid = (v_lo + v_hi) / 2.0
            cost = n_steps * physics.power_drained(v_mid) * dt_hours
            if cost < available:
                v_lo = v_mid
            else:
                v_hi = v_mid
        return (v_lo + v_hi) / 2.0

    def next_speed(self, step: int, current_speed: float, current_soc: float, soc_error: float, bdr: float, physics, ghi_data: np.ndarray, dt_hours: float) -> float:
        should_update = self.held_speed is None or step % self.interval_steps == 0
        if should_update:
            remaining_ghi = ghi_data[step:]
            budget_speed = self._solve_remaining_speed(current_soc, physics, remaining_ghi, dt_hours)

            weather_factor = 1.0
            if step < len(ghi_data):
                # Keep speed flatter in low irradiance periods, increase in strong sun.
                weather_factor = float(np.clip(0.9 + 0.2 * (ghi_data[step] / 900.0), 0.85, 1.1))

            soc_factor = 1.0 + float(np.clip(soc_error * self.config.aggressiveness, -0.12, 0.12))
            candidate = budget_speed * weather_factor * soc_factor
            self.held_speed = self.clamp(candidate)

        if current_soc < self.config.min_soc + 0.02:
            self.held_speed = self.clamp(min(self.held_speed, self.config.min_speed_mps + 0.5))

        return self.held_speed


def build_strategy(name: str, race_config):
    normalized = name.strip().lower()
    if normalized == "pi":
        return PIControllerStrategy(race_config)
    if normalized == "stepped":
        return SteppedIfElseStrategy(race_config)
    if normalized in {"interval-hold", "interval", "linear"}:
        return IntervalHoldStrategy(race_config)
    raise ValueError(f"Unknown strategy: {name}")
