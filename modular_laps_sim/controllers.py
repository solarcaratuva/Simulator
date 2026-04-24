from abc import ABC, abstractmethod
import numpy as np

# Build target SOC trajectory from start_soc to target_soc.
# More drop is allocated in clearer periods, less in cloudier periods.
def compute_ideal_soc_curve(cloud_data: np.ndarray, race_config, alpha: float = 0.5) -> np.ndarray:
    n_steps = len(cloud_data)
    soc = np.zeros(n_steps)
    soc[0] = race_config.start_soc
    soc_drop_total = race_config.start_soc - race_config.target_soc

    cloud_factors = (1 - cloud_data) * alpha + (1 - alpha)
    total_weighted = np.sum(cloud_factors)
    if total_weighted <= 0:
        total_weighted = 1.0
    # Normalize per-step SOC drops so the total exactly matches soc_drop_total.
    normalized_drops = cloud_factors * (soc_drop_total / total_weighted)

    for i in range(1, n_steps):
        soc[i] = soc[i - 1] - normalized_drops[i]
        soc[i] = max(soc[i], race_config.target_soc)

    return soc

# Find a constant speed that consumes the total available race energy budget.
# Used as baseline speed for SOC feedback strategies to adjust around.
def solve_optimal_constant_speed(physics, race_config, ghi_data: np.ndarray, dt_hours: float) -> float:
    cap = physics.car.battery_capacity
    usable_energy = (race_config.start_soc - race_config.target_soc) * cap
    total_solar = np.sum(physics.solar_power(ghi_data)) * dt_hours
    total_available = usable_energy + total_solar
    n_steps = len(ghi_data)

    # Binary search for highest feasible constant speed in configured bounds.
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
    # Shared strategy interface and common helpers.
    def __init__(self, race_config):
        self.config = race_config
        self.optimal_speed = None

    # Precompute baseline references before simulation starts.
    def prepare(self, physics, ghi_data: np.ndarray, cloud_data: np.ndarray, dt_hours: float):
        self.optimal_speed = solve_optimal_constant_speed(physics, self.config, ghi_data, dt_hours)
        self.ideal_soc = compute_ideal_soc_curve(cloud_data, self.config)

    @abstractmethod
    def next_speed(self, step: int, current_speed: float, current_soc: float, soc_error: float, bdr: float, physics, ghi_data: np.ndarray, dt_hours: float) -> float:
        pass

    # Keep speed command within configured min/max limits.
    def clamp(self, speed: float) -> float:
        return float(np.clip(speed, self.config.min_speed_mps, self.config.max_speed_mps))

# Proportional-Integral strategy: continuous SOC feedback with memory of past error.
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

        # Integrate SOC error over time with anti-windup clipping. (hardbounds accumulated error integral to [-5, 5])
        self.integral_error += soc_error 
        self.integral_error = float(np.clip(self.integral_error, -5.0, 5.0))

        # Compute bounded PI correction around baseline speed.
        correction = kp * soc_error + ki * self.integral_error
        correction = float(np.clip(correction, -0.15, 0.15))
        speed = base_speed * (1.0 + correction)

        # Near minimum SOC, smooth toward minimum speed for battery protection.
        margin = 0.05
        soc_above_min = current_soc - self.config.min_soc
        if soc_above_min < margin:
            urgency = 1.0 - max(soc_above_min, 0.0) / margin
            speed = speed * (1 - urgency) + self.config.min_speed_mps * urgency

        return self.clamp(speed)

# Stepped strategy: applies discrete speed changes based on SOC error thresholds.
class SteppedIfElseStrategy(BaseStrategy):
    name = "stepped"

    def next_speed(self, step: int, current_speed: float, current_soc: float, soc_error: float, bdr: float, physics, ghi_data: np.ndarray, dt_hours: float) -> float:
        agg = self.config.aggressiveness
        step_small = 0.25 * agg
        step_large = 0.60 * agg
        speed = current_speed

        # Increase/decrease speed in fixed increments by error band.
        if soc_error > 0.03:
            speed += step_large
        elif soc_error > 0.01:
            speed += step_small
        elif soc_error < -0.03:
            speed -= step_large
        elif soc_error < -0.01:
            speed -= step_small

        # Apply extra slowdown if SOC approaches minimum threshold.
        if current_soc < self.config.min_soc + 0.03:
            speed -= step_large

        return self.clamp(speed)

# Interval-hold strategy: periodically recompute speed, then hold between updates
class IntervalHoldStrategy(BaseStrategy):
    name = "interval-hold"

    # update every in 5 minute intervals
    def __init__(self, race_config, interval_minutes: int = 5):
        super().__init__(race_config)
        self.interval_steps = max(1, int(interval_minutes / race_config.time_step_minutes))
        self.held_speed = None

    # Remaining budget = usable battery above target SOC + expected solar input.
    def _solve_remaining_speed(self, current_soc: float, physics, ghi_remaining: np.ndarray, dt_hours: float) -> float:
        cap = physics.car.battery_capacity
        usable_energy = max(0.0, (current_soc - self.config.target_soc) * cap)
        solar_energy = np.sum(physics.solar_power(ghi_remaining)) * dt_hours
        available = usable_energy + solar_energy

        n_steps = len(ghi_remaining)
        if n_steps == 0:
            return self.config.min_speed_mps

        # Binary search a constant speed that fits remaining energy budget.
        v_lo, v_hi = self.config.min_speed_mps, self.config.max_speed_mps
        for _ in range(60):
            v_mid = (v_lo + v_hi) / 2.0
            cost = n_steps * physics.power_drained(v_mid) * dt_hours
            if cost < available:
                v_lo = v_mid
            else:
                v_hi = v_mid
        return (v_lo + v_hi) / 2.0

    # Update speed only at interval boundaries; otherwise keep held speed.
    def next_speed(self, step: int, current_speed: float, current_soc: float, soc_error: float, bdr: float, physics, ghi_data: np.ndarray, dt_hours: float) -> float:
        should_update = self.held_speed is None or step % self.interval_steps == 0
        if should_update:
            # Replan feasible speed using current state and remaining irradiance.
            remaining_ghi = ghi_data[step:]
            budget_speed = self._solve_remaining_speed(current_soc, physics, remaining_ghi, dt_hours)

            weather_factor = 1.0
            if step < len(ghi_data):
                # Keep speed flatter in low irradiance periods, increase in strong sun.
                weather_factor = float(np.clip(0.9 + 0.2 * (ghi_data[step] / 900.0), 0.85, 1.1))

            # SOC-based nudge around the budget-constrained speed.
            soc_factor = 1.0 + float(np.clip(soc_error * self.config.aggressiveness, -0.12, 0.12))
            candidate = budget_speed * weather_factor * soc_factor
            self.held_speed = self.clamp(candidate)

        # Hard safety cap when SOC is very close to minimum.
        if current_soc < self.config.min_soc + 0.02:
            self.held_speed = self.clamp(min(self.held_speed, self.config.min_speed_mps + 0.5))

        return self.held_speed


class FixedSpeedStrategy(BaseStrategy):
    name = "fixed"

    def prepare(self, physics, ghi_data: np.ndarray, cloud_data: np.ndarray, dt_hours: float):
        # Keep ideal SOC for reporting parity, but force fixed commanded speed.
        super().prepare(physics, ghi_data, cloud_data, dt_hours)
        self.optimal_speed = self.clamp(self.config.fixed_speed_mps)

    def next_speed(self, step: int, current_speed: float, current_soc: float, soc_error: float, bdr: float, physics, ghi_data: np.ndarray, dt_hours: float) -> float:
        return self.clamp(self.config.fixed_speed_mps)


def build_strategy(name: str, race_config):
    # Factory: map strategy name from config/CLI to implementation.
    normalized = name.strip().lower()
    if normalized == "pi":
        return PIControllerStrategy(race_config)
    if normalized == "stepped":
        return SteppedIfElseStrategy(race_config)
    if normalized in {"interval-hold", "interval", "linear"}:
        return IntervalHoldStrategy(race_config)
    if normalized == "fixed":
        return FixedSpeedStrategy(race_config)
    raise ValueError(f"Unknown strategy: {name}")
