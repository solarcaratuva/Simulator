import numpy as np


class PhysicsEngine:
    def __init__(self, car):
        self.car = car

    def get_crr(self, v: float) -> float:
        v_kmh = v * 3.6
        return 0.005 + (1 / self.car.tire_pressure) * (0.01 + 0.0095 * (v_kmh / 100) ** 2)

    def force_drag(self, v: float) -> float:
        return 0.5 * self.car.C_dA * self.car.rho * v ** 2

    def force_rolling_resistance(self, v: float) -> float:
        return self.car.mass * self.car.g * self.get_crr(v)

    def power_drained(self, v: float) -> float:
        f_d = self.force_drag(v)
        f_r = self.force_rolling_resistance(v)
        return (f_d + f_r) * v * self.car.electrical_efficiency

    def solar_power(self, ghi: float) -> float:
        return self.car.solar_panel_area * ghi * self.car.solar_panel_efficiency

    def regen_energy(self, v_old: float, v_new: float) -> float:
        if v_new >= v_old:
            return 0.0
        delta_ke = 0.5 * self.car.mass * (v_old ** 2 - v_new ** 2)
        return (delta_ke / 3600.0) * self.car.regen_efficiency

    def battery_drain_rate(self, v: float, ghi: float) -> float:
        power_out = self.power_drained(v)
        power_in = self.solar_power(ghi)
        return (power_out - power_in) / self.car.battery_capacity
