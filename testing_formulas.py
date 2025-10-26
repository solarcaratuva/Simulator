import math
import numpy as np

# define constants:
g = 9.81 # m/s^2, acceleration due to gravity
regen_efficiency = float('inf') # efficiency coefficient of the regenerative braking system
electrical_efficiency = float('inf') # efficiency coefficient of the electrical system
solar_panel_efficiency = float('inf') # efficiency coefficient of the solar panel
wheels = 4 # number of wheels on the vehicle
mass = float('inf') # kg, mass of the vehicle
battery_capacity = float('inf') # Wh, capacity of the vehicle's battery
battery_voltage = float('inf') # V, voltage of the vehicle's battery
solar_panel_area = float('inf') # m^2, area of the solar panels on the vehicle
C_dA = float('inf') # drag coefficient times frontal area of the vehicle

def calculate_regenerative_energy(v_initial: float) -> float:
    """
    Calculate the energy recovered through regenerative braking.

    Args:
        v_initial (float): Initial velocity in m/s.
    
    Returns:
        float: Energy recovered in Wh.
    """
    return (0.5 * mass * v_initial ** 2) * regen_efficiency / 3600


def calculate_solar_charge(global_horizontal_coefficient: float) -> float:
    """
    Calculate the charge from the solar panels.

    Args:
        global_horizontal_coefficient (float): Global horizontal irradiance coefficient.
    
    Returns:
        float: Charge from solar panels in W.
    """
    return solar_panel_area * global_horizontal_coefficient * solar_panel_efficiency

def power_consumption(e_curr, e_in, e_out): 
    """ Calculates the powerConsumption of the solar car by the minute minute

    Args:
        e_curr(float) - current charge of battery, e_in(float) - energy from solar panels and regenerative energy, e_out(float) - energy that goes to the motor

    Returns:
        Returns the total energy consumption by the minute
    """
    e_total = e_curr + e_in/60 + e_out/60
    return e_total

def power_produced(F_d: float, F_r: float, F_gr: float, v: float) -> float:
    """
    Calculates the power produced by the solar car

    Args:
        F_d (float): Drag force
        F_r (float): Rolling resistance force
        F_gr (float): Gradient Resistance Force
        v (float): Velocity of the car

    Returns:
        float: Power produced by the solar car, in Watts
    """
    F_net = F_d + F_r + F_gr
    return F_net * v * electrical_efficiency