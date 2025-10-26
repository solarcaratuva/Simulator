import math
import numpy as np

# define constants:
g = 9.80665 # m/s^2, acceleration due to gravity
regen_efficiency = float('inf') # efficiency coefficient of the regenerative braking system
electrical_efficiency = 0.99 # efficiency coefficient of the electrical system
solar_panel_efficiency = 0.23 # efficiency coefficient of the solar panel
wheels = 4 # number of wheels on the vehicle
mass = 337 # kg, mass of the vehicle
battery_capacity = 5000 # Wh, capacity of the vehicle's battery
battery_voltage = float('inf') # V, voltage of the vehicle's battery
solar_panel_area = 4 # m^2, area of the solar panels on the vehicle
C_dA = 0.73809 # drag coefficient times frontal area of the vehicle
rho = 1.192 # kg/m^3, air density in kentucky
v_starting = 90 #km/h, starting velocity of the vehicle
tire_pressure = 5 #tire pressure in bar

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

def power_drained(F_d: float, F_r: float, F_gr: float, v: float) -> float:
    """
    Calculates the power drained by the solar car

    Args:
        F_d (float): Drag force (Newtons)
        F_r (float): Rolling resistance force (Newtons)
        F_gr (float): Gradient Resistance Force (Newtons)
        v (float): Velocity of the car (m/s)

    Returns:
        float: Power drained by the solar car, in Watts
    """
    F_net = F_d + F_r + F_gr
    return F_net * v * electrical_efficiency

def force_drag(v: float) -> float:
    """
    Calculates the drag force on the solar car

    Args:
        v (float): Velocity of the car (m/s)

    Returns:
        float: Drag force on the solar car, in Newtons
    """
    return 0.5 * C_dA * rho * v**2

def force_rolling_resistance(v: float) -> float:
    """
    Calculates the rolling resistance force on the solar car

    Args:
        v (float): Velocity of the car (m/s)
    
    Returns:
        float: Rolling resistance force on the solar car, in Newtons
    """
    C_rr = get_crr(v)
    return mass * g * C_rr

def get_crr(v: float) -> float:
    """
    Get the coefficient of rolling resistance based on velocity.

    Args:
        v (float): Velocity in m/s.
    
    Returns:
        float: Coefficient of rolling resistance.
    """
    # return 0.008539
    v = v * 3.6  # convert m/s to km/h
    return 0.005 + (1/tire_pressure) * (0.01 + 0.0095 * (v / 100)**2)