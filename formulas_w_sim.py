import numpy as np

# constants
g = 9.80665 # m/s^2, acceleration due to gravity
electrical_efficiency = 0.99 # efficiency coefficient of the electrical system
solar_panel_efficiency = 0.23 # efficiency coefficient of the solar panel
mass = 337 # kg, mass of the vehicle
battery_capacity = 5000 # Wh, capacity of the vehicle's battery
solar_panel_area = 4 # m^2, area of the solar panels on the vehicle
C_dA = 0.73809 # drag coefficient times frontal area of the vehicle
rho = 1.192 # kg/m^3, air density in kentucky
tire_pressure = 5  # bar
mu = electrical_efficiency

#hello

# dummy solar GHI Curve
# assuming the GHI curve is a parabola w/ 0 W/m^2 at 6:30 AM, peaks at 1 PM with 1200 W/m^2, and 0 W/m^2 at 6:30 PM
def generate_ghi_curve():
    start_day = 6.5 * 60  # 6:30 AM in minutes
    end_day = 18.5 * 60   # 6:30 PM in minutes
    t_peak = 13 * 60      # 1:00 PM
    peak_val = 1200       # W/m^2

    time_minutes = np.arange(start_day, end_day + 1)
    a = peak_val / ((t_peak - start_day) ** 2)
    ghi = np.maximum(0, -a * (time_minutes - t_peak) ** 2 + peak_val)
    return time_minutes, ghi

# crr, Fd, Fr, power drained (E out), solar charge (E in)
def get_crr(v):
    v_kmh = v * 3.6  # convert to km/h
    return 0.005 + (1 / tire_pressure) * (0.01 + 0.0095 * (v_kmh / 100) ** 2)

def force_drag(v):
    return 0.5 * C_dA * rho * v ** 2

def force_rolling_resistance(v):
    return mass * g * get_crr(v)

def power_drained(v):
    F_d = force_drag(v)
    F_r = force_rolling_resistance(v)
    return (F_d + F_r) * v * mu

def calculate_solar_charge(ghi):
    return solar_panel_area * ghi * solar_panel_efficiency

# race simulator
def simulate_race():
    time_minutes, ghi_data = generate_ghi_curve()

    # assuming that race is 10 AM to 4 PM
    start_idx = int((10 - 6.5) * 60)   # 10:00 AM
    end_idx = int((16 - 6.5) * 60)     # 4:00 PM
    ghi_race = ghi_data[start_idx:end_idx + 1]
    time_race = time_minutes[start_idx:end_idx + 1]

    soc = 1.0 # 100% capacity charge at start
    v = 13.4  # m/s (30 mph), starting speed
    soc_list, bdr_list, speed_list, ghi_list, dist_list, total_dist_list = [], [], [], [], [], []

    total_distance = 0

    for i in range(0, len(ghi_race), 10):  # 10-min intervals
        ghi = ghi_race[i]
        ein = calculate_solar_charge(ghi)
        eout = power_drained(v)

        bdr = (eout - ein) / battery_capacity  # per hour
        soc += (ein - eout) * (10 / 60) / battery_capacity  # 10-min step

        '''this is the main logic that can be changed to
          figure out how to optimize speed 
          based on SoC and BDR to gain most dist'''
        # race logic: adjust speed
        if soc > 0.25:
            if 0.03 < bdr < 0.08:
                #v *= 1.03
                v *= 1.15
            elif bdr < 0.03:
                #v *= 1.05
                v *= 1.25
            elif bdr > 0.08:
                v *= 0.95
                

        else:
            v *= 0.85  # conserve energy if SoC is low

        v = np.clip(v, 5, 35)
        soc = max(soc, 0.2)

        # calculate dist travelled per interval and increment to total dist
        interval_sec = 10 * 60  # 10 minutes in seconds
        dist_step = v * interval_sec
        total_distance += dist_step

        # store soc, bdr, speed, ghi, dist, and total dist
        soc_list.append(soc)
        bdr_list.append(bdr)
        speed_list.append(v)
        ghi_list.append(ghi)
        dist_list.append(dist_step)
        total_dist_list.append(total_distance)

    return soc_list, bdr_list, speed_list, ghi_list, dist_list, total_dist_list

# Run simulation
soc, bdr, speed, ghi_list, dist_list, total_dist_list = simulate_race()

for i in range(len(soc)):
    print(f"t={i*10} min | SoC={soc[i]*100:.2f}% | BDR={bdr[i]:.3f} | Speed={speed[i]:.2f} m/s | GHI={ghi_list[i]:.1f} W/m² | "
          f"Step Dist={dist_list[i]:.1f} m | Total Dist={total_dist_list[i]:.1f} m")
