import numpy as np
import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache
import json
import math

with open ("car_params.json", "r") as f:
    params = json.load(f)

for key, value in params.items():
    if value == "inf":
        params[key] = math.inf

g = params["g"]
regen_efficiency = params["regen_efficiency"]
electrical_efficiency = params["electrical_efficiency"]
solar_panel_efficiency = params["solar_panel_efficiency"]
wheels = params["wheels"]
mass = params["mass"]
battery_capacity = params["battery_capacity"]
battery_voltage = params["battery_voltage"]
solar_panel_area = params["solar_panel_area"]
C_dA = params["C_dA"]
rho = params["rho"]
v_starting = params["v_starting"]
tire_pressure = params["tire_pressure"]


# dummy solar GHI Curve
# assuming the GHI curve is a parabola w/ 0 W/m^2 at 6:30 AM, peaks at 1 PM with 1200 W/m^2, and 0 W/m^2 at 6:30 PM
def generate_ghi_curve():
    start_day = 6.5 * 60  # 6:30 AM in minutes
    end_day = 18.5 * 60   # 6:30 PM in minutes
    t_peak = 13 * 60      # 1:00 PM
    peak_val = 1200       # W/m^2

    # generative Cloud Sky Irradience Curve
    time_minutes = np.arange(start_day, end_day + 1)
    a = peak_val / ((t_peak - start_day) ** 2)
    ghi_base = np.maximum(0, -a * (time_minutes - t_peak) ** 2 + peak_val)
    
    # add random, smoothed noise to simulate clouds
    random_factor = np.clip(np.random.normal(1.0, 0.08, len(ghi_base)), 0.6, 1.3)
    smooth_noise = np.convolve(random_factor, np.ones(15)/15, mode='same')
    ghi_noisy = ghi_base * smooth_noise

    return time_minutes, ghi_noisy

def get_ghi_curve():
    openmeteo = openmeteo_requests.Client(
    session = requests_cache.CachedSession(
        cache_name='openmeteo_cache',
        backend='memory'
    )
    )
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 37.000495,
        "longitude": -86.368110,
        "hourly": ["shortwave_radiation", "shortwave_radiation_instant"]
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    hourly_shortwave_radiation = hourly.Variables(0).ValuesAsNumpy()
    hourly_shortwave_radiation_instant = hourly.Variables(1).ValuesAsNumpy()
    hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
    )}
    hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
    hourly_data["shortwave_radiation_instant"] = hourly_shortwave_radiation_instant
    hourly_dataframe = pd.DataFrame(data = hourly_data)
    central_timezone = "America/Chicago"
    hourly_dataframe_central = hourly_dataframe.set_index("date").tz_convert(central_timezone)
    daytime_data = hourly_dataframe_central.between_time("10:00", "18:00")
    # daytime_data = daytime_data.head(2)
    daytime_data = daytime_data[1:10]  # first day only
    daytime_data = daytime_data.reset_index()
    daytime_data = daytime_data.resample("1T", on="date").mean().interpolate(method="spline", order=3)
    time_minutes = np.arange(0, len(daytime_data) * 1, 1) + 10 * 60  # minutes since midnight
    ghi_data = daytime_data["shortwave_radiation"].to_numpy()
    return time_minutes, ghi_data
    

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
    return (F_d + F_r) * v / electrical_efficiency

def calculate_solar_charge(ghi):
    return solar_panel_area * ghi * solar_panel_efficiency

# race simulator
def simulate_race(target_soc=0.10, aggressiveness=1.0, start_soc=1.0):
    """
    Simulate a race with optimized speed control to hit target SoC at the end.
    
    Parameters:
    - target_soc: Target state of charge at end of race (default 0.10 = 10%)
    - aggressiveness: Multiplier for speed adjustments (default 1.0)
                     Higher values = more aggressive speed changes
    """
    time_minutes, ghi_data = get_ghi_curve()

    # assuming that race is 10 AM to 6 PM
    start_idx = int((10 - 6.5) * 60)   # 10:00 AM
    end_idx = int((18 - 6.5) * 60)     # 6:00 PM
    # ghi_race = ghi_data[start_idx:end_idx + 1]
    # time_race = time_minutes[start_idx:end_idx + 1]
    ghi_race = ghi_data
    time_race = time_minutes

    soc = start_soc # starting SoC
    v = 13.4  # m/s (30 mph), starting speed
    soc_list, bdr_list, speed_list, ghi_list, dist_list, total_dist_list = [], [], [], [], [], []

    total_distance = 0
    total_steps = len(range(0, len(ghi_race), 10))

    for step, i in enumerate(range(0, len(ghi_race), 10)):  # 10-min intervals
        ghi = ghi_race[i].clip(0)  # ensure non-negative GHI
        ein = calculate_solar_charge(ghi)
        eout = power_drained(v)

        bdr = (eout - ein) / battery_capacity  # per hour
        soc += (ein - eout) * (10 / 60) / battery_capacity  # 10-min step

        '''Optimized race logic: adjust speed to target SoC at end
           Uses predictive control based on remaining time and current SoC'''
        
        # Calculate how far we are from target trajectory
        remaining_steps = total_steps - step
        if remaining_steps > 0:
            # linear decline from current to target
            time_fraction = step / total_steps
            ideal_soc = start_soc - (start_soc - target_soc) * time_fraction
            soc_error = soc - ideal_soc
            
            # Adjust speed based on SoC error and battery drain rate
            # Scale adjustments by aggressiveness parameter
            if soc_error > 0.20:  # Way above target - speed up significantly
                v *= (1 + 0.25 * aggressiveness)
            elif soc_error > 0.10:  # Above target - speed up moderately
                v *= (1 + 0.15 * aggressiveness)
            elif soc_error > 0.03:  # Slightly above - speed up a bit
                v *= (1 + 0.05 * aggressiveness)
            elif soc_error > -0.03:  # Near target - fine tune
                if bdr > 0.05:
                    v *= (1 - 0.02 * aggressiveness)
                else:
                    v *= (1 + 0.02 * aggressiveness)
            elif soc_error > -0.10:  # Below target - slow down
                v *= (1 - 0.10 * aggressiveness)
            else:  # Way below target - slow down significantly
                v *= (1 - 0.20 * aggressiveness)
        else:
            # basically done, maintain current speed
            continue

        v = np.clip(v, 0, 35)
        soc = max(soc, 0.05)  # Allow lower SoC since we're targeting 10%

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

    return soc_list, bdr_list, speed_list, ghi_list, dist_list, total_dist_list, time_race

# Run simulation with target 10% SoC and aggressiveness factor
# Increase aggressiveness to make more aggressive speed adjustments
TARGET_SOC = 0.10  # target SoC at end
AGGRESSIVENESS = 1.6  # more aggressive speed adjustments
soc, bdr, speed, ghi_list, dist_list, total_dist_list, time_race = simulate_race(target_soc=TARGET_SOC, aggressiveness=AGGRESSIVENESS)

print("\n SIM RESULTS: ")
for i in range(len(soc)):
    print(f"t={i*10} min | SoC={soc[i]*100:.2f}% | BDR={bdr[i]:.3f} | "
          f"Speed={speed[i]:.2f} m/s | GHI={ghi_list[i]:.1f} W/m² | "
          f"Step Dist={dist_list[i]:.1f} m | Total Dist={total_dist_list[i]:.1f} m")

# Summary statistics
print("\n" + "="*80)
print("RACE SUMMARY:")
print(f"Final SoC: {soc[-1]*100:.2f}% (Target: {TARGET_SOC*100:.2f}%)")
print(f"SoC Error: {abs(soc[-1]*100 - TARGET_SOC*100):.2f}%")
print(f"Total Distance: {total_dist_list[-1]/1000:.2f} km ({total_dist_list[-1]/1609.34:.2f} miles)")
print(f"Average Speed: {np.mean(speed):.2f} m/s ({np.mean(speed)*3.6:.2f} km/h, {np.mean(speed)*2.237:.2f} mph)")
print(f"Max Speed: {np.max(speed):.2f} m/s ({np.max(speed)*3.6:.2f} km/h)")
print(f"Min Speed: {np.min(speed):.2f} m/s ({np.min(speed)*3.6:.2f} km/h)")
print("="*80 + "\n")

# plot GHI curve 
plt.figure(figsize=(10, 5))
# matching time points 
t_plot = np.arange(0, len(ghi_list) * 10, 10) / 60 + 10  # hours since 10 AM

plt.plot(t_plot, ghi_list, color="orange")
plt.xlabel("Time (hours)")
plt.ylabel("GHI (W/m²)")
plt.title("Randomized Solar Irradiance (GHI) During Race Day")
plt.grid(True)

plt.savefig("ghi_plot.png")  # saves plot as PNG
plt.close()  

# Plot SoC over the day
plt.figure(figsize=(10, 5))

# time axis: one point every 10 minutes from 10:00 to ~18:00
t_soc_hours = np.arange(0, len(soc) * 10, 10) / 60 + 10  # 10 = 10 AM

plt.plot(t_soc_hours, np.array(soc) * 100)  # convert to %
plt.xlabel("Time (hours)")
plt.ylabel("State of Charge (%)")
plt.title("Battery SoC Over Race Day")
plt.ylim(0, 100)
plt.grid(True)

plt.savefig("soc_plot.png")
plt.close()
