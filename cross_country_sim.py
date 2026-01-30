import numpy as np
import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache

# constants
g = 9.80665 # m/s^2, acceleration due to gravity
electrical_efficiency = 0.99 # efficiency coefficient of the electrical system
solar_panel_efficiency = 0.23 # efficiency coefficient of the solar panel
mass = 337 # kg, mass of the vehicle
battery_capacity = 5000 # Wh, capacity of the vehicle's battery
solar_panel_area = 4 # m^2, area of the solar panels on the vehicle
C_dA = 0.4162 * 1.8 # drag coefficient times frontal area of the vehicle
rho = 1.192 # kg/m^3, air density in kentucky
tire_pressure = 5  # bar
mu = electrical_efficiency

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

#NEW GHI CURVE WITH API DATA AND CLOUD COVER
def get_ghi_curve():
    openmeteo = openmeteo_requests.Client(
        session = requests_cache.CachedSession(
            cache_name='openmeteo_cache',
            backend='memory'
        )
    )
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 46.417585,
        "longitude": -94.284839,
        "hourly": ["shortwave_radiation_instant", "cloud_cover"],
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_shortwave_radiation_instant = hourly.Variables(0).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
    )}
    hourly_data["shortwave_radiation_instant"] = hourly_shortwave_radiation_instant
    hourly_data["cloud_cover"] = hourly_cloud_cover

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    central_timezone = "America/Chicago"
    hourly_dataframe_central = hourly_dataframe.set_index("date").tz_convert(central_timezone)
    daytime_data = hourly_dataframe_central.between_time("10:00", "18:00")
    # daytime_data = daytime_data.head(2)
    daytime_data = daytime_data[1:10]  # first day only
    daytime_data = daytime_data.reset_index()

    # resample to 1 minute intervals
    daytime_data = daytime_data.resample("min", on="date").mean()
    # interpolate shortwave radiation columns with cubic spline
    daytime_data["shortwave_radiation_instant"] = daytime_data["shortwave_radiation_instant"].interpolate(method="spline", order=3)
    # interpolate cloud cover linearly
    daytime_data["cloud_cover"] = daytime_data["cloud_cover"].interpolate(method="linear")

    time_minutes = np.arange(0, len(daytime_data) * 1, 1) + 10 * 60  # minutes since midnight
    ghi_data = daytime_data["shortwave_radiation_instant"].to_numpy()
    cloud_data = daytime_data["cloud_cover"].to_numpy() / 100 

    return time_minutes, ghi_data, cloud_data
    

def ideal_soc_cloud(start_soc, target_soc, cloud_data, alpha=0.5):

    """
    Calculate an ideal SoC curve that reacts to cloudiness:
    - Sunny (cloud low) → can use more battery (drop SoC faster)
    - Cloudy (cloud high) → conserve battery (drop SoC slower)
    alpha: controls strength of cloud impact (0=ignore clouds, 1=fully reactive)
    Ensures the final SoC is close to target_soc by normalizing cloud-weighted drops
    """
    n_steps = len(cloud_data)
    soc = np.zeros(n_steps)
    soc[0] = start_soc
    soc_drop_total = start_soc - target_soc
    
    # Calculate cloud-weighted factors for each step
    # More sunny (low cloud) → higher factor → faster drop
    cloud_factors = (1 - cloud_data) * alpha + (1 - alpha)
    
    # Normalize cloud factors so the total drop equals soc_drop_total
    total_weighted_steps = np.sum(cloud_factors)
    normalized_factors = cloud_factors * (soc_drop_total / total_weighted_steps)

    for i in range(1, n_steps):
        # Apply the normalized drop at each step
        soc[i] = soc[i-1] - normalized_factors[i]
        soc[i] = max(soc[i], target_soc)  # don't drop below target

    time_x = np.arange(n_steps)

    plt.figure(figsize=(8, 4))
    plt.plot(time_x, soc, label="Ideal SoC with Cloud Cover", linewidth=2)
    plt.plot(time_x, 1 - cloud_data * (1 - target_soc), "--", alpha=0.3, label="Cloud Cover (scaled)")
    plt.title("Ideal SoC with Cloud Cover")
    plt.xlabel("Time Step")
    plt.ylabel("State of Charge")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/cross_country/soc_curve_with_cloud.png", dpi=300)
    plt.close()
    
    return soc

#race simulator with cloud adjustment
def simulate_race_cloud(start_soc=1.0, target_soc=0.10, aggressiveness=1.0):
    time_minutes, ghi_data, cloud_data = get_ghi_curve()

    # Plot cloud cover curve
    plt.figure(figsize=(8, 4))
    plt.plot(time_minutes / 60, cloud_data, label="Cloud Cover", color='orange', linewidth=2)
    plt.title("Cloud Cover Over Time")
    plt.xlabel("Time (hours)")
    plt.savefig("plots/cross_country/cloud_cover_curve.png", dpi=300)

    soc = start_soc
    v = 13.4
    soc_list, bdr_list, speed_list, ghi_list, dist_list, total_dist_list = [], [], [], [], [], []
    total_distance = 0
    interval_sec = 10 * 60  # 10-min step

    ideal_soc_curve = ideal_soc_cloud(start_soc, target_soc, cloud_data)

    for step, i in enumerate(range(0, len(ghi_data), 10)):
        ghi = ghi_data[i].clip(0)
        cloud = cloud_data[i]
        
        ein = calculate_solar_charge(ghi)
        eout = power_drained(v)
        
        #added dt for time step (10 min time step)
        dt_hours = 10 / 60
        bdr = (eout - ein) * dt_hours / battery_capacity
        soc -= bdr
        
        # bdr = (eout - ein) / battery_capacity
        # soc += (ein - eout) * (10 / 60) / battery_capacity
        

        soc_error = soc - ideal_soc_curve[i]

        # speed adjustment based on SoC error
        if soc_error > 0.20:
            v *= (1 + 0.25 * aggressiveness)
        elif soc_error > 0.10:
            v *= (1 + 0.15 * aggressiveness)
        elif soc_error > 0.03:
            v *= (1 + 0.05 * aggressiveness)
        elif soc_error > -0.03:
            if bdr > 0.05:
                v *= (1 - 0.02 * aggressiveness)
            else:
                v *= (1 + 0.02 * aggressiveness)
        elif soc_error > -0.10:
            v *= (1 - 0.10 * aggressiveness)
        else:
            v *= (1 - 0.20 * aggressiveness)

        # cloud-reactive speed adjustment
        # if cloud > 0.5:
        #     v *= 1 + (1 - cloud) * 0.1 * aggressiveness - cloud * 0.05 * aggressiveness

        v = np.clip(v, 0, 35)
        soc = max(soc, 0.05)

        dist_step = v * interval_sec
        total_distance += dist_step

        soc_list.append(soc)
        bdr_list.append(bdr)
        speed_list.append(v)
        ghi_list.append(ghi)
        dist_list.append(dist_step)
        total_dist_list.append(total_distance)

    return soc_list, bdr_list, speed_list, ghi_list, dist_list, total_dist_list, time_minutes, ideal_soc_curve

# Run simulation with target 10% SoC and aggressiveness factor
# Increase aggressiveness to make more aggressive speed adjustments
TARGET_SOC = 0.10  # target SoC at end
AGGRESSIVENESS = 1.6  # more aggressive speed adjustments

#simulating with cloud cover
soc, bdr, speed, ghi_list, dist_list, total_dist_list, time_race, ideal_soc_cloud = simulate_race_cloud(target_soc=TARGET_SOC, aggressiveness=AGGRESSIVENESS)

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

'''
# plot soc curve with cloud
plt.figure(figsize=(10, 5))
t_plot = np.arange(len(cloud)) * 10 / 60 + 10  # hours since 10 AM

plt.plot(t_plot, cloud, label="Cloudiness (0 = sunny, 1 = cloudy)")
plt.xlabel("Time (hours)")
plt.ylabel("Cloudiness")
plt.title("Cloudiness Profile Over Race")
plt.grid(True)

plt.savefig("cloudiness_curve.png")
plt.close()

print("Cloudiness plot saved as cloudiness_curve.png")
'''