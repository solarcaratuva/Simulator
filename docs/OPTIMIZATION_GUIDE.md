# Race Simulation Battery Optimization Guide

## Overview
This guide explains how the race simulation has been optimized to reach as close to 10% State of Charge (SoC) as possible by the end of the race, maximizing average speed and distance traveled.

## Optimization Strategy: Predictive Control

The simulation uses a **predictive control** algorithm that:
1. Calculates an ideal SoC trajectory (linear decline from 100% to 10%)
2. Compares current SoC to the ideal trajectory
3. Adjusts speed dynamically based on the error

### How It Works

#### Key Parameters
- `target_soc`: Target battery level at end of race (default: 0.10 = 10%)
- `aggressiveness`: Controls how aggressively to adjust speed (default: 1.6)
- `start_soc`: Battery level at start of race (default: 1.0 = 100%)

#### Control Logic
```
SoC Error = Current SoC - Ideal SoC (at this time point)

If SoC Error > 0.20:   # Way above target
    Speed up by 25% × aggressiveness
    
If SoC Error > 0.10:   # Above target
    Speed up by 15% × aggressiveness
    
If SoC Error > 0.03:   # Slightly above
    Speed up by 5% × aggressiveness
    
If -0.03 < SoC Error < 0.03:  # On target
    Fine-tune based on battery drain rate
    
If SoC Error < -0.10:  # Below target
    Slow down by 10% × aggressiveness
    
If SoC Error < -0.20:  # Way below target
    Slow down by 20% × aggressiveness
```

## Usage

### Run with default settings (targets 10% SoC):
```python
soc, bdr, speed, ghi_list, dist_list, total_dist_list, time_race = simulate_race()
```

### Customize target and aggressiveness:
```python
# Target 15% SoC with moderate aggressiveness
soc, ... = simulate_race(target_soc=0.15, aggressiveness=1.0)

# Target 10% SoC with high aggressiveness
soc, ... = simulate_race(target_soc=0.10, aggressiveness=1.6)

# Target 8% SoC with very high aggressiveness
soc, ... = simulate_race(target_soc=0.08, aggressiveness=2.0)
```

## Results

With `aggressiveness=1.6` and `target_soc=0.10`:
- **Final SoC**: 10.39%
- **Error from Target**: 0.39%
- **Total Distance**: ~402.8 km (250.3 miles)
- **Average Speed**: ~13.7 m/s (49.3 km/h, 30.7 mph)

## Tuning Guide

### If you end ABOVE target (e.g., 15% instead of 10%):
- **Increase** `aggressiveness` (try 1.8, 2.0, etc.)
- This makes the car speed up more aggressively when battery is high

### If you end BELOW target (e.g., 5% instead of 10%):
- **Decrease** `aggressiveness` (try 1.4, 1.2, etc.)
- This makes speed adjustments more conservative

### To change the target:
- Modify `target_soc` parameter (e.g., 0.08 for 8%, 0.15 for 15%)

## Alternative Approaches (Not Implemented)

1. **PID Controller**: More sophisticated feedback control
2. **Optimization (scipy)**: Find optimal speed profile mathematically
3. **Look-ahead**: Consider future solar predictions
4. **Dynamic Programming**: Solve for global optimum

The current predictive control approach is simple, effective, and easy to tune!

## Notes

- Random noise in GHI (solar irradiance) means results vary between runs
- The algorithm adapts in real-time, no pre-computation needed
- Speed is limited to 5-35 m/s safety range
- Minimum SoC floor is set to 5% to prevent complete battery depletion
