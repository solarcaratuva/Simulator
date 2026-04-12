# Modular Laps Simulator

This is a modularized version of the laps simulator in a separate directory for comparisons.

## Supported locations

- (1) Shenandoah Speedway
- (2) Virginia International Raceway (Patriot Course)
- (3) Brainerd International Raceway (Donnybrooke Course)
  
When running, choose from the given locations (1, 2, 3). 


## Strategies

All strategies use the same SOC reference idea:
- A target SOC trajectory (`ideal_soc`) is precomputed for the full run.
- At each simulation step, SOC tracking error is computed as `soc_error = current_soc - ideal_soc[step]`.
- Positive `soc_error` means the car is above target SOC (more energy margin).
- Negative `soc_error` means the car is below target SOC (less energy margin).

1. `pi` - Proportional-Integral controller

	How it works:
	- Starts from a baseline speed computed from full-race energy budget.
	- Applies a proportional correction from current SOC error.
	- Applies an integral correction from accumulated SOC error over time.
	- Anti-windup: clamps the integral state (hardbounds accumulated error) and clamps correction magnitude. 
	- Adds low-SOC protection by blending speed toward minimum speed near `min_soc`.

	Update behavior:
	- Recomputes speed every simulation timestep.

	Best for: continuously changing speed based on changing cloud conditions, keeping SOC consistent.

2. `stepped` - Threshold-based if-else controller

	How it works:
	- Uses discrete SOC error bands (small/large positive or negative).
	- Increases or decreases speed by fixed step sizes based on the band.
	- Applies an extra slowdown when SOC approaches `min_soc`.

	Update behavior:
	- Recomputes speed every simulation timestep, but only changes speed in fixed increments.

	Best for: simple control behavior with minimal tuning complexity (threshold based).

3. `interval-hold` - Periodic recompute with held speed

	How it works:
	- Recomputes a feasible constant speed for the remaining horizon using:
	  - current SOC relative to `target_soc`, and
	  - remaining solar forecast (`ghi`).
	- Uses weather and SOC factors to nudge that budget-constrained speed.
	- Holds the resulting speed constant until the next update point.
	- Applies a hard safety cap near `min_soc`.

	Update behavior:
	- Recomputes only when `step % interval_steps == 0` (and at first step),
	  where `interval_steps = max(1, int(interval_minutes / time_step_minutes))`.
	- Keeps speed unchanged between those update points.

	Best for: More stable, less jittery speed changes, still adaptive to SOC and forecast changes. (most realistic and ideal for car control)

Quick comparison:
- `pi`: continuous error-tracking control.
- `stepped`: rule-based threshold control.
- `interval-hold`: receding-budget planning with constant speed.

## Usage

**need to add input selection for strategies, but currently you can choose between pi, stepped, and interval-hold strategy with command line arguments (and an aggressiveness parameter)

* python3 modular_laps_sim/main.py --strategy interval-hold 
* python3 modular_laps_sim/main.py --strategy pi --aggressiveness 0.9
* python3 modular_laps_sim/main.py --strategy stepped --aggressiveness 1.0

## Output plots

Plots are saved to `plots/laps_modular/` with file names that include the strategy.
