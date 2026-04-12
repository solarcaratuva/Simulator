# Modular Laps Simulator

This is a modularized version of the laps simulator in a separate directory for comparisons.

## Supported locations

- (1) Shenandoah Speedway
- (2) Virginia International Raceway (Patriot Course)
- (3) Brainerd International Raceway (Donnybrooke Course)
  
When running, choose from the given locations (1, 2, 3). 


## Strategies

1. `pi` - Proportional-Integral controller
2. `stepped` - stepped if-else controller
3. `interval-hold` - piecewise-constant speed updates at fixed intervals

## Usage

**need to add input selection for strategies, but currently you can choose between pi, stepped, and interval-hold strategy with command line arguments (and an aggressiveness parameter)

* python3 modular_laps_sim/main.py --strategy interval-hold 
* python3 modular_laps_sim/main.py --strategy pi --aggressiveness 0.9
* python3 modular_laps_sim/main.py --strategy stepped --aggressiveness 1.0

## Output plots

Plots are saved to `plots/laps_modular/` with file names that include the strategy.
