# Modular Laps Simulator

This is a modularized version of the laps simulator in a separate directory for comparisons.

## Supported location

- (1) Shenandoah Speedway
- (2) Virginia International Raceway (Patriot Course)
- (3) Brainerd International Raceway (Donnybrooke Course)

## Strategies

1. `pi` - Proportional-Integral controller
2. `stepped` - stepped if-else controller
3. `interval-hold` - piecewise-constant speed updates at fixed intervals

## Usage

Choose from the given locations (1, 2, 3). 

## Output plots

Plots are saved to `plots/laps_modular/` with file names that include the strategy.
