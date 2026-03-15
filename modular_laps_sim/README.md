# Modular Laps Simulator

This is a modularized version of the laps simulator in a separate directory for comparisons.

## Supported location

- `shenandoah` (Shenandoah Speedway, VA)

## Strategies

1. `pi` - Proportional-Integral controller
2. `stepped` - stepped if-else controller
3. `interval-hold` - piecewise-constant speed updates at fixed intervals

## Usage

```bash
python modular_laps_sim/main.py --location shenandoah --strategy pi
python modular_laps_sim/main.py --location shenandoah --strategy stepped --aggressiveness 1.3
python modular_laps_sim/main.py --location shenandoah --strategy interval-hold --aggressiveness 0.9
```

Use `--synthetic-weather` to run without API weather.

## Output plots

Plots are saved to `plots/laps_modular/` with file names that include the strategy.
