import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import energy_distance

from config import CarConfig, RaceConfig, get_available_tracks
from simulator import LapsRaceSimulator
from reporting import ResultsReporter
from plotting import SimulationPlotter, save_all_plots


DEFAULT_AGGRESSIVENESS = {
    "pi": 1.5,
    "stepped": 1.2,
    "interval-hold": 1.0,
    "fixed": 1.0,
}

STRATEGY_OPTIONS = ["pi", "stepped", "interval-hold", "fixed"]
SPEED_SWEEP_MPH = list(range(10, 47, 2))


def parse_args():
    parser = argparse.ArgumentParser(description="Modular laps race simulator")
    parser.add_argument(
        "--location",
        choices=["shenandoah"],
        default="shenandoah",
        help="Race location selection (currently only Shenandoah)",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_OPTIONS,
        default=None,
        help="Race control strategy (if omitted, you will be prompted)",
    )
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=None,
        help="Strategy aggressiveness override",
    )
    parser.add_argument(
        "--speed-sweep",
        action="store_true",
        help="Run fixed-speed sweeps and print summary results",
    )
    return parser.parse_args()


def choose_strategy() -> str:
    print("\nChoose a strategy:")
    for i in range(len(STRATEGY_OPTIONS)):
        print(f"({i+1}) {STRATEGY_OPTIONS[i]}")
    print("Enter a number:")
    while True:
        inp = input().strip().lower()
        if inp.isdigit():
            num = int(inp)
            if 1 <= num <= len(STRATEGY_OPTIONS):
                return STRATEGY_OPTIONS[num - 1]
        
        print("Invalid strategy. Enter a number or strategy name.")


def run_speed_sweep(track, car, args):
    print("\nRunning fixed-speed sweep")
    print(f"  Track: {track.name}")
    print(f"  Location: {track.location}")

    rows = []

    def fmt_elapsed(minute_of_day: float, start_time_hour: float) -> str:
        elapsed_min = minute_of_day - (start_time_hour * 60.0)
        elapsed_min = max(0.0, elapsed_min)
        hours = int(elapsed_min // 60)
        minutes = int(round(elapsed_min % 60))
        if minutes == 60:
            minutes = 0
            hours += 1
        return f"{hours}h {minutes:02d}m"

    for mph in SPEED_SWEEP_MPH:
        mps = mph / 2.237
        race = RaceConfig(
            start_soc=1.0,
            target_soc=0.10,
            aggressiveness=DEFAULT_AGGRESSIVENESS["fixed"],
            fixed_speed_mps=mps,
            time_step_minutes=1.0,
            strategy="fixed",
        )

        simulator = LapsRaceSimulator(
            track=track,
            car=car,
            race=race,
            use_api_weather=True,
        )
        results = simulator.run()
        rows.append(
            {
                "mph": mph,
                "final_soc": results.final_soc * 100.0,
                "laps": results.total_laps,
                "miles": results.total_distance_miles,
                "cutoff": (
                    "Full 8h"
                    if results.completed_full_window
                    else fmt_elapsed(results.reached_min_soc_time_minutes, race.start_time_hour)
                ),
            }
        )

    print("\nSpeed sweep summary")
    print("  Speed (mph) | Final SoC (%) | Laps | Distance (mi) | 10% reached at")
    print("  ----------- | ------------- | ---- | ------------- | --------------")
    for row in rows:
        print(
            f"  {row['mph']:>11.0f} | {row['final_soc']:>13.1f} |"
            f" {row['laps']:>4} | {row['miles']:>13.1f} | {row['cutoff']:>11}"
        )


def main():
    args = parse_args()
    tracks = get_available_tracks()

    repo_root = Path(__file__).resolve().parents[1]
    car_json = repo_root / "car_params.json"

    try:
        car = CarConfig.from_json(str(car_json))
    except FileNotFoundError:
        car = CarConfig()
        print("Using default car parameters")

    print("Choose a track to test on:")
    for i in range(len(tracks)):
        print(f"({i+1}) {tracks[i].name}")
    num = 0
    while True:
        inp = input()
        if inp.isdigit():
            num = int(inp)
            if num <= 0 or num > len(tracks):
                print("Invalid int")
            else:
                break
        elif inp == "":
            break

    track = tracks[num-1]

    if args.speed_sweep:
        run_speed_sweep(track, car, args)
        return

    selected_strategy = args.strategy if args.strategy else choose_strategy()

    aggressiveness = (
        args.aggressiveness
        if args.aggressiveness is not None
        else DEFAULT_AGGRESSIVENESS[selected_strategy]
    )

    print("Enter a safety scale (0.01-1, default 1):")
    print("This tells the simulator to use a percentage of the calculated energy budget")
    energy_safety_scale = 1.0
    while True:
        inp = input()
        if inp == "": break
        try:
            energy_safety_scale = float(inp)
            if 0.01 <= energy_safety_scale <= 1:
                break
            else:
                print("Must be between 0.01 and 1")
        except ValueError:
            print("Invalid float")


    race = RaceConfig(
        start_soc=1.0,
        target_soc=0.10,
        aggressiveness=aggressiveness,
        energy_safety_scale=energy_safety_scale,
        initial_speed_mps=20.0,
        time_step_minutes=1.0,
        strategy=selected_strategy,
    )

    print("\nStarting Modular Laps Race Simulation")
    print(f"  Track: {track.name}")
    print(f"  Location: {track.location}")
    print(f"  Strategy: {selected_strategy}")
    print(f"  Aggressiveness: {race.aggressiveness:.2f}")

    simulator = LapsRaceSimulator(
        track=track,
        car=car,
        race=race,
        use_api_weather=True,
    )
    results = simulator.run()

    reporter = ResultsReporter(results, track, race)
    reporter.print_summary()

    output_dir = os.path.join("plots", "laps_modular")
    prefix = f"shenandoah_{selected_strategy}"
    plotter = SimulationPlotter(results, track)
    saved = save_all_plots(plotter, output_dir, prefix)

    print("Saved plots:")
    print(f"  - {saved['dashboard']}")
    print(f"  - {saved['soc']}")
    print(f"  - {saved['speed']}")
    print(f"  - {saved['weather']}")


if __name__ == "__main__":
    main()
