import argparse
import os
from pathlib import Path

from config import CarConfig, RaceConfig, get_available_tracks
from simulator import LapsRaceSimulator
from reporting import ResultsReporter
from plotting import SimulationPlotter, save_all_plots


DEFAULT_AGGRESSIVENESS = {
    "pi": 1.5,
    "stepped": 1.2,
    "interval-hold": 1.0,
}


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
        choices=["pi", "stepped", "interval-hold"],
        default="pi",
        help="Race control strategy",
    )
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=None,
        help="Strategy aggressiveness override",
    )
    parser.add_argument(
        "--synthetic-weather",
        action="store_true",
        help="Force synthetic weather (skip API)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tracks = get_available_tracks()
    track = tracks[args.location]

    repo_root = Path(__file__).resolve().parents[1]
    car_json = repo_root / "car_params.json"

    try:
        car = CarConfig.from_json(str(car_json))
        print(f"Loaded car parameters from {car_json}")
    except FileNotFoundError:
        car = CarConfig()
        print("Using default car parameters")

    aggressiveness = (
        args.aggressiveness
        if args.aggressiveness is not None
        else DEFAULT_AGGRESSIVENESS[args.strategy]
    )

    race = RaceConfig(
        start_soc=1.0,
        target_soc=0.10,
        aggressiveness=aggressiveness,
        initial_speed_mps=20.0,
        time_step_minutes=1.0,
        strategy=args.strategy,
    )

    print("\nStarting Modular Laps Race Simulation")
    print(f"  Track: {track.name}")
    print(f"  Location: {track.location}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Aggressiveness: {race.aggressiveness:.2f}")

    simulator = LapsRaceSimulator(
        track=track,
        car=car,
        race=race,
        use_api_weather=not args.synthetic_weather,
    )
    results = simulator.run()

    reporter = ResultsReporter(results, track, race)
    reporter.print_summary()

    output_dir = os.path.join("plots", "laps_modular")
    prefix = f"shenandoah_{args.strategy}"
    plotter = SimulationPlotter(results, track)
    saved = save_all_plots(plotter, output_dir, prefix)

    print("Saved plots:")
    print(f"  - {saved['dashboard']}")
    print(f"  - {saved['soc']}")
    print(f"  - {saved['speed']}")
    print(f"  - {saved['weather']}")


if __name__ == "__main__":
    main()
