import numpy as np


class ResultsReporter:
    def __init__(self, results, track, race):
        self.results = results
        self.track = track
        self.race = race

    def print_summary(self):
        r = self.results
        print("\n" + "=" * 80)
        print(f"  LAPS RACE SIMULATION RESULTS - {self.track.name}")
        print("=" * 80)

        print(f"\nTrack: {self.track.name}")
        print(f"   Location: {self.track.location}")
        print(f"   Lap Distance: {self.track.lap_distance_km:.3f} km ({self.track.lap_distance_km/1.609:.2f} miles)")

        print("\nRace Results:")
        print(f"   Total Laps Completed: {r.total_laps}")
        print(f"   Total Distance: {r.total_distance_km:.2f} km ({r.total_distance_miles:.2f} miles)")

        print("\nBattery:")
        print(f"   Starting SoC: {self.race.start_soc*100:.0f}%")
        print(f"   Final SoC: {r.final_soc*100:.1f}%")
        print(f"   Target SoC: {self.race.target_soc*100:.0f}%")
        print(f"   SoC Error: {abs(r.final_soc - self.race.target_soc)*100:.1f}%")

        print("\nSpeed:")
        print(f"   Average: {r.avg_speed_mps:.2f} m/s ({r.avg_speed_mph:.1f} mph)")
        print(f"   Maximum: {np.max(r.speed):.2f} m/s ({np.max(r.speed)*2.237:.1f} mph)")
        print(f"   Minimum: {np.min(r.speed):.2f} m/s ({np.min(r.speed)*2.237:.1f} mph)")

        if r.lap_times:
            print("\nLap Times (minutes):")
            print(f"   Average: {np.mean(r.lap_times):.2f}")
            print(f"   Fastest: {np.min(r.lap_times):.2f}")
            print(f"   Slowest: {np.max(r.lap_times):.2f}")

        print("\n" + "=" * 80 + "\n")
