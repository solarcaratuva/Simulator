import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class SimulationPlotter:
    def __init__(self, results, track):
        self.results = results
        self.track = track

    def _time_to_hours(self):
        return self.results.time_minutes / 60.0

    def plot_soc(self, save_path: Optional[str] = None, show: bool = False):
        fig, ax = plt.subplots(figsize=(10, 5))
        hours = self._time_to_hours()

        ax.plot(hours, self.results.soc * 100, label="Actual SoC", linewidth=2)
        ax.plot(hours, self.results.ideal_soc * 100, "--", label="Ideal SoC", alpha=0.7, linewidth=1.5)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("State of Charge (%)")
        ax.set_title(f"Battery SoC - {self.track.name}")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close()

    def plot_speed(self, save_path: Optional[str] = None, show: bool = False):
        fig, ax = plt.subplots(figsize=(10, 5))
        hours = self._time_to_hours()
        speed_mph = self.results.speed * 2.237

        ax.plot(hours, speed_mph, color="green", linewidth=1.5)
        ax.axhline(y=np.mean(speed_mph), color="red", linestyle="--", label=f"Average: {np.mean(speed_mph):.1f} mph", alpha=0.7)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Speed (mph)")
        ax.set_title(f"Speed Profile - {self.track.name}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close()

    def plot_weather(self, save_path: Optional[str] = None, show: bool = False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        hours = self._time_to_hours()

        ax1.plot(hours, self.results.ghi, color="orange", linewidth=1.5)
        ax1.set_ylabel("GHI (W/m^2)")
        ax1.set_title(f"Solar Irradiance - {self.track.name}")
        ax1.grid(True, alpha=0.3)

        ax2.plot(hours, self.results.cloud_cover * 100, color="gray", linewidth=1.5)
        ax2.fill_between(hours, 0, self.results.cloud_cover * 100, alpha=0.3, color="gray")
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Cloud Cover (%)")
        ax2.set_title("Cloud Cover")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close()

    def plot_dashboard(self, save_path: Optional[str] = None, show: bool = False):
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        hours = self._time_to_hours()

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(hours, self.results.soc * 100, label="Actual", linewidth=2)
        ax1.plot(hours, self.results.ideal_soc * 100, "--", label="Target", alpha=0.7)
        ax1.set_ylabel("SoC (%)")
        ax1.set_title("Battery State of Charge")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        ax2 = fig.add_subplot(gs[0, 1])
        speed_mph = self.results.speed * 2.237
        ax2.plot(hours, speed_mph, color="green", linewidth=1.5)
        ax2.axhline(y=np.mean(speed_mph), color="red", linestyle="--", alpha=0.7)
        ax2.set_ylabel("Speed (mph)")
        ax2.set_title(f"Speed (Avg: {np.mean(speed_mph):.1f} mph)")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(hours, self.results.ghi, color="orange", linewidth=1.5)
        ax3.set_ylabel("GHI (W/m^2)")
        ax3.set_title("Solar Irradiance")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.fill_between(hours, 0, self.results.cloud_cover * 100, alpha=0.5, color="gray")
        ax4.set_ylabel("Cloud Cover (%)")
        ax4.set_title("Cloud Cover")
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(hours, self.results.bdr, color="red", linewidth=1.5)
        ax5.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax5.set_xlabel("Time (hours)")
        ax5.set_ylabel("BDR (1/hr)")
        ax5.set_title("Battery Drain Rate")
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis("off")
        summary_text = f"""
RACE SUMMARY - {self.track.name}
========================================

Total Laps: {self.results.total_laps}
Total Distance: {self.results.total_distance_km:.2f} km
               ({self.results.total_distance_miles:.2f} miles)
Final SoC: {self.results.final_soc*100:.1f}%

Speed Statistics:
  Average: {self.results.avg_speed_mph:.1f} mph
  Max: {np.max(self.results.speed)*2.237:.1f} mph
  Min: {np.min(self.results.speed)*2.237:.1f} mph
"""
        ax6.text(
            0.1,
            0.95,
            summary_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle("Solar Car Laps Race Simulation", fontsize=14, fontweight="bold")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def save_all_plots(plotter: SimulationPlotter, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    dashboard = os.path.join(output_dir, f"{prefix}_race_dashboard.png")
    soc = os.path.join(output_dir, f"{prefix}_soc_plot.png")
    speed = os.path.join(output_dir, f"{prefix}_speed_plot.png")
    weather = os.path.join(output_dir, f"{prefix}_weather_plot.png")

    plotter.plot_dashboard(save_path=dashboard)
    plotter.plot_soc(save_path=soc)
    plotter.plot_speed(save_path=speed)
    plotter.plot_weather(save_path=weather)

    return {
        "dashboard": dashboard,
        "soc": soc,
        "speed": speed,
        "weather": weather,
    }
