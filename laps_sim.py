import testing_formulas

class LapSim:
    def __init__(self, track_length, car_speed, start_soc, battery_capacity):
        self.track_length = track_length
        self.car_speed = car_speed
        self.start_soc = start_soc
        self.battery_capacity = battery_capacity

    def run_lap(self):
        time_minutes = self.track_length / (self.car_speed * 60)  # time in minutes
        e_curr = self.start_soc * self.battery_capacity  # current energy in Wh
        e_in = testing_formulas.calculate_solar_charge(self.car_speed) * time_minutes  # energy in Wh
        e_out = testing_formulas.power_drained(self.car_speed) * time_minutes / 60  # energy in Wh

        e_total = testing_formulas.total_energy_consumption(e_curr, e_in, e_out)
        final_soc = e_total / self.battery_capacity

        return final_soc, time_minutes

def simulate_lap(start_soc, target_soc, ghi_data, cloud_data, battery_capacity, aggressiveness):
    soc = start_soc
    soc_list = []
    bdr_list = []
    speed_list = []
    ghi_list = []
    dist_list = []
    total_dist_list = []
    time_minutes = []

    v = 25  # initial speed in m/s