import testing_formulas as formulas

def print_status(v, ghi):
    print("Energy from solar panels: ", formulas.calculate_solar_charge(ghi), "W")
    print("Drag force: ", formulas.force_drag(v), "N")
    print("Rolling resistance: ", formulas.force_rolling_resistance(v), "N")
    print("Total resistance: ", formulas.force_drag(v) + formulas.force_rolling_resistance(v), "N")
    print("Power being drained: ", formulas.power_drained(formulas.force_drag(v), formulas.force_rolling_resistance(v), 0, v), "W")


print_status(13.38, 1000)