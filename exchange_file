import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the data file (assuming it is space-separated)
file_path = "NissanLeafTrip49.csv"  # Modify the filename as needed
df = pd.read_csv(file_path, delim_whitespace=True)

# Ensure timestamps are properly parsed
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Energy efficiency (km/kWh)
def energy_consumption(df):
    BatteryCapacity = 40  # kWh capacity by manufacturer
    SOC_in = float(df["SOC"].iloc[0])
    SOC_fi = df["SOC"].iloc[-1]
    Energyconsumed = BatteryCapacity * (SOC_in - SOC_fi)
    Odo_ini = df["Odometer"].iloc[0]
    Odo_fi = df["Odometer"].iloc[-1]
    Distancetravelled = Odo_fi - Odo_ini
    Energy_Efficiency = Distancetravelled / Energyconsumed if Energyconsumed != 0 else np.nan
    return Energyconsumed, Energy_Efficiency

# Calculate acceleration
def calculate_acceleration(speed, time):
    acceleration = np.diff(speed) / np.diff(time)
    return np.insert(acceleration, 0, np.nan)  # First value will be NaN

# Calculate jerk
def calculate_jerk(acceleration, time):
    jerk = np.diff(acceleration) / np.diff(time[1:])  # Skip first NaN
    return np.insert(jerk, 0, np.nan)  # First value will be NaN

# Calculate harsh braking events
def calculate_harsh_braking_events(braking_data, threshold_multiplier=2):
    if not isinstance(braking_data, np.ndarray):
        braking_data = np.array(braking_data)
    
    mean_braking = np.mean(braking_data)
    std_braking = np.std(braking_data)
    threshold = mean_braking + threshold_multiplier * std_braking
    harsh_braking_indices = np.where(braking_data > threshold)[0]
    
    return harsh_braking_indices

# Detect coasting
def detect_coasting(df, accel_threshold=0.1, brake_threshold=0.1):
    df["Speed_m_s"] = df["Speed"] * (1000 / 3600)
    df["Time_diff_s"] = df["Timestamp"].diff().dt.total_seconds()
    df["Acceleration_m_s2"] = df["Speed_m_s"].diff() / df["Time_diff_s"]
    
    coasting_conditions = (
        (df["Acceleration_m_s2"].abs() < accel_threshold) &
        (df["Brake"] < brake_threshold) &
        (df["Speed_m_s"] > 0)
    )
    
    return df[coasting_conditions]

# Process the data
df["Acceleration"] = calculate_acceleration(df["Speed"], df["Timestamp"].astype(int))
df["Jerk"] = calculate_jerk(df["Acceleration"], df["Timestamp"].astype(int))
energy_consumed, energy_efficiency = energy_consumption(df)
harsh_braking_events = calculate_harsh_braking_events(df["Brake"])
coasting_events = detect_coasting(df)

# Save results to a new file
output_file = "trip_results.csv"
df.to_csv(output_file, index=False)

print(f"Processed trip data saved to: {output_file}")
