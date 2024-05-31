import pandas as pd
import numpy as np

import random
import os

random.seed(137)
np.random.seed(137)

pd.options.mode.chained_assignment = None  # default="warn"


feature_names = [
    "m1_valuenum", "m2_valuenum", "gender", "age", "mi", "chf", "pvd", "cevd", 
    "dementia", "cpd", "rheumd", "pud", "mld", "diab", "diabwc", "hp", "rend", 
    "canc", "msld", "metacanc", "aids", "CCI", "height", "weight", "bmi", 
    "height_avail", "weight_avail", "bmi_avail", "respiration", "coagulation",
    "liver", "cardiovascular", "cns", "renal", "sofa_total", "mv_invasive",
    "mv_non_vasive", "mv_unknown", "mv_oxygen_therapy", "mv_unknown", "mv_none", 
    "race_w", "race_b", "race_l", "race_o"
]


def read_events(filepath):
    events = pd.read_csv(filepath)
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events["deathtime"] = pd.to_datetime(events["deathtime"], errors="coerce")
    events["intime"] = pd.to_datetime(events["intime"], errors="coerce")
    events["outtime"] = pd.to_datetime(events["outtime"], errors="coerce")
    events["time_diff"] = pd.to_timedelta(events["time_diff"], errors="coerce")
    return events


def merge_measurements(measurement_1, measurement_2, start_time, end_time):
    m1_clean = measurement_1[(measurement_1["time_diff"] >= start_time) & (measurement_1["time_diff"] < end_time)]
    m2_clean = measurement_2[(measurement_2["time_diff"] >= start_time) & (measurement_2["time_diff"] < end_time)]
    m1_clean.rename(columns={"valuenum": "m1_valuenum"}, inplace=True)
    m2_clean = m2_clean[["stay_id", "valuenum"]]
    m2_clean.rename(columns={"valuenum": "m2_valuenum"}, inplace=True)
    return pd.merge(m1_clean, m2_clean, on="stay_id")


def get_median_of_measurements(measurements):
    medians_data = measurements.groupby("stay_id").agg(
        {
            "m1_valuenum": "median", "m2_valuenum": "median", "age": "median", "gender": "first",
            "mi": "first", "chf": "first", "pvd": "first", "cevd": "first", "dementia": "first", 
            "cpd": "first", "rheumd": "first", "pud": "first", "mld": "first", "diab": "first", 
            "diabwc": "first", "hp": "first", "rend": "first", "canc": "first", "msld": "first", 
            "metacanc": "first", "aids": "first", "CCI": "first", "height": "median", "weight": "median", "bmi": "median", 
            "height_avail": "first", "weight_avail": "first", "bmi_avail": "first", 
            "respiration": "first", "coagulation": "first", "liver": "first", "cardiovascular": "first", 
            "cns": "first", "renal": "first", "sofa_total": "first",
            "mv_invasive": "first", "mv_non_vasive": "first", "mv_unknown": "first", "mv_oxygen_therapy": "first", "mv_none": "first",
            "race_w": "first", "race_b": "first", "race_l": "first", "race_o": "first",
            "death_in_24_hours": "max"#, "hospital_expire_flag": "max"
        }
    )
    return medians_data.reset_index()

prediction_window = 12
time = 30
outcome_window = 24
data_dir = "data/data_for_training/mimic-iv"
dst_dir = "data/data_for_visualization/mimic-iv"
os.makedirs(dst_dir, exist_ok=True)

hr_events = read_events(os.path.join(data_dir, "merged_hr_events.csv"))

hr_events = hr_events.iloc[:, 1:]
sbp_events = read_events(os.path.join(data_dir, "merged_sbp_events.csv"))
sbp_events = sbp_events.iloc[:, 1:]


print(prediction_window, time, outcome_window)
end_time = pd.to_timedelta(time, unit="h")
start_time = end_time - pd.to_timedelta(prediction_window, unit="h")

merged_measurements = merge_measurements(hr_events, sbp_events, start_time, end_time)
print(merged_measurements.head())
measurements_median = get_median_of_measurements(merged_measurements)
print(measurements_median.head())

measurements_median.to_csv(os.path.join(dst_dir, f"model_data_PW_{prediction_window}__MOP_{time}__OW_{outcome_window}.csv"), index=False)