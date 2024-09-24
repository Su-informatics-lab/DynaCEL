import os
import pickle
import random
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

random.seed(137)
np.random.seed(137)

pd.options.mode.chained_assignment = None  # default="warn"


feature_names = [
    "m1_valuenum",
    "m2_valuenum",
    "gender",
    "age",
    "mi",
    "chf",
    "pvd",
    "cevd",
    "dementia",
    "cpd",
    "rheumd",
    "pud",
    "mld",
    "diab",
    "diabwc",
    "hp",
    "rend",
    "canc",
    "msld",
    "metacanc",
    "aids",
    "CCI",
    "height",
    "weight",
    "bmi",
    "height_avail",
    "weight_avail",
    "bmi_avail",
    "respiration",
    "coagulation",
    "liver",
    "cardiovascular",
    "cns",
    "renal",
    "sofa_total",
    "mv_invasive",
    "mv_non_vasive",
    "mv_oxygen_therapy",
    "mv_unknown",
    "mv_none",
    "race_w",
    "race_b",
    "race_l",
    "race_o",
]


def read_events(filepath):
    events = pd.read_csv(filepath)
    return events


def merge_measurements(measurement_1, measurement_2, start_time, end_time):
    m1_clean = measurement_1[
        (measurement_1["nursingchartoffset"] >= start_time)
        & (measurement_1["nursingchartoffset"] < end_time)
    ]
    m2_clean = measurement_2[
        (measurement_2["nursingchartoffset"] >= start_time)
        & (measurement_2["nursingchartoffset"] < end_time)
    ]
    m1_clean.rename(columns={"nursingchartvalue": "m1_valuenum"}, inplace=True)
    m2_clean = m2_clean[["patientunitstayid", "nursingchartvalue"]]
    m2_clean.rename(columns={"nursingchartvalue": "m2_valuenum"}, inplace=True)
    return pd.merge(m1_clean, m2_clean, on="patientunitstayid")


def get_median_of_measurements(measurements):
    medians_data = measurements.groupby("patientunitstayid").agg(
        {
            "m1_valuenum": "median",
            "m2_valuenum": "median",
            "age": "median",
            "gender": "first",
            "mi": "first",
            "chf": "first",
            "pvd": "first",
            "cevd": "first",
            "dementia": "first",
            "cpd": "first",
            "rheumd": "first",
            "pud": "first",
            "mld": "first",
            "diab": "first",
            "diabwc": "first",
            "hp": "first",
            "rend": "first",
            "canc": "first",
            "msld": "first",
            "metacanc": "first",
            "aids": "first",
            "CCI": "first",
            "height": "median",
            "weight": "median",
            "bmi": "median",
            "height_avail": "first",
            "weight_avail": "first",
            "bmi_avail": "first",
            "respiration": "first",
            "coagulation": "first",
            "liver": "first",
            "cardiovascular": "first",
            "cns": "first",
            "renal": "first",
            "sofa_total": "first",
            "mv_invasive": "first",
            "mv_non_vasive": "first",
            "mv_unknown": "first",
            "mv_oxygen_therapy": "first",
            "mv_none": "first",
            "race_w": "first",
            "race_b": "first",
            "race_l": "first",
            "race_o": "first",
            "death_in_24_hours": "max",  # , "hospital_expire_flag": "max"
        }
    )
    return medians_data.reset_index()


prediction_window = 12
time = 30
outcome_window = 24
data_dir = "data/data_for_training/eicu"
dst_dir = f"data/data_for_visualization/eicu"
os.makedirs(dst_dir, exist_ok=True)

hr_events = read_events(os.path.join(data_dir, "merged_hr_events.csv"))
print(hr_events.columns)
hr_events = hr_events[
    ["patientunitstayid", "hospitaldischargeoffset", "hospitaldischargestatus"]
]
print(hr_events["hospitaldischargeoffset"].isna().sum())
assert False
hr_events = hr_events.iloc[:, 1:]
sbp_events = read_events(os.path.join(data_dir, "merged_sbp_events.csv"))
sbp_events = sbp_events.iloc[:, 1:]


print(prediction_window, time, outcome_window)
end_time = time * 60
start_time = end_time - prediction_window * 60

merged_measurements = merge_measurements(hr_events, sbp_events, start_time, end_time)
print(merged_measurements.head())
measurements_median = get_median_of_measurements(merged_measurements)
print(measurements_median.head())

measurements_median = measurements_median.rename(
    columns={"patientunitstayid": "stay_id"}
)
# measurements_median.loc[measurements_median["age"].isna(), "age"] = 64
print(measurements_median["age"].isna().sum())
measurements_median.to_csv(
    os.path.join(
        dst_dir,
        f"model_data_PW_{prediction_window}__MOP_{time}__OW_{outcome_window}.csv",
    ),
    index=False,
)
