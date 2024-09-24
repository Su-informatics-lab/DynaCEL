import os
import os.path as osp

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def read_events(filepath):
    events = pd.read_csv(filepath)
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events["deathtime"] = pd.to_datetime(events["deathtime"], errors="coerce")
    events["intime"] = pd.to_datetime(events["intime"], errors="coerce")
    events["outtime"] = pd.to_datetime(events["outtime"], errors="coerce")
    events["time_diff"] = pd.to_timedelta(events["time_diff"], errors="coerce")
    return events


def filter_events(events, prediction_window, moment_of_prediction, outcome_window):
    end_time = pd.to_timedelta(moment_of_prediction, unit="h")
    start_time = end_time - pd.to_timedelta(prediction_window, unit="h")

    events = events[
        (events["time_diff"] >= start_time) & (events["time_diff"] < end_time)
    ]
    return events


cohort_data_dir = "/N/project/waveform_mortality/JL/Xiang's Model/MOP18Cohort/MIMIC_MOP18_roughcohort.csv"
event_data_dir = "data/data_for_training/mimic-iv"
dst_dir = "data/data_for_training/mimic-iv/datasets/PW_12__MOP_18__OW_24"
os.makedirs(event_data_dir, exist_ok=True)
os.makedirs(dst_dir, exist_ok=True)

prediction_window = 12
moment_of_prediction = 18
outcome_window = 24

cohort_data = pd.read_csv(cohort_data_dir)
print(len(cohort_data))

hr_events = read_events(os.path.join(event_data_dir, "merged_hr_events.csv"))
hr_events = hr_events.iloc[:, 1:]
sbp_events = read_events(os.path.join(event_data_dir, "merged_sbp_events.csv"))
sbp_events = sbp_events.iloc[:, 1:]
print(len(hr_events), len(sbp_events))

print("\nicustay of interest")
stay_id_of_interest = cohort_data["stay_id"].values
print(len(hr_events), len(sbp_events), len(stay_id_of_interest))

hr_events = hr_events.loc[hr_events["stay_id"].isin(stay_id_of_interest)]
sbp_events = sbp_events.loc[sbp_events["stay_id"].isin(stay_id_of_interest)]
print(
    len(hr_events),
    len(hr_events["stay_id"].unique()),
    len(sbp_events),
    len(sbp_events["stay_id"].unique()),
)

print("\nfiltering events")
hr_events = filter_events(
    hr_events, prediction_window, moment_of_prediction, outcome_window
)
sbp_events = filter_events(
    sbp_events, prediction_window, moment_of_prediction, outcome_window
)
stay_id_interasected = set(hr_events["stay_id"].values).intersection(
    set(sbp_events["stay_id"].values)
)
print(len(hr_events), len(sbp_events), len(stay_id_interasected))

hr_events = hr_events.loc[hr_events["stay_id"].isin(stay_id_interasected)]
hr_events = hr_events.drop(columns=["death_in_24_hours"])
sbp_events = sbp_events.loc[sbp_events["stay_id"].isin(stay_id_interasected)]
print(
    len(hr_events),
    len(hr_events["stay_id"].unique()),
    len(sbp_events),
    len(sbp_events["stay_id"].unique()),
)

hr_events = hr_events.merge(cohort_data[["stay_id", "dead24"]], on="stay_id")
hr_events = hr_events.rename(columns={"dead24": "mortality", "valuenum": "m1_valuenum"})
sbp_events = sbp_events.rename(columns={"valuenum ": "m2_valuenum"})

print(
    hr_events.groupby("stay_id")
    .agg({"mortality": "first"})["mortality"]
    .value_counts()
    .to_dict()
)

print("\nmedian")
hr_events = hr_events.groupby("stay_id").agg(
    {
        "m1_valuenum": "median",
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
        "mv_oxygen_therapy": "first",
        "mv_none": "first",
        "mv_unknown": "first",
        "race_w": "first",
        "race_b": "first",
        "race_l": "first",
        "race_o": "first",
        "mortality": "first",
    }
)

sbp_events = sbp_events[["stay_id", "m2_valuenum"]]
sbp_events = sbp_events.groupby("stay_id").agg({"m2_valuenum": "median"})

hr_events = hr_events.reset_index()
sbp_events = sbp_events.reset_index()

hr_events.to_csv(os.path.join(dst_dir, "merged_hr_events.csv"), index=False)
sbp_events.to_csv(os.path.join(dst_dir, "merged_sbp_events.csv"), index=False)

merged_events = pd.merge(hr_events, sbp_events, on="stay_id")
merged_events.to_csv(os.path.join(dst_dir, "merged_events.csv"), index=False)

merged_events = merged_events.reset_index()

print(len(merged_events))
print(
    merged_events.groupby("stay_id")
    .agg({"mortality": "first"})["mortality"]
    .value_counts()
    .to_dict()
)

merged_events["subset"] = "train"

print("\nSaving data")
merged_events.to_csv(os.path.join(dst_dir, "merged_events.csv"), index=False)
