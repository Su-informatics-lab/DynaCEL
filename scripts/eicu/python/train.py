from statistics import median
import pandas as pd
import numpy as np

import pickle

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

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
    "mv_non_vasive", "mv_oxygen_therapy", "mv_none", "mv_unknown", "race_w", 
    "race_b", "race_l", "race_o"
]


def read_events(filepath):
    events = pd.read_csv(filepath)
    return events


def merge_measurements(measurement_1, measurement_2, start_time, end_time):
    m1_clean = measurement_1[(measurement_1["nursingchartoffset"] >= start_time) & (measurement_1["nursingchartoffset"] < end_time)]
    m2_clean = measurement_2[(measurement_2["nursingchartoffset"] >= start_time) & (measurement_2["nursingchartoffset"] < end_time)]
    m1_clean.rename(columns={"nursingchartvalue": "m1_valuenum"}, inplace=True)
    m2_clean = m2_clean[["patientunitstayid", "nursingchartvalue"]]
    m2_clean.rename(columns={"nursingchartvalue": "m2_valuenum"}, inplace=True)
    return pd.merge(m1_clean, m2_clean, on="patientunitstayid")


def get_median_of_measurements(measurements):
    medians_data = measurements.groupby("patientunitstayid").agg(
        {
            "m1_valuenum": "median", "m2_valuenum": "median", "age": "median", "gender": "first",
            "mi": "first", "chf": "first", "pvd": "first", "cevd": "first", "dementia": "first", 
            "cpd": "first", "rheumd": "first", "pud": "first", "mld": "first", "diab": "first", 
            "diabwc": "first", "hp": "first", "rend": "first", "canc": "first", "msld": "first", 
            "metacanc": "first", "aids": "first", "CCI": "first", "height": "median", "weight": "median", "bmi": "median", 
            "height_avail": "first", "weight_avail": "first", "bmi_avail": "first", 
            "respiration": "first", "coagulation": "first", "liver": "first", "cardiovascular": "first", 
            "cns": "first", "renal": "first", "sofa_total": "first",
            "mv_invasive": "first", "mv_non_vasive": "first", "mv_oxygen_therapy": "first", "mv_none": "first", "mv_unknown": "first",
            "race_w": "first", "race_b": "first", "race_l": "first", "race_o": "first",
            "death_in_24_hours": "max",
            # "death_in_6_hours": "max", "death_in_12_hours": "max", "death_in_24_hours": "max",
            # "death_in_36_hours": "max", "death_in_48_hours": "max", "hospitalstatus": "max",
        }
    )
    return medians_data.reset_index()


def train_model_and_cal_auc(measurements_median, start_time, end_time, col):
    
    clfs = []
    aucs = []
    col_pred = f"pred_{col}"
    measurements_to_predict = measurements_median[ feature_names+[col] ].copy(True)

    # RF training
    X = measurements_to_predict[feature_names]
    y = measurements_to_predict[col]
    stay_ids = measurements_median["patientunitstayid"]

    kf = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

        sm = SMOTE(random_state=42)
        X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)    

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_smote, y_train_smote)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        clfs.append(clf)
        aucs.append(auc)
    
    return clfs, aucs




def cal_mortality(
    measurement_1,
    measurement_2,
    dst_dir,
    prediction_moments=[3, 6, 12, 18, 24, 30, 36],
    prediction_windows=[3, 6, 12],
    outcome_windows=[6, 12, 24, 48],
    M1_name="HR",
    M2_name="SBP",
):

    schema={"PW": int, "MOP": int, "OW": int, "Fold": int, "AUC": float}
    auc_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    for prediction_window in prediction_windows:
        for outcome_window in outcome_windows:
            for time in prediction_moments:
                if time - prediction_window < 0: continue
                # define measurement time frame
                end_time = time * 60
                start_time = end_time - prediction_window * 60
                
                # remove measurements outside time frame
                merged_measurements = merge_measurements(measurement_1, measurement_2, start_time, end_time)
                measurements_median = get_median_of_measurements(merged_measurements)
                models, aucs = train_model_and_cal_auc(
                    measurements_median, start_time, end_time, col=f"death_in_{outcome_window}_hours"
                )

                for i in range(len(models)):
                    model, auc = models[i], aucs[i]
                    print(prediction_window, time, outcome_window, i, auc)
                    
                    model_path = os.path.join(dst_dir, f"model__PW_{prediction_window}__MOP_{time}__OW_{outcome_window}__Fold_{i}.pkl")
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)

                        auc_datapoint = pd.DataFrame([[prediction_window, time, outcome_window, i, auc]], columns=schema.keys())
                        auc_df = pd.concat([auc_df, auc_datapoint], ignore_index=True)
                        auc_df.to_csv(os.path.join(dst_dir, "AUCs.csv"), index=False)


data_dir = "/N/project/waveform_mortality/personalized_hemodynamics/data/data_for_training/eicu"
dst_dir = "/N/project/waveform_mortality/personalized_hemodynamics/results/models/eicu"
os.makedirs(dst_dir, exist_ok=True)

hr_events = read_events(os.path.join(data_dir, "merged_hr_events.csv"))
sbp_events = read_events(os.path.join(data_dir, "merged_sbp_events.csv"))


prediction_windows = [12] # [3, 6, 12]
prediction_moments = [18] #[3, 6, 12, 18, 24, 30, 36]
outcome_windows = [24] # [6, 12, 24, 48]
cal_mortality(
    hr_events,
    sbp_events,
    dst_dir=dst_dir,
    prediction_moments=prediction_moments,
    prediction_windows=prediction_windows,
    outcome_windows=outcome_windows,
    M1_name="HR",
    M2_name="SBP",
)

