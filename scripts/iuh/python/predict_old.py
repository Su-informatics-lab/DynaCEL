import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from imblearn.over_sampling import SMOTE

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
    events["EventDT"] = pd.to_datetime(events["EventDT"], errors="coerce")
    events["ArriveDTS"] = pd.to_datetime(events["ArriveDTS"], errors="coerce")
    events["DischargeDTS"] = pd.to_datetime(events["DischargeDTS"], errors="coerce")
    events["Deceased"] = pd.to_datetime(events["Deceased"], errors="coerce")
    events["time_diff"] = pd.to_timedelta(events["time_diff"], errors="coerce")
    return events

def merge_measurements(measurement_1, measurement_2, start_time, end_time):
    m1_clean = measurement_1[(measurement_1["time_diff"] >= start_time) & (measurement_1["time_diff"] < end_time)]
    m2_clean = measurement_2[(measurement_2["time_diff"] >= start_time) & (measurement_2["time_diff"] < end_time)]
    m1_clean.rename(columns={"Result": "m1_valuenum"}, inplace=True)
    m2_clean = m2_clean[["MRN", "FIN", "Result"]]
    m2_clean.rename(columns={"Result": "m2_valuenum"}, inplace=True)
    return pd.merge(m1_clean, m2_clean, on=["MRN", "FIN"])


def get_median_of_measurements(measurements):
    medians_data = measurements.groupby(["MRN", "FIN"]).agg(
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
            # "death_in_36_hours": "max", "death_in_48_hours": "max", "death_status": "max",
        }
    )
    return medians_data.reset_index()


def cal_auc(measurements_median, start_time, end_time, col):
    print(measurements_median.head())
    clfs = []
    aucs = []
    col_pred = f"pred_{col}"
    measurements_to_predict = measurements_median[ feature_names+[col] ].copy(True)

    # RF training
    X = measurements_to_predict[feature_names]
    y = measurements_to_predict[col]
    patient_ids = measurements_median["MRN"]
    stay_ids = measurements_median["FIN"]

    kf = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

        sm = SMOTE(random_state=42)
        X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)    

        # clf = RandomForestClassifier(n_estimators=100, random_state=42)
        # clf = LogisticRegression()
        clf = SVC(probability=True)
        clf.fit(X_train_smote, y_train_smote)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        clfs.append(clf)
        aucs.append(auc)
        break
    
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)
    auc = metrics.roc_auc_score(y, y_pred_proba[:, 1])
    
    return patient_ids.to_list(), stay_ids.to_list(), auc, y.to_numpy(), y_pred, y_pred_proba

    
    # return clfs, aucs


def cal_mortality(
    measurement_1,
    measurement_2,
    dst_dir,
    prediction_moments=[3, 6, 12, 18, 24, 30, 36],
    prediction_windows=[3, 6, 12],
    outcome_windows=[6, 12, 24, 48]
):

    schema={"MRN": int, "FIN": int, "PW": int, "MOP": int, "OW": int, "Fold": int, "pred_0_prob": float, "pred_1_prob": float, "pred_label": "int", "gt_label": int}
    pred_prob_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    for prediction_window in prediction_windows:
        for outcome_window in outcome_windows:
            for time in prediction_moments:
                if time - prediction_window < 0: continue
                # define measurement time frame
                end_time = pd.to_timedelta(time, unit="h")
                start_time = end_time - pd.to_timedelta(prediction_window, unit="h")
                
                # remove measurements outside time frame
                merged_measurements = merge_measurements(
                    measurement_1, measurement_2, start_time, end_time
                )
                measurements_median = get_median_of_measurements(merged_measurements)

                # for k in range(5):
                #     model_path = os.path.join(model_dir, f"model__PW_{prediction_window}__MOP_{time}__OW_{outcome_window}__Fold_{k}.pkl")
                #     with open(model_path, "rb") as f:
                #         model = pickle.load(f)

                #     stay_ids, auc, y_gt, y_pred, y_pred_prob = cal_auc(
                #         model, measurements_median, start_time, end_time, col=f"death_in_{outcome_window}_hours"
                #     )
                #     print(prediction_window, time, outcome_window, k, auc)
                #     pred_probs_list = []
                #     for i in np.arange(y_pred.shape[0]):
                #         stay_id = stay_ids[i]
                #         pred_0_prob = y_pred_prob[i, 0]
                #         pred_1_prob = y_pred_prob[i, 1]
                #         gt_label = y_gt[i]
                #         label = y_pred[i]
                #         pred_probs_list.append([stay_id, prediction_window, time, outcome_window, k, pred_0_prob, pred_1_prob, label, gt_label])
                #     pred_prob_datapoint = pd.DataFrame(pred_probs_list, columns=schema.keys())
                #     pred_prob_df = pd.concat([pred_prob_df, pred_prob_datapoint], ignore_index=True)
                patient_ids, stay_ids, auc, y_gt, y_pred, y_pred_prob = cal_auc(
                        measurements_median, start_time, end_time, col=f"death_in_{outcome_window}_hours"
                )
                pred_probs_list = []
                for i in np.arange(y_pred.shape[0]):
                    patient_id = patient_ids[i]
                    stay_id = stay_ids[i]
                    pred_0_prob = y_pred_prob[i, 0]
                    pred_1_prob = y_pred_prob[i, 1]
                    gt_label = y_gt[i]
                    label = y_pred[i]
                    pred_probs_list.append([patient_id, stay_id, prediction_window, time, outcome_window, 0, pred_0_prob, pred_1_prob, label, gt_label])
                pred_prob_datapoint = pd.DataFrame(pred_probs_list, columns=schema.keys())
                pred_prob_df = pd.concat([pred_prob_df, pred_prob_datapoint], ignore_index=True)
                print(prediction_window, time, outcome_window, 0, auc)
                
        pred_probs_gt_labels = pred_prob_df.groupby(["MRN", "FIN", "PW", "MOP", "OW"]).agg({"pred_1_prob": "first", "gt_label": "first"})
        pred_probs_gt_labels = pred_probs_gt_labels.reset_index()
        pred_probs_gt_labels = pred_probs_gt_labels.drop(columns=["PW", "MOP", "OW"])
        pred_probs_gt_labels = pred_probs_gt_labels.rename(columns={"pred_1_prob": "pred_prob"})
        pred_probs_gt_labels.to_csv(os.path.join(dst_dir, "pred_probs_gt_labels.csv"), index=False)



data_dir = "/N/project/waveform_mortality/xiang/personalized_hemodynamics/data/data_for_training/iuh/"
dst_dir = "/N/project/waveform_mortality/xiang/personalized_hemodynamics/results/predictions/iuh"
os.makedirs(dst_dir, exist_ok=True)


hr_events = read_events(os.path.join(data_dir, "merged_hr_events.csv"))
hr_events[f"death_in_24_hours"] = (hr_events["EventDT"] + pd.Timedelta(hours=24)) >= hr_events["Deceased"]
hr_events.loc[hr_events["Deceased"].isna(), f"death_in_24_hours"] = False
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
)


