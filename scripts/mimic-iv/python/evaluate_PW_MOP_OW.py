import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
from sklearn import metrics

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
    "mv_none",
    "mv_unknown",
    "race_w",
    "race_b",
    "race_l",
    "race_o",
]


def read_events(filepath):
    events = pd.read_csv(filepath)
    return events


def evaluate_and_cal_auc(model, events, y_col="mortality"):

    X = events[feature_names]
    y = events[y_col]
    stay_ids = events["stay_id"]

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    auc = metrics.roc_auc_score(y, y_pred_proba)
    bacc = metrics.balanced_accuracy_score(y, y_pred)

    return (
        auc,
        bacc,
        model.predict(X),
        y_pred_proba,
    )


PW = 12
MOP = 18
OW = 24

data_dir = f"data/data_for_training/mimic-iv/datasets/PW_{PW}__MOP_{MOP}__OW_{OW}"
dst_dir = "/N/project/waveform_mortality/personalized_hemodynamics/results/mimic-iv"
model_dir = "results/eicu/models"
model_name = "XGBoost"


K = 5
pred_dir = osp.join(dst_dir, "predictions")
performance_dir = osp.join(dst_dir, "performance")
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(performance_dir, exist_ok=True)

schema = {"PW": int, "MOP": int, "OW": int, "Fold": int, "AUC": float}
auc_df = pd.DataFrame(columns=schema.keys()).astype(schema)
merged_events = read_events(osp.join(data_dir, f"merged_events.csv"))
for k in range(K):
    model_path = os.path.join(
        model_dir,
        f"{model_name}__PW_{PW}__MOP_{MOP}__OW_{OW}__Fold_{k}.pkl",
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    auc, bacc, y_pred, y_pred_proba = evaluate_and_cal_auc(
        model, merged_events, y_col="mortality"
    )

    print(PW, MOP, OW, k, auc, bacc)

    auc_datapoint = pd.DataFrame([[PW, MOP, OW, k, auc]], columns=schema.keys())
    auc_df = pd.concat([auc_df, auc_datapoint], ignore_index=True)
    auc_df.to_csv(
        osp.join(
            performance_dir, f"{model_name}_AUCs__PW_{PW}__MOP_{MOP}__OW_{OW}.csv"
        ),
        index=False,
    )

    pred_df = pd.DataFrame(
        {
            "stay_id": merged_events["stay_id"],
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "gt": merged_events["mortality"],
        }
    )
    pred_df.to_csv(
        osp.join(
            pred_dir,
            f"{model_name}_predictions__PW_{PW}__MOP_{MOP}__OW_{OW}__Fold_{k}.csv",
        ),
    )
