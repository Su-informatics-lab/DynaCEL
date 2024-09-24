import os
import os.path as osp
import pickle
import random

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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


def train_model_and_cal_auc(events, y_col="mortality"):
    train_mask = events["subset"] == "train"
    test_mask = events["subset"] == "test"

    # RF training
    X = events[feature_names]
    y = events[y_col]
    stay_ids = events["patientunitstayid"]

    X_train, X_test, y_train, y_test = (
        X[train_mask],
        X[test_mask],
        y[train_mask],
        y[test_mask],
    )

    sm = SMOTE(random_state=42)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train_smote, y_train_smote)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    return (
        model,
        auc,
        model.predict(X),
        model.predict_proba(X)[:, 1],
    )


PW = 12
MOP = 18
OW = 24
K = 5
data_dir = f"data/data_for_training/eicu/datasets/PW_{PW}__MOP_{MOP}__OW_{OW}"
dst_dir = "/N/project/waveform_mortality/personalized_hemodynamics/results/eicu"
model_dir = osp.join(dst_dir, "models")
pred_dir = osp.join(dst_dir, "predictions")
performance_dir = osp.join(dst_dir, "performance")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(performance_dir, exist_ok=True)

schema = {"PW": int, "MOP": int, "OW": int, "Fold": int, "AUC": float}
auc_df = pd.DataFrame(columns=schema.keys()).astype(schema)
for k in range(K):
    merged_events = read_events(osp.join(data_dir, f"merged_events__Fold_{k}.csv"))
    model, auc, y_pred, y_pred_proba = train_model_and_cal_auc(
        merged_events, y_col="mortality"
    )

    print(PW, MOP, OW, k, auc)

    model_path = osp.join(model_dir, f"RF__PW_{PW}__MOP_{MOP}__OW_{OW}__Fold_{k}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    auc_datapoint = pd.DataFrame([[PW, MOP, OW, k, auc]], columns=schema.keys())
    auc_df = pd.concat([auc_df, auc_datapoint], ignore_index=True)
    auc_df.to_csv(
        osp.join(performance_dir, f"RF_AUCs__PW_{PW}__MOP_{MOP}__OW_{OW}.csv"),
        index=False,
    )

    pred_df = pd.DataFrame(
        {
            "patientunitstayid": merged_events["patientunitstayid"],
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "gt": merged_events["mortality"],
            "subset": merged_events["subset"],
        }
    )
    pred_df.to_csv(
        osp.join(pred_dir, f"predictions__PW_{PW}__MOP_{MOP}__OW_{OW}__Fold_{k}.csv"),
    )
