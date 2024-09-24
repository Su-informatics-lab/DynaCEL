import os
import os.path as osp
import pickle
import random

import cupy as cp
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

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


def hyperparameter_tuning(events, y_col="mortality"):
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

    param_grid_1 = {
        "tree_method": ["hist", "approx"],
        "max_depth": range(3, 12, 1),
        "min_child_weight": [1, 3, 5, 7, 9, 200],
    }

    param_grid_2 = {
        "tree_method": ["hist", "approx"],
        "gamma": [i / 10.0 for i in range(0, 11)],
        "min_child_weight": [3, 5, 7, 9],
    }

    param_grid_3 = {
        "tree_method": ["hist", "approx"],
        # "min_child_weight": [7, 9],
        # "subsample": [i / 10.0 for i in range(6, 10)],
        # "colsample_bytree": [i / 10.0 for i in range(6, 10)],
        "reg_alpha": [1e-5, 1e-2, 0.1, 1, 100],
        "reg_lambda": list(range(0, 11)),
    }

    param_grid_4 = {
        "tree_method": ["hist", "approx"],
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "n_estimators": [10, 100, 1000, 2000, 5000],
    }

    search = GridSearchCV(
        estimator=XGBClassifier(
            tree_method="hist",
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=3,
            min_child_weight=9,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.6,
            objective="binary:logistic",
            nthread=-1,
            scale_pos_weight=1,
            reg_alpha=1,
            reg_lambda=10,
            seed=42,
            device="cuda",
        ),
        param_grid=param_grid_4,
        scoring="roc_auc",
        n_jobs=1,
        cv=5,
        verbose=0,
    )

    # no over-sampling as oversampling leads to data leakage
    search.fit(cp.array(X_train), y_train)

    return search.best_params_, search.best_score_


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
    best_params, best_score = hyperparameter_tuning(merged_events, y_col="mortality")
    print(best_params)
    print(best_score)
