import argparse
import math
import os
import pickle
import random
import time
import warnings
from statistics import median

import cupy as cp
import numpy as np
import pandas as pd
from scipy import stats

# Start the timer
start_time = time.time()


parser = argparse.ArgumentParser()
parser.add_argument("--batch_idx", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=5000)
args = parser.parse_args()

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


def get_avg_ci(df):
    cols = ["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"]
    means = df[cols].mean(axis=1)
    stds = df[cols].std(axis=1)
    sem = stds / np.sqrt(len(cols))
    confidence_level = 0.95

    # Calculate the margin of error (critical value * standard error)
    z_score = np.abs(stats.norm.ppf((1 - confidence_level) / 2))
    margin_of_error = z_score * sem

    lower_bound = means - margin_of_error
    upper_bound = means + margin_of_error

    df["average_pred"] = means
    df["lower_bound"] = lower_bound
    df["upper_bound"] = upper_bound
    df["CI_Interval"] = upper_bound - lower_bound
    return df


def get_average_target_point(df, nums):
    df_sorted = df.sort_values(by="average_pred")
    smallest_100 = df_sorted.head(nums)
    average_x = smallest_100["m1_valuenum"].mean()
    average_y = smallest_100["m2_valuenum"].mean()
    return average_x, average_y


batch_idx = args.batch_idx
batch_size = args.batch_size
PW = 12
MOP = 18
OW = 24
data_dir = f"data/data_for_training/mimic-iv/datasets/PW_{PW}__MOP_{MOP}__OW_{OW}"
model_dir = f"results/eicu/models"
output_dir = f"results/mimic-iv/predicted_optimals/batched_predicted_optimals"
os.makedirs(output_dir, exist_ok=True)
model_name = "XGBoost"
events = pd.read_csv(os.path.join(data_dir, "merged_events.csv"))

K = 5
models = []
for k in range(K):
    with open(
        os.path.join(
            model_dir,
            f"{model_name}__PW_{PW}__MOP_{MOP}__OW_{OW}__Fold_{k}.pkl",
        ),
        "rb",
    ) as model_file:
        model = pickle.load(model_file)
        models.append(model)

x_range = range(40, 151)  # x range from 40 to 150 (inclusive)
y_range = range(60, 201)  # y range from 60 to 200 (inclusive)
# Generate the hr_bp_combination within the specified ranges
hr_bp_combination = [(x, y) for x in x_range for y in y_range]
# Create hr_arr and bp_arr from hr_bp_combination
hr_arr = [x for x, y in hr_bp_combination]
bp_arr = [y for x, y in hr_bp_combination]

df_res = pd.DataFrame(columns={"m1_valuenum": int, "m2_valuenum": int})
df_res["m1_valuenum"] = hr_arr
df_res["m2_valuenum"] = bp_arr

cross_events = events.drop(columns=["m1_valuenum", "m2_valuenum"])
n_batches = int(math.ceil(cross_events.shape[0] / batch_size))
assert batch_idx <= n_batches

start_index = batch_idx * batch_size
end_index = (batch_idx + 1) * batch_size
print(batch_idx, start_index, end_index)

batched_cross_events = cross_events.iloc[start_index:end_index, :]
batched_cross_events = batched_cross_events.merge(df_res, how="cross")
for model_i, model in enumerate(models):
    name_out = f"pred_{model_i}"
    Y = model.predict_proba(cp.array(batched_cross_events[feature_names]))
    pred_prob = Y[:, 1]
    batched_cross_events[name_out] = pred_prob

batched_cross_events = get_avg_ci(batched_cross_events)


optimal_hrs_bps = []
for case_id in list(batched_cross_events["stay_id"].unique()):
    events_of_interest = events.loc[events["stay_id"] == case_id, :]
    hr = events_of_interest["m1_valuenum"].item()
    bp = events_of_interest["m2_valuenum"].item()

    case_df = batched_cross_events[batched_cross_events["stay_id"] == case_id]
    average_x, average_y = get_average_target_point(case_df, 100)
    optimal_hrs_bps.append(
        {"case_id": case_id, "optimal_hr": average_x, "optimal_bp": average_y}
    )

optimal_hrs_bps = pd.DataFrame(optimal_hrs_bps)
events = pd.merge(events, optimal_hrs_bps, left_on="stay_id", right_on="case_id")

model = models[-1]
actual_risks = model.predict_proba(cp.array(events[feature_names]))
events["actual_risk"] = actual_risks[:, 1]


feature_names[0], feature_names[1] = "optimal_hr", "optimal_bp"
optimal_risks = model.predict_proba(cp.array(events[feature_names]))
events["optimal_risk"] = optimal_risks[:, 1]

events = events.rename(columns={"m1_valuenum": "hr", "m2_valuenum": "bp"})
events = events[
    [
        "case_id",
        "hr",
        "bp",
        "optimal_hr",
        "optimal_bp",
        "actual_risk",
        "optimal_risk",
        "mortality",
    ]
]


output_path = os.path.join(
    output_dir, f"predicted_optimals_batch_{batch_idx}_batch_size_{batch_size}.csv"
)
events.to_csv(output_path, index=False)

# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
elapsed_time = elapsed_time / 60

print(f"Time taken to run the code: {elapsed_time:.2f} min")
