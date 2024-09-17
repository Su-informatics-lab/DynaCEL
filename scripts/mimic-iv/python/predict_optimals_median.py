import argparse
import math
import os
import pickle
import random
import time
import warnings
from statistics import median

import numpy as np
import pandas as pd
from scipy import stats

# Start the timer
start_time = time.time()


parser = argparse.ArgumentParser()
parser.add_argument("--batch_idx", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=100)
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
prediction_window = 12
mop = 18
outcome_window = 24
data_dir = "/N/project/waveform_mortality/xiang/Projects/icu-contour/results/v2"
model_dir = f"/N/project/waveform_mortality/xiang/Model_data/Models/"
output_dir = f"/N/project/waveform_mortality/xiang/personalized_hemodynamics/results/predictions/mimic-iv/predicted_optimals_median"
os.makedirs(output_dir, exist_ok=True)

# merged_measurements = pd.read_csv(os.path.join(data_dir, f"icustays_model_data_PW_{prediction_window}__MOP_{mop}__OW_{outcome_window}.csv"))
measurements_median = pd.read_csv(
    os.path.join(
        data_dir,
        f"icustays_model_data_median_PW_{prediction_window}__MOP_{mop}__OW_{outcome_window}.csv",
    )
)

n_folds = 5
models = []
for fold in range(n_folds):
    with open(
        os.path.join(
            model_dir,
            f"model__PW_{prediction_window}__MOP_{mop}__OW_{outcome_window}__Fold_{fold}.pkl",
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
df_res


cross_measurements = measurements_median.drop(columns=["m1_valuenum", "m2_valuenum"])
n_batches = int(math.ceil(cross_measurements.shape[0] / batch_size))
if batch_idx >= n_batches:
    print("Batch idx >= # batches")
    assert False

start_index = batch_idx * batch_size
end_index = (batch_idx + 1) * batch_size
print(batch_idx, start_index, end_index)
batched_cross_measurements = cross_measurements.iloc[start_index:end_index, :]
batched_cross_measurements = batched_cross_measurements.merge(df_res, how="cross")
for model_i, model in enumerate(models):
    name_out = f"pred_{model_i}"
    Y = model.predict_proba(batched_cross_measurements[feature_names])
    pred_prob = Y[:, 1]
    batched_cross_measurements[name_out] = pred_prob

batched_cross_measurements = get_avg_ci(batched_cross_measurements)

final_result_df = pd.DataFrame()
for case_id in list(batched_cross_measurements["stay_id"].unique()):
    measurements_of_interest = measurements_median.loc[
        measurements_median["stay_id"] == case_id, :
    ]
    hr = measurements_of_interest["m1_valuenum"].item()
    bp = measurements_of_interest["m2_valuenum"].item()

    case_df = batched_cross_measurements[
        batched_cross_measurements["stay_id"] == case_id
    ]
    average_x, average_y = get_average_target_point(case_df, 100)

    actual_risk = optimal_risk = 0
    for model_i, model in enumerate(models):
        actual_data = measurements_median[measurements_median["stay_id"] == case_id]
        Y = model.predict_proba(actual_data[feature_names])
        actual_risk = Y[:, 1].item()

        optimal_data = measurements_median[
            measurements_median["stay_id"] == case_id
        ].copy(True)
        optimal_data.loc[:, "m1_valuenum"] = average_x
        optimal_data.loc[:, "m2_valuenum"] = average_y
        Y = model.predict_proba(optimal_data[feature_names])
        optimal_risk = Y[:, 1].item()

    # Create a DataFrame to store the results
    result_df = pd.DataFrame(
        {
            "case_id": [case_id],
            "hr": [hr],
            "bp": [bp],
            "optimal_hr": [average_x],
            "optimal_bp": [average_y],
            "actual_risk": [actual_risk],
            "optimal_risk": [optimal_risk],
        }
    )

    # Append result_df to final_result_df
    final_result_df = pd.concat([final_result_df, result_df], ignore_index=True)


final_result_df = final_result_df.merge(
    measurements_median[["stay_id", "death_in_24_hours"]],
    left_on="case_id",
    right_on="stay_id",
    how="left",
)

# Drop the stay_id column after merge if not needed
final_result_df.drop(columns=["stay_id"], inplace=True)


output_path = os.path.join(
    output_dir, f"predict_optimal_median_batch_{batch_idx}_batch_size_{batch_size}.csv"
)
final_result_df.to_csv(output_path, index=False)

# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
elapsed_time = elapsed_time / 60

print(f"Time taken to run the code: {elapsed_time:.2f} min")
