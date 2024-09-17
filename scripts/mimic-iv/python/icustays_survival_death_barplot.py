import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 22})

def read_icustays(filepath):
    icustays = pd.read_csv(filepath)
    icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")
    icustays["outtime"] = pd.to_datetime(icustays["outtime"], errors="coerce")
    icustays["time_diff"] = icustays["outtime"] - icustays["intime"]
    return icustays


def read_mortality_status(filepath):
    admissions = pd.read_csv(filepath)
    return admissions[["subject_id", "hadm_id", "hospital_expire_flag"]]

def death_survival_barplot(data, ys, title, save_path):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True, gridspec_kw={"hspace": 0})
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0)  # figure background to be transparent
    sns.barplot(data=data, x="time_points", y=ys[0], color="#CC79A7", ax=ax1)
    sns.barplot(data=data, x="time_points", y=ys[1], color="#0072B2", ax=ax2)
    
    # y_max = data[ys].max(axis=None)
    ax1.set_ylim(0, 10000)
    ax2.set_ylim(0, 80000)
    ax1.set_ylabel("Death (n)", fontsize=36)
    ax2.set_ylabel("Survival (n)", fontsize=36)
    ax2.invert_yaxis()

    max_time = data["time_points"].max()
    ax2.set_xticks(range(0, 102, 6))
    ax2.set_xlabel("Time (h)", fontsize=36)
    for label in ax2.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    ax1.patch.set_alpha(0)
    ax2.patch.set_alpha(0)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    # plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

data_dir = "/N/project/waveform_mortality/xiang/personalized_hemodynamics/data/raw/mimic-iv/mimic-iv-2.2/"
dst_dir = "/N/project/waveform_mortality/xiang/personalized_hemodynamics/results/barplots/mimic-iv/icustays"
os.makedirs(dst_dir, exist_ok=True)

icustays = read_icustays(os.path.join(data_dir, "icu/icustays.csv"))
mortality_status = read_mortality_status(os.path.join(data_dir, "hosp/admissions.csv"))
icustays = pd.merge(icustays, mortality_status, how="left", on=["subject_id", "hadm_id"])

time_points = []
n_expired_list = []
n_survival_list = []
expired_rates = []
survival_rates = []
pw = 1
time = 0
while time <= 96:
    start_time = pd.to_timedelta(time, unit="h")
    end_time = start_time + pd.to_timedelta(pw, unit="h")
    windowed_icustays = icustays[icustays["time_diff"] >= end_time]
    windowed_icustays = windowed_icustays.drop_duplicates("stay_id")
    n_expired = (windowed_icustays["hospital_expire_flag"]==1).sum()
    n_survival = (windowed_icustays["hospital_expire_flag"]==0).sum()
    n_expired_list.append(n_expired)
    n_survival_list.append(n_survival)

    n_total = n_expired + n_survival
    expired_rates.append(n_expired/n_total)
    survival_rates.append(n_survival/n_total)

    time_points.append(time)
    time += 1

n_expired_list = np.array(n_expired_list)
n_survival_list = np.array(n_survival_list)
expired_rates = np.array(expired_rates)
survival_rates = np.array(survival_rates)

data = pd.DataFrame({
    "time_points": time_points, "n_expired": n_expired_list, "n_survival": n_survival_list,
    "log_n_expired": np.log1p(n_expired_list), "log_n_survival": np.log1p(n_survival_list),
    "expired_rates": expired_rates, "survival_rates": survival_rates})

save_path = os.path.join(dst_dir, "mimic-iv.png")
death_survival_barplot(data, ys=["n_expired", "n_survival"], title="Time vs # Expired/Survival", save_path=save_path)

save_path = os.path.join(dst_dir, "mimic-iv_log.png")
death_survival_barplot(data, ys=["log_n_expired", "log_n_survival"], title="Time vs Log # Expired/Survival", save_path=save_path)


save_path = os.path.join(dst_dir, "mimic-iv_prob.png")
death_survival_barplot(data, ys=["expired_rates", "survival_rates"], title="Time vs Expired/Survival Rates", save_path=save_path)



