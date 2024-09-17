import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 22})

def read_events(filepath):
    events = pd.read_csv(filepath)
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    # events["dischtime"] = pd.to_datetime(events["dischtime"], errors="coerce")
    events["deathtime"] = pd.to_datetime(events["deathtime"], errors="coerce")
    events["intime"] = pd.to_datetime(events["intime"], errors="coerce")
    events["outtime"] = pd.to_datetime(events["outtime"], errors="coerce")
    events["time_diff"] = pd.to_timedelta(events["time_diff"], errors="coerce")
    return events

def death_survival_barplot(data, ys, title, save_path):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 10), sharex=True, gridspec_kw={"hspace": 0})
    sns.barplot(data=data, x="time_points", y=ys[0], color="#CC79A7", ax=ax1)
    sns.barplot(data=data, x="time_points", y=ys[1], color="#0072B2", ax=ax2)
    
    # y_max = data[ys].max(axis=None)
    ax1.set_ylim(0, 8000)
    ax2.set_ylim(0, 60000)
    ax1.set_ylabel("# Expired")
    ax2.set_ylabel("# Survival")
    ax2.invert_yaxis()

    max_time = data["time_points"].max()
    ax2.set_xticks(range(0, 102, 6))
    ax2.set_xlabel("Time")
    for label in ax2.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()


data_dir = "/N/project/waveform_mortality/xiang/personalized_hemodynamics/data/data_for_training/mimic-iv/"
dst_dir = "/N/project/waveform_mortality/xiang/personalized_hemodynamics/results/barplots/mimic-iv/chartevents"
os.makedirs(dst_dir, exist_ok=True)

events = read_events(os.path.join(data_dir, "merged_hr_events.csv"))
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
    windowed_events = events[(events["time_diff"] >= start_time) & (events["time_diff"] < end_time)]
    windowed_events = windowed_events.drop_duplicates("stay_id")
    n_expired = (windowed_events["hospital_expire_flag"]==1).sum()
    n_survival = (windowed_events["hospital_expire_flag"]==0).sum()
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

save_path = os.path.join(dst_dir, "mimic-iv.pdf")
death_survival_barplot(data, ys=["n_expired", "n_survival"], title="Time vs # Expired/Survival", save_path=save_path)

save_path = os.path.join(dst_dir, "mimic-iv_log.pdf")
death_survival_barplot(data, ys=["log_n_expired", "log_n_survival"], title="Time vs Log # Expired/Survival", save_path=save_path)


save_path = os.path.join(dst_dir, "mimic-iv_prob.pdf")
death_survival_barplot(data, ys=["expired_rates", "survival_rates"], title="Time vs Expired/Survival Rates", save_path=save_path)



