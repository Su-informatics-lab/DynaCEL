import os
import numpy as np
import pandas as pd

eICU_data_dir = "data/data_for_visualization/eicu"
MIMIC_IV_data_dir = "data/data_for_visualization/mimic-iv"
dst_dir = "data/data_for_visualization/merged"

PW = 12
OW = 24
MOP = [12, 18, 24, 30]
for mop in MOP:
    eICU_data = pd.read_csv(os.path.join(eICU_data_dir, f"model_data_PW_{PW}__MOP_{mop}__OW_{OW}.csv"))
    eICU_data["data_source"] = "eICU"
    # eICU_data = eICU_data.iloc[:, 1:]
    MIMIC_IV_data = pd.read_csv(os.path.join(MIMIC_IV_data_dir, f"model_data_PW_{PW}__MOP_{mop}__OW_{OW}.csv"))
    # MIMIC_IV_data = MIMIC_IV_data.iloc[:, 1:]
    MIMIC_IV_data["data_source"] = "MIMIC_IV"
    merged_data = pd.concat([eICU_data, MIMIC_IV_data], ignore_index=True)
    merged_data.loc[merged_data["age"].isna(), "age"] = 64
    merged_data = merged_data[merged_data["height"] > 50]

    merged_data = merged_data.rename(columns={f"death_in_{OW}_hours": "death"})
    # print(merged_data["death"])
    merged_data.to_csv(os.path.join(dst_dir, f"model_data_PW_{PW}__MOP_{mop}__OW_{OW}.csv"), index=False)