import os
import os.path as osp

import pandas as pd

data_dir = "results/predictions/mimic-iv/predicted_optimals_median"
dst_dir = "results/predictions/mimic-iv/"

data = []
for filename in os.listdir(data_dir):
    data.append(pd.read_csv(osp.join(data_dir, filename)))

print(len(data))

data = pd.concat(data, ignore_index=True)
data = data.drop_duplicates("case_id")
print(data.shape[0])
data.to_csv(osp.join(dst_dir, "predicted_optimals.csv"), index=False)
