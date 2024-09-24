import os
import os.path as osp

import pandas as pd

data_dir = "results/mimic-iv/predicted_optimals/batched_predicted_optimals"
dst_dir = "results/mimic-iv/predicted_optimals"

data = []
for filename in os.listdir(data_dir):
    data.append(pd.read_csv(osp.join(data_dir, filename)))

print(len(data))

data = pd.concat(data, ignore_index=True)
data = data.drop_duplicates("case_id")
print(data.shape)
data.to_csv(osp.join(dst_dir, "predicted_optimals.csv"), index=False)
