import os
import os.path as osp
import pickle
import torch
import numpy as np

import torch
import torchvision.transforms as T
from torchvision import datasets
from torchvision import models
from tqdm import tqdm
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = "Food-101N_release"
meta_dir = osp.join(data_root, "meta")

verified_train = pd.read_csv(osp.join(meta_dir, "verified_train.tsv"), sep='\t')
verified_val = pd.read_csv(osp.join(meta_dir, "verified_val.tsv"), sep='\t')

verified_train.set_index("class_name/key", inplace=True)
verified_val.set_index("class_name/key", inplace=True)

img_paths = np.loadtxt(osp.join(data_root,"feats", "image_paths.txt"), dtype=str)
features = np.loadtxt(osp.join(data_root,"feats", "feats.txt"))
labels = np.loadtxt(osp.join(data_root,"feats", "labels.txt"), dtype=float).astype(np.int64)

verified_label = []
feats = []
labs = []
trainval = []
imgps = []

for p, f, l in zip(img_paths, features, labels):
    key = p.split(os.sep)[-2:]
    key = "/".join(key)

    if key in verified_train.index:
        lb = (verified_train.loc[key]["verification_label"])
        train = 1
    elif key in verified_val.index:
        lb = int(verified_val.loc[key]["verification_label"])
        train = 0
    else:
        continue

    verified_label.append(lb)
    feats.append(f)
    labs.append(l)
    imgps.append(key)
    trainval.append(train)

df = pd.DataFrame({"features": feats, "class_labels": labs, "verified_labels": verified_label, "trainval": trainval}, index=imgps)
df.to_csv(osp.join(data_root, "food101n_dataframe.csv"))