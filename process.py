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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = "."
test_image_dir = osp.join("food-101", "images")
train_image_dir = osp.join("Food-101N_release", "images")

# basic transformation
trans = T.Compose([
    T.Resize(224),
    T.RandomCrop(224),
    T.ToTensor(),]
)

# with a little data augmentation
trans2 = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.3),
    T.ToTensor(),]
)

model = models.resnet50(pretrained=True).to(device)
model.eval()

class ImageFolderP(datasets.ImageFolder):
    def __getitem__(self, index):
        return *super().__getitem__(index), self.samples[index][0] 


def extract_feat(image_dir, transform = None):
    feat_dir = image_dir.replace("images", "feats")
    if not osp.exists(feat_dir):
        os.mkdir(feat_dir)

    feat_ls = []
    label_ls = []
    imgpth_ls = []

    dataset = ImageFolderP(root=image_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 8)
    for images, labels, image_path in tqdm(dataloader):
        images = images.to(device)
        feats = model(images).detach().cpu().numpy()
        feat_ls.append(feats)
        label_ls.append(labels)
        imgpth_ls.append(image_path)

    feat_arr = np.vstack(feat_ls)
    label_arr = np.hstack(label_ls).astype(np.int)
    imgpth_arr = np.hstack(imgpth_ls).astype(str)
    feat_fn = "feats.txt"
    label_fn = "labels.txt"
    imgpth_fn = "image_paths.txt"
    class_to_label_fn = "class_to_label.pkl"
    # array(features)
    np.savetxt(osp.join(feat_dir, feat_fn), feat_arr)
    # array(labels)
    np.savetxt(osp.join(feat_dir, label_fn), label_arr)
    # array(image_paths)
    np.savetxt(osp.join(feat_dir, imgpth_fn), imgpth_arr, fmt='%s')
    # dict{class name: label}
    with open(class_to_label_fn, "rb") as f:
        pickle.dump(dataset.class_to_idx, f)



print("Extracting Food 101N")
extract_feat(train_image_dir, transform=trans)
print("Extracting Food 101")
extract_feat(test_image_dir, transform=trans)