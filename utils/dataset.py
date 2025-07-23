# dataset.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def get_color_histogram(img: Image.Image, bins=32):
    hsv = img.convert("HSV")
    h = np.array(hsv)[:, :, 0]
    hist, _ = np.histogram(h, bins=bins, range=(0, 255), density=True)
    return torch.tensor(hist, dtype=torch.float32)

class AestheticDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = int(row["image_num"])
        mean_score = torch.tensor(row["mean_score"], dtype=torch.float32)

        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        img = Image.open(image_path).convert("RGB")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = img

        color_hist = get_color_histogram(img)
        return img_tensor, color_hist, mean_score
