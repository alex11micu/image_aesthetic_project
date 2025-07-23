# enhanced_dataset.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def get_hsv_histogram(img: Image.Image, bins=32):
    """Extract HSV histogram"""
    hsv = img.convert("HSV")
    h = np.array(hsv)[:, :, 0]
    hist, _ = np.histogram(h, bins=bins, range=(0, 255), density=True)
    return torch.tensor(hist, dtype=torch.float32)

def get_rgb_histogram(img: Image.Image, bins=32):
    """Extract RGB histogram"""
    img_array = np.array(img)
    hist = []
    for channel in range(3):
        channel_hist, _ = np.histogram(img_array[:, :, channel], bins=bins, range=(0, 255), density=True)
        hist.extend(channel_hist)
    return torch.tensor(hist, dtype=torch.float32)

def get_lab_histogram(img: Image.Image, bins=32):
    """Extract LAB histogram using PIL (no cv2 required)"""
    img_array = np.array(img)
    
    # Convert RGB to LAB using color transformations
    # This is a simplified LAB conversion
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    # Normalize RGB values
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    
    # Convert to XYZ (simplified)
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    
    # Convert to LAB (simplified)
    # Using a simplified LAB approximation
    l = 116 * np.power(y, 1/3) - 16
    a = 500 * (np.power(x, 1/3) - np.power(y, 1/3))
    b_lab = 200 * (np.power(y, 1/3) - np.power(z, 1/3))
    
    # Clip values and scale to 0-255
    l = np.clip(l, 0, 100) * 2.55
    a = np.clip(a + 128, 0, 255)
    b_lab = np.clip(b_lab + 128, 0, 255)
    
    # Create histograms
    hist = []
    for channel in [l, a, b_lab]:
        channel_hist, _ = np.histogram(channel, bins=bins, range=(0, 255), density=True)
        hist.extend(channel_hist)
    
    return torch.tensor(hist, dtype=torch.float32)

def get_composition_features(img: Image.Image):
    """Extract composition features (RGB mean/std)"""
    img_array = np.array(img)
    features = []
    
    # RGB mean and std
    for channel in range(3):
        features.append(np.mean(img_array[:, :, channel]))
        features.append(np.std(img_array[:, :, channel]))
    
    return torch.tensor(features, dtype=torch.float32)

class EnhancedAestheticDataset(Dataset):
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

        # Extract multiple color histograms
        hsv_hist = get_hsv_histogram(img)
        rgb_hist = get_rgb_histogram(img)
        lab_hist = get_lab_histogram(img)
        
        # Extract composition features
        comp_features = get_composition_features(img)

        return img_tensor, hsv_hist, rgb_hist, lab_hist, comp_features, mean_score