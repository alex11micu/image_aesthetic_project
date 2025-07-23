# Save this as utils/vision.py (or define inline in your script)

import torch
import numpy as np
from PIL import Image

def extract_color_histogram(image: Image.Image, bins_per_channel=32):
    """
    Extract a normalized RGB color histogram from a PIL image.
    Output is a 1D tensor of shape (bins_per_channel * 3,)
    """
    histogram = []
    for channel in range(3):  # R, G, B
        hist = image.histogram()[channel * 256:(channel + 1) * 256]
        hist = np.array(hist[:bins_per_channel * (256 // bins_per_channel)]).reshape(-1, (256 // bins_per_channel)).sum(axis=1)
        histogram.extend(hist)

    hist = np.array(histogram, dtype=np.float32)
    hist /= (hist.sum() + 1e-6)  # normalize
    return torch.tensor(hist)
