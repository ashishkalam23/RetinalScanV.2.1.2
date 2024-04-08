# patch_extraction.py
import numpy as np
from config import Config
from data_preprocessing import load_image, preprocess_image

def extract_patches(img, size=64, stride=32):
    """Extract patches from a preprocessed image."""
    patches = []
    for i in range(0, img.shape[0] - size + 1, stride):
        for j in range(0, img.shape[1] - size + 1, stride):
            patch = img[i:i+size, j:j+size]
            patches.append(patch)
    return np.array(patches)
