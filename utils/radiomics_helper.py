import pandas as pd
import numpy as np
import radiomics
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import torch
import radiomics
from radiomics import featureextractor




# --- 1. DEFINE SETTINGS FOR BREAST ULTRASOUND ---
# These settings are critical for US reproducibility
params = {}

# Normalization (CRITICAL for Ultrasound)
params['normalize'] = True
params['normalizeScale'] = 100  # Rescales image intensities to 0-100 range
params['removeOutliers'] = 3    # Remove statistical outliers (noise)

# Discretization (How we group pixel values)
params['binWidth'] = 5          # With scale 100, this creates ~20 bins (good for texture)

# Dimension settings
params['force2D'] = True        # Treat slice-by-slice (if you have a 3D volume)
params['force2Ddimension'] = 0  # 0=Axial, usually correct for single US snapshots

# Feature classes to enable
params['featureClass'] = {
    'shape2D': None,           # VITAL for 2D US images
    'firstorder': None,        # Intensity stats
    'glcm': None,              # Texture (Co-occurrence)
    'glrlm': None,             # Texture (Run Length)
    'glszm': None,             # Texture (Size Zone)
    'gldm': None,              # Texture (Dependence)
    'ngtdm': None              # Texture (Neighbor Tone)
}




#Functions for image handling and display

def read_as_grayscale(img_path):
    # Read image with OpenCV in grayscale mode
    img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        raise ValueError(f"Failed to read image from {img_path}")
    img_cv = img_cv.astype(np.float32) / 255.0
    return img_cv

def reshape_to(img_to, img_from):
    h,w = img_from.shape
    return cv2.resize(img_to, (w, h), interpolation=cv2.INTER_LINEAR)

def imdisp(img):
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.show()




# Function to extract radiomics features

def robust_radiomics_extractor(config_dict):
    extractor = featureextractor.RadiomicsFeatureExtractor(**config_dict)
    # Explicitly enable features
    for feat in ['shape2D', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
        extractor.enableFeatureClassByName(feat)
    return extractor


def extract_radiomic_features(img, mask, params=params):
    extractor = robust_radiomics_extractor(params)

    try:
        result = extractor.execute(img, mask)
        clean_result = {k: v for k, v in result.items() if "diagnostic" not in k}
        return clean_result
    
    except Exception as e:
        print(f"Error: {e}")
        return None