import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the image (update the filename if needed)
# image_path = os.path.join('datasets', os.listdir('datasets')[0])
image_path = '/workspaces/CT-PR-0155_Vein-Finder/datasets/WIN_20250901_15_01_08_Pro.jpg'

# Load the image
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return img


# Preprocess the image: grayscale, denoise, enhance contrast, normalize
def preprocess_image(img, output_dir=None):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '1_gray.png'), gray)

    # Denoise with Gaussian blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '2_denoised.png'), denoised)

    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(denoised)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '3_contrast.png'), contrast)

    # Normalize intensity
    norm = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, '4_normalized.png'), norm)
    return norm

# Enhance veins using contrast and thresholding
def enhance_veins(img):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    # Thresholding to segment veins
    _, veins = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return veins

# Visualize results
def show_images(original, preprocessed, veins):
    # Save images to files in the output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'original_image.png'), original)
    cv2.imwrite(os.path.join(output_dir, 'preprocessed_image.png'), preprocessed)
    cv2.imwrite(os.path.join(output_dir, 'vein_map.png'), veins)
    print(f'Images saved in {output_dir} as original_image.png, preprocessed_image.png, and vein_map.png')

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    img = load_image(image_path)
    preprocessed = preprocess_image(img, output_dir=output_dir)
    veins = enhance_veins(preprocessed)
    show_images(img, preprocessed, veins)
