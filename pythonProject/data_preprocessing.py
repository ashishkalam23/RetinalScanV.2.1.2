# data_preprocessing.py
import os
import cv2
import numpy as np
import zipfile
import argparse

def load_images_from_folder(folder):
    """Load all images from the specified folder and return a list of images."""
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def convert_to_grayscale(images):
    """Convert a list of images to grayscale."""
    grayscale_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    return grayscale_images

def apply_clahe(images, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a list of images."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_images = [clahe.apply(img) for img in images]
    return clahe_images

def resize_images(images, size=(512, 512)):
    """Resize a list of images to the specified size."""
    resized_images = [cv2.resize(img, size, interpolation=cv2.INTER_AREA) for img in images]
    return resized_images

def normalize_images(images):
    """Normalize a list of images to the range [0, 1]."""
    normalized_images = [img.astype('float32') / 255.0 for img in images]
    return normalized_images

def preprocess_images(folder):
    """Preprocess all images in the specified folder."""
    images = load_images_from_folder(folder)
    images = convert_to_grayscale(images)
    images = apply_clahe(images)
    images = resize_images(images)
    images = normalize_images(images)
    return np.array(images)

def extract_images_from_zip(zip_path, extract_to_folder):
    """Extracts images from a zip file to the specified folder."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
    print(f"Images extracted to {extract_to_folder}")

def main():
    parser = argparse.ArgumentParser(description="Process some images from a ZIP file.")
    parser.add_argument('zip_path', type=str, help='Path to the ZIP file containing images')
    parser.add_argument('--output', type=str, default='processed_images.npy', help='Output file name for the processed images')

    args = parser.parse_args()

    # Create a directory to extract images
    extract_to_folder = os.path.join(os.path.dirname(args.zip_path), "extracted_images")
    os.makedirs(extract_to_folder, exist_ok=True)

    # Extract images
    extract_images_from_zip(args.zip_path, extract_to_folder)

    # Now process the images
    processed_images = preprocess_images(extract_to_folder)
    np.save(args.output, processed_images)
    print(f"Processed images saved to '{args.output}'.")

if __name__ == "__main__":
    main()
