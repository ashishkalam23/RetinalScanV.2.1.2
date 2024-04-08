import os
import numpy as np
from data_preprocessing import preprocess_images
from train import train
from evaluate import evaluate


def main():
    # Path to the ZIP file containing images
    zip_path = "C:/Users/aks23/Documents/SFU/Spring 2024/CMPT 340/all-images.zip"

    # Output directory for processed images
    output_dir = "processed_images"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preprocess images
    print("Preprocessing images...")
    processed_images = preprocess_images(zip_path)

    # Save processed images
    for i, img in enumerate(processed_images):
        filename = os.path.join(output_dir, f"image_{i}.npy")
        np.save(filename, img)

    print(f"Processed images saved to '{output_dir}'.")

    # Train the model
    print("Training the model...")
    model = train(processed_images)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate(model)


if __name__ == "__main__":
    main()
