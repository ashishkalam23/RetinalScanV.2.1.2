# evaluate.py

from model import unet_model
from data_preprocessing import preprocess_images
from config import Config
from sklearn.metrics import classification_report

def evaluate():
    # Load the trained model
    model = unet_model()
    model.load_weights(Config.model_save_path)

    # Preprocess the test images
    test_images = preprocess_images(Config.test_images)

    # Make predictions
    predictions = model.predict(test_images)

    # Evaluate the model
    print("Evaluation Report:")
    print(classification_report(test_images.flatten(), predictions.flatten()))

# If this script is run directly, call the evaluate function
if __name__ == "__main__":
    evaluate()
