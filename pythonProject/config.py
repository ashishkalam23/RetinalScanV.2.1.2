# config.py

class Config:
    # Path settings
    data_path = "data/"
    train_images = data_path + "train/"
    test_images = data_path + "test/"
    processed_images = data_path + "processed/"  # Update with the new processed images directory
    model_save_path = "models/u_net.h5"

    # Image processing settings
    img_width = 256  # Update with the new image width
    img_height = 256  # Update with the new image height
    channels = 1  # grayscale

    # Training settings
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
