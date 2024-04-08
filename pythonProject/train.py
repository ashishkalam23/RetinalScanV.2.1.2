# train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def train(train_data, train_labels, val_data, val_labels, epochs=10, batch_size=32):
    # Define the model architecture
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=train_data[0].shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_data, val_labels))

    return model, history

if __name__ == "__main__":
    # Example usage:
    # Load your training and validation data here
    train_data = np.load('train_data.npy')
    train_labels = np.load('train_labels.npy')
    val_data = np.load('val_data.npy')
    val_labels = np.load('val_labels.npy')

    # Train the model
    trained_model, training_history = train(train_data, train_labels, val_data, val_labels)

    # Optionally save the trained model
    trained_model.save('trained_model.h5')
