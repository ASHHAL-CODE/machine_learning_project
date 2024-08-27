import tensorflow as tf
from tensorflow.keras import models, layers
import os
import cv2
import numpy as np


# Load dataset
def load_data():
    data = []
    labels = []
    gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'subtract', 'multiply', 'divide']
    for gesture in gestures:
        path = f'dataset/{gesture}'
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (64, 64))
            data.append(img_array)
            labels.append(gestures.index(gesture))
    
    data = np.array(data).reshape(-1, 64, 64, 1)
    data = data / 255.0
    labels = np.array(labels)
    return data, labels

data, labels = load_data()

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(14, activation='softmax')  # 10 digits + 4 operators
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=10, validation_split=0.2)

# Save the model
model.save('gesture_model.h5')
