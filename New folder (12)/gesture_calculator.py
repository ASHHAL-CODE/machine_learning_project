import cv2
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
model = tf.keras.models.load_model('gesture_model.h5')

# Gesture labels
gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']

cap = cv2.VideoCapture(0)
current_input = ""

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = image.reshape(1, 64, 64, 1) / 255.0
    return image

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Assume hand is in the center of the frame
    hand_image = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    processed_image = preprocess_image(hand_image)
    prediction = model.predict(processed_image)
    gesture_index = np.argmax(prediction)
    gesture = gestures[gesture_index]

    # Update calculator input
    current_input += gesture

    # Display current input on the screen
    cv2.putText(frame, current_input, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Calculator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
