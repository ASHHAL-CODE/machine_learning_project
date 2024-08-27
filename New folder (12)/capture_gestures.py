import cv2
import os

# Create directories for each gesture
gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'subtract', 'multiply', 'divide']
for gesture in gestures:
    os.makedirs(f'dataset/{gesture}', exist_ok=True)

cap = cv2.VideoCapture(0)
current_gesture = '0'
image_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the current gesture
    cv2.putText(frame, f'Gesture: {current_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the image to the dataset folder
    if image_count < 100:
        cv2.imwrite(f'dataset/{current_gesture}/image_{image_count}.png', frame)
        image_count += 1
    else:
        print(f"Captured 100 images for {current_gesture}. Change the gesture.")
        current_gesture = input("Enter next gesture: ")
        image_count = 0

    cv2.imshow('Capture Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
