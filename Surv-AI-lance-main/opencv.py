import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.lite import Interpreter
import time

# Load the TFLite model
model_path = "/home/madbonze/Downloads/GKV1.tflite"
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
categories = ['Money', 'Card', 'Wallet', 'Smartphone', 'Knife', 'Pistol']

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img = Image.fromarray(img).convert('RGB')
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = categories[np.argmax(output)]
    confidence = np.max(output)

    # Display the frame and prediction results
    cv2.putText(frame, f"Predicted Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Live Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Add a delay to avoid excessive resource consumption
    time.sleep(0.1)

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
