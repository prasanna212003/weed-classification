import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained deep learning model
model = load_model('model.h5')

# Define class labels
class_labels = ['weed_class1', 'weed_class2', 'weed_class3']

# OpenCV window properties
window_name = 'Weed Classifier'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)
# Open webcam
cap = cv2.VideoCapture(0)
while True:
    # Capture webcam frame
    ret, frame = cap.read()
# Preprocess frame
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
# Run inference on pre-trained model
    predictions = model.predict(input_frame)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
 # Display predicted class label and confidence on frame
    label = '{}: {:.2f}%'.format(predicted_class_label, confidence * 100)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(window_name, frame)
 # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
