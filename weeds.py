Python 3.11.2 (tags/v3.11.2:878ead1, Feb  7 2023, 16:38:35) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the input and output directories for each class
class1_input_dir = "/content/drive/MyDrive/StudentProjects/CELOSIA ARGENTEA L"
class1_output_dir ="/content/drive/MyDrive/StudentProjects/celosia_out"
class2_input_dir = "/content/drive/MyDrive/StudentProjects/CROWFOOT GRASS"
class2_output_dir = "/content/drive/MyDrive/StudentProjects/crow_out"
class3_input_dir = "/content/drive/MyDrive/StudentProjects/PURPLE CHLORIS"
class3_output_dir = "/content/drive/MyDrive/StudentProjects/purplr_out"

# Define the target image size
target_size = (224, 224)  # Change the size to whatever you need

# Preprocess images in class 1
for filename in os.listdir(class1_input_dir):
    img = cv2.imread(os.path.join(class1_input_dir, filename))
    resized_img = cv2.resize(img, target_size)
    height, width = resized_img.shape[:2]
    start_row, start_col = int(height * 0.25), int(width * 0.25)
    end_row, end_col = int(height * 0.75), int(width * 0.75)
    cropped_img = resized_img[start_row:end_row, start_col:end_col]
    normalized_img = cropped_img / 255.0
    output_filename = os.path.join(class1_output_dir, filename)
    cv2.imwrite(output_filename, normalized_img)

# Preprocess images in class 2
for filename in os.listdir(class2_input_dir):
    img = cv2.imread(os.path.join(class2_input_dir, filename))
    resized_img = cv2.resize(img, target_size)
    height, width = resized_img.shape[:2]
    start_row, start_col = int(height * 0.25), int(width * 0.25)
    end_row, end_col = int(height * 0.75), int(width * 0.75)
    cropped_img = resized_img[start_row:end_row, start_col:end_col]
    normalized_img = cropped_img / 255.0
    output_filename = os.path.join(class2_output_dir, filename)
    cv2.imwrite(output_filename, normalized_img)

# Preprocess images in class 3
for filename in os.listdir(class3_input_dir):
    img = cv2.imread(os.path.join(class3_input_dir, filename))
    resized_img = cv2.resize(img, target_size)
    height, width = resized_img.shape[:2]
    start_row, start_col = int(height * 0.25), int(width * 0.25)
    end_row, end_col = int(height * 0.75), int(width * 0.75)
    cropped_img = resized_img[start_row:end_row, start_col:end_col]
    normalized_img = cropped_img / 255.0
    output_filename = os.path.join(class3_output_dir, filename)
    cv2.imwrite(output_filename, normalized_img)


# Define the directories for the dataset
train_dir = '/content/drive/MyDrive/student_output/celosia_out-20230416T064611Z-001'
val_dir = '/content/drive/MyDrive/student_output/celosia_out-20230416T064611Z-001'
test_dir = '/content/drive/MyDrive/student_output/celosia_out-20230416T064611Z-001'

# Define the parameters for image resizing and batch size
img_height = 224
img_width = 224
batch_size = 32

# Create data generators for training, validation, and test data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model with an appropriate loss and optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=25, validation_data=val_data)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)

#save the model
model.save('weed_classification_model.h')
from pickle import NONE
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('/content/weed_classification_model.h')

# Define the parameters for image resizing
img_height = 224
img_width = 224

# Define the lower and upper bounds for the color of the weed
lower = np.array([0, 0, 0])
upper = np.array([50, 50, 50])

# Open a video capture device
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    if(frame is not None):
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove the background noise using the GaussianBlur function
      blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to binarize it
      _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

    # Apply a mask to the image to remove the background
      mask = cv2.inRange(frame, lower, upper)
      masked = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Resize the image to match the input size of the trained model
      resized = cv2.resize(masked, (img_height, img_width))

    # Normalize the pixel values
      normalized = resized / 255.0

    # Add a batch dimension to the image
      img = np.expand_dims(normalized, axis=0)

    # Use the trained model to predict the type of weed
      predictions = model.predict(img)
      class_index = np.argmax(predictions[0])
      if class_index == 0:
        label = "Weed Type A"
...       elif class_index == 1:
...         label = "Weed Type B"
...       else:
...         label = "Weed Type C"
... 
...     # Display the label on the frame
...       cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
... 
...     # Display the frame
...       cv2.imshow("Frame", frame)
... 
...     # Press the 'q' key to exit the loop
...       if cv2.waitKey(1) & 0xFF == ord('q'):
...           break
... 
... # Release the video capture device and close all windows
... cap.release()
... cv2.destroyAllWindows()
... import matplotlib.pyplot as plt
... def plot_hist(history):
...   acc = history.history['accuracy']
...   val_acc = history.history['val_accuracy']
...   loss = history.history['loss']
...   val_loss = history.history['val_loss']
...   plt.figure(figsize=(10, 5))
...   plt.subplot(1, 2, 1)
...   plt.plot(acc, label='Training Accuracy')
...   plt.plot(val_acc, label='Validation Accuracy')
...   plt.legend(loc='lower right')
...   plt.title('Training and Validation Accuracy')
...   plt.grid()
...   plt.subplot(1, 2, 2)
...   plt.plot(loss, label='Training Loss')
...   plt.plot(val_loss, label='Validation Loss')
...   plt.legend(loc='upper right')
...   plt.title('Training and Validation Loss')
...   plt.grid()
...   plt.show()
