from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Halfborn/OD-WeaponDetection.git

!cp -r "/content/OD-WeaponDetection/Weapons and similar handled objects/Sohas_weapon-Classification" "/content/drive/MyDrive"

import os
import pandas as pd
from sklearn.utils import shuffle
from keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.vgg16 import VGG16
from keras import layers, models
import tensorflow as tf

data_dir = '/content/drive/MyDrive/Sohas_weapon-Classification'
categories = ['Money', 'Card', 'Wallet', 'Smartphone', 'Knife', 'Pistol']
num_classes = len(categories)
img_height, img_width = 224, 224
batch_size = 32

# Create ImageDataGenerator with preprocessing
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    validation_split=0.2
)

# Load train and validation datasets using image_dataset_from_directory
train_ds = data_generator.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    seed=42,
    shuffle=True,
    class_mode='sparse',
    classes=categories
)

validation_ds = data_generator.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    seed=42,
    shuffle=True,
    class_mode='sparse',
    classes=categories
)

# Create the base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

# Build the model architecture
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)
callbacks = [early_stop, lr_scheduler]

print(model.summary())

# Train the model
epochs = 20
model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=callbacks
)

model.save("/content/drive/MyDrive/GunsKnives1.h5")

import os
import numpy as np
import random
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("/content/drive/MyDrive/GunsKnives.h5")

# Get the subdirectories representing each class
class_dirs = [os.path.join(data_dir, cls) for cls in categories]

# Select 50 random images from different class folders
selected_images = []
for class_dir in class_dirs:
    class_images = os.listdir(class_dir)
    random.shuffle(class_images)
    selected_images.extend([os.path.join(class_dir, img) for img in class_images[:50 // num_classes]])

# Shuffle the selected images
random.shuffle(selected_images)

# Iterate over the selected images and make predictions
for image_path in selected_images:
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)

    # Make the prediction
    prediction = model.predict(img)
    predicted_label = categories[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Get the actual label from the image file path
    actual_label = os.path.basename(os.path.dirname(image_path))

    # Display the results
    print("Image:", os.path.basename(image_path))
    print("Actual Label:", actual_label)
    print("Predicted Label:", predicted_label)
    print("Confidence:", confidence)
    print("-----------")

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("/content/drive/MyDrive/GunsKnives.h5")

# Get the subdirectories representing each class
class_dirs = [os.path.join(data_dir, cls) for cls in categories]

# Select 50 random images from different class folders
selected_images = []
for class_dir in class_dirs:
    class_images = os.listdir(class_dir)
    random.shuffle(class_images)
    selected_images.extend([os.path.join(class_dir, img) for img in class_images[:50 // num_classes]])

# Shuffle the selected images
random.shuffle(selected_images)

# Iterate over the selected images and make predictions
for image_path in selected_images:
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)

    # Make the prediction
    prediction = model.predict(img)
    predicted_label = categories[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Get the actual label from the image file path
    actual_label = os.path.basename(os.path.dirname(image_path))

    # Normalize and clip the image array
    img = img[0] / 255.0
    #img = np.clip(img, 0, 1)

    # Display the image and the results
    plt.imshow(img)
    plt.title(f"Actual Label: {actual_label}\nPredicted Label: {predicted_label}\nConfidence: {confidence}")
    plt.axis("off")
    plt.show()

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("/content/drive/MyDrive/GunsKnives.h5")
image_path = '/content/Guns.jpg'
img = image.load_img(image_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = tf.keras.applications.vgg16.preprocess_input(img)

# Make the prediction
prediction = model.predict(img)
predicted_label = categories[np.argmax(prediction)]
confidence = np.max(prediction)

# Get the actual label from the image file path
actual_label = os.path.basename(os.path.dirname(image_path))

# Normalize and clip the image array
img = img[0] / 255.0
#img = np.clip(img, 0, 1)

# Display the image and the results
plt.imshow(img)
plt.title(f"Actual Label: {actual_label}\nPredicted Label: {predicted_label}\nConfidence: {confidence}")
plt.axis("off")
plt.show()

import tensorflow as tf
model = tf.keras.models.load_model('/content/drive/MyDrive/GunsKnives.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
file = open( 'GKV1.tflite' , 'wb' ) 
file.write( tflmodel )