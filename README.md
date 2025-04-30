# Weather_Image_Classification
# Multimedia Project
- - - - - - - - - - - - - - - -
Name: Andrea Figueroa

Title: Multimedia Project

Course: CPSC 4330

Professor Jiangjiang Liu

Term: Fall 2024
- - - - - - - - - - - - - - - -
Followed the Image Classification tutorial in Google Colab and selected a database from Kaggle named Weather Image Recognition by Jehan Bhathena. 

Link to Kaggle database: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data

Link to Image Classification tutorial: https://www.tensorflow.org/tutorials/images/classification

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

### Installation
	1. Download the file 'ProjectMultimedia.ipynb'
	2. Open Google Colab and upload the downloaded 'ProjectMultimedia.ipynb' file
	3. Once opened in Google colab, select 'Runtime' from the top navigation bar and click 'Run all'
	
	
### Usage
This project follows the TensorFlow Image Classification tutorial on Google Colab. I chose an image dataset from Kaggle named Weather Image Recognition by Jehan Bhathena (link above). The dataset contains 6,862 images, split into 11 subfolder categories: dew, fogsmog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, and snow. 

Here is a step-by-step rundown of the project:

To start, the necessary libraries are imported as follows:

```
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
import pathlib
import kagglehub
```

The Kaggle dataset is downloaded:

```
#Dataset kaggle link:
# https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data
# Download the latest version of the dataset
path = kagglehub.dataset_download("jehanbhathena/weather-dataset")

data_dir = pathlib.Path(path)
data_dir = data_dir/'dataset'  #Goes into main dataset folder, which contains all the category folders
image_count = len(list(data_dir.glob('*/*.jpg')))
```

Now that we can access the Kaggle image dataset, we can print the number of images and open individual images:

```
# Print the number of images
print("Number of images in the dataset:", image_count)

#Outputs some images from dew folder
dew = list(data_dir.glob('dew/*'))
PIL.Image.open(str(dew[0]))
PIL.Image.open(str(dew[1]))

#Outputs some images from frost folder
frost = list(data_dir.glob('frost/*'))
PIL.Image.open(str(frost[0]))
PIL.Image.open(str(frost[1]))
```

Now that the Kaggle dataset is ready to go, it's time to load data using a Keras utility.

First, we create a dataset:

```
#Parameters:
batch_size = 32 
img_height = 180
img_width = 180

#80% training and 20% testing using validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, # Directory containing the image folders
  validation_split = 0.2,
  subset = "training",
  seed = 123,
  image_size = (img_height, img_width), # Resize images
  batch_size = batch_size  # Number of images to return per batch
)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, # Directory containing the image folders
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width), # Resize images
  batch_size=batch_size  # Number of images to return per batch
)
```

The following lines output the subfolder names, which are the categories of the images, as well as display the first nine images from the training dataset:

```
#Subfolder names, outputs directory names in alphabetical order
class_names = train_ds.class_names
print(class_names)

#Displays the first nine images from the training dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
```

Now that the dataset has been created, we need to configure it for performance:

```
#Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

Now we standardize the data and apply a normalization layer in order to make the input values ideal for a neural network:

```
#Standardize the data
normalization_layer = layers.Rescaling(1./255)

#Applies the normalization layer to the dataset
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
```

The dataset is now ready to go; we now create and compile the Keras Sequential model:

```
#Create a basic Keras model
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

This line outputs a table showing all the layers of the network:

```
#Model summary (view all the layers of the network)
model.summary()
```

The model is then trained; in the output you can see each Epoch trial as well as their corresponding ms/step, accuracy, loss, and value accuracy/loss:

```
#Train the model
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

These lines visualize the training results, outputting two plots 'Training and Validation Accuracy' and 'Training and Validation Loss':

```
#Visualize the training results (outputs two plots)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

So far, we have created a basic Keras model that has not been tuned for high accuracy, so now we have to account for overfitting (which is when the difference in accuracy between training and validation accuracy is noticeable). The first way we do this is by using data augmentation, which uses random transformations that the model can use to generalize data better:

```
#To fight overfitting in the training process, use data augmentation and add dropout to the model

#Data augmentation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
```

You can see visual examples of the random transformations with these lines:

```
#Visualize a few augmentated examples
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
```

Sidenote: A second approach we can use to fight overfitting is by using dropout regularization

Now we create a new neural network before training it using the augmented images:

```
#creating a new neural network and training it using the augmented images
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])
```

Here we compile and train the model:

```
#Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

Now we can visualize the training results, in which there now is less overfitting than before (reflected in the two plots outputted):

```
#Visualize training results (now with less overfitting than before, and training and validation accuracy are closer aligned)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

Finally, now that the model has been created/trained and overfitting has been reduced, we can test our model to classify an image that wasn't included in the training or validation sets.

In the following lines of code, an new image of a sandstorm is used, and the model outputs its category prediction and confidence percentage:

```
#Predict on new data
sandstorm_url = "https://t4.ftcdn.net/jpg/06/96/34/45/360_F_696344593_ABB9foTEZu1xnOwnVzFCzj1Xs7T3IarT.jpg"
sandstorm_path = tf.keras.utils.get_file('sandstorm_photo', origin=sandstorm_url)

img = tf.keras.utils.load_img(
    sandstorm_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
```

Lastly, this section converts the Keras model into a TensorFlow Lite model, also testing it with a new image and printing out the corresponding category prediction and percentage of confidence:

```
#Convert the Keras Sequential model to a TensorFlow Lite model
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

#Run the TensorFlow Lite model
TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

#Print signatures from converted model to get names of the inputs and outputs
interpreter.get_signature_list()

#Test the loaded TensorFlow Model by performing inference on a sample image
classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite

#Use TensorFlow Lite model to classify images that weren't included in the training/validation sets
predictions_lite = classify_lite(keras_tensor_93=img_array)['output_0']
score_lite = tf.nn.softmax(predictions_lite)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)

print(np.max(np.abs(predictions - predictions_lite)))


```

###Conclusion
Overall, this project creates a basic Kera model for image classification, training it with Kaggle dataset chosen above and testing it with a new image not used in its training/validation. The Kera model is then converted into the TensorFlow Lite format. 
