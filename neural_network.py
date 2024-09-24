{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Neural Network: \
\
import tensorflow as tf        # imports the TensorFlow library\
import os  # os library imported which provides a way to interact with the operating system\
\
# Code to optimise memory consumption\
# Avoid OOM errors by setting GPU Memory Consumption Growth\
gpus = tf.config.experimental.list_physical_devices('GPU')\
for gpu in gpus:  \
    tf.config.experimental.set_memory_growth(gpu, True)\
\
import cv2  # imports OpenCV\
import imghdr  # used to identify the type of an image file by examining its content\
\
data_dir = r'C:\\Nathans Projects\\EE297_Final\\data'\
image_exts = ['jpeg', 'jpg', 'bmp', 'png']\
\
# removes dodgy images from the data directory using\
for image_class in os.listdir(data_dir):  # loops inside every folder in our data directory\
    for image in os.listdir(os.path.join(data_dir, image_class)):    \
        image_path = os.path.join(data_dir, image_class, image)\
        try:  \
            img = cv2.imread(image_path)   # tries and sees if the image can be opened by OpenCV \
            tip = imghdr.what(image_path)  # checks if our image matches one of the types in image_exts\
            if tip not in image_exts:  \
                print('Image not in ext list \{\}'.format(image_path))\
                os.remove(image_path)  # If the image is weird remove it from the file \
        except Exception as e:  \
            print('Issue with image \{\}'.format(image_path)) \
\
# Training and validating the neural network\
import numpy as np  # Python library for numerical computing to support arrays and matrices \
from matplotlib import pyplot as plt  # used for creating visualizations such as graphs and charts \
\
data = tf.keras.utils.image_dataset_from_directory(data_dir)  # creates dataset object containing images and labels\
data = data.map(lambda x, y: (x/255, y))  # normalizes images\
\
# Define and train the CNN model\
from tensorflow.keras.models import Sequential\
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten\
from tensorflow.keras.losses import CategoricalCrossentropy\
from tensorflow.keras.optimizers import Adam\
\
model = Sequential()\
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))\
model.add(MaxPooling2D())\
model.add(Conv2D(32, (3,3), 1, activation='relu'))\
model.add(MaxPooling2D())\
model.add(Conv2D(64, (3,3), 1, activation='relu'))\
model.add(MaxPooling2D())\
model.add(Flatten())\
model.add(Dense(256, activation='relu'))\
model.add(Dense(3, activation='softmax'))\
\
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])\
\
logdir = 'logs'\
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)\
\
hist = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[tensorboard_callback])\
\
# Plotting loss and accuracy\
fig = plt.figure()\
plt.plot(hist.history['loss'], color='teal', label='loss')\
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')\
fig.suptitle('Loss', fontsize=20)\
plt.legend(loc="upper left")\
plt.show()\
\
fig = plt.figure()\
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')\
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')\
fig.suptitle('Accuracy', fontsize=20)\
plt.legend(loc="upper left")\
plt.show()\
\
# Save the trained model\
model.save(os.path.join(r'C:\\Nathans Projects\\EE297_Final', 'brown_clear_green_glass.h5'))\
}