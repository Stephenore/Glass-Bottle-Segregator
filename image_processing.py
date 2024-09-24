{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import cv2\
import time\
import serial\
from PIL import Image\
import numpy as np\
from keras.models import load_model\
\
# Connect to the Arduino Uno\
ser = serial.Serial('COM5', 9600)\
\
# Load the model\
model_path = r'C:\\Nathans Projects\\EE297_Final\\brown_clear_green_glass.h5'\
model = load_model(model_path)\
\
# Define the labels for each class\
labels = ['brown', 'green', 'clear']\
\
while True:\
    # Capture an image from the webcam\
    cap = cv2.VideoCapture(0)\
    ret, frame = cap.read()\
    cap.release()\
\
    # Resize the image\
    img = Image.fromarray(frame)\
    img = img.resize((256, 256))\
\
    # Convert the image to a numpy array\
    x = np.array(img)\
\
    # Expand the dimensions of the array to match the model's input shape\
    x = np.expand_dims(x, axis=0)\
\
    # Make the prediction\
    prediction = model.predict(x)\
\
    # Get the predicted class label\
    predicted_class = labels[np.argmax(prediction)]\
\
    # Send a signal to the Arduino based on the prediction\
    if predicted_class == 'brown':\
        ser.write(b'1')  # Brown bottle detected\
        time.sleep(1.5)\
        ser.write(b'4')  # Stop actuator\
    elif predicted_class == 'green':\
        ser.write(b'2')  # Green bottle detected\
        time.sleep(2.5)\
        ser.write(b'4')  # Stop actuator\
    else:\
        ser.write(b'3')  # Clear bottle detected\
}