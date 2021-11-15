# %%
import os
import time
from datetime import datetime
import cv2
import numpy as np
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

GPIO.setmode(GPIO.BOARD)
inputPin = 15
outputPin = 23
GPIO.setup(inputPin, GPIO.IN)
GPIO.setup(outputPin, GPIO.OUT)
file_counter = 0
counter = 0

if __name__ == '__main__':
    while True:
        x = GPIO.input(inputPin)

        if x == 1:  # Default value on input pin 15
            counter = 0
            GPIO.output(outputPin, 0)  # Led signal off

        if (x == 0) and (counter == 0):  # If sensor active
            time.sleep(0.12)
            ret, frame = camera.read()  # stream from camera
            filename = datetime.now().strftime("%Y-%m-%d") + "-" + datetime.now().strftime("%H:%M:%S") + '_' + str(
                file_counter) + ".png"
            cv2.imwrite(f'img/{filename}', frame)
            counter += 1
            file_counter += 1

    camera.release()
