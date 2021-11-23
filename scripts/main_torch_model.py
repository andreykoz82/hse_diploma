# %%
import torch
import time
import cv2
import RPi.GPIO as GPIO
from scripts.utils import visualize

print('Loading model')
model = torch.hub.load('yolov5', 'custom',
                       path='/home/andrey/Jetson_Yolo/models/last_1000_epochs.pt',
                       source='local')  # local repo
model.eval()
model.cuda()
print('Done')

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
            time.sleep(0.12)  # Sensor signal delay
            start_time = time.time()
            ret, frame = camera.read()  # stream from camera

            print('Making prediction')

            with torch.no_grad():
                results = model([frame], size=640)
            res = results.pandas().xyxy[0]
            res = res.query('confidence>=0.65')
            predicted_bbox = res[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy(dtype='int').tolist()
            pred_labels = res.name.to_list()

            print("Done in %s seconds ---" % (time.time() - start_time))
            visualize(frame, predicted_bboxes=predicted_bbox,
                      predicted_class_labels=pred_labels)

            counter += 1

    camera.release()
