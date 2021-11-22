import sys
import ctypes
from scripts.yolov5_trt import YoLov5TRT, warmUpThread, inferThread
import RPi.GPIO as GPIO
import pycuda.autoinit
import cv2, queue, threading, time

# load custom plugin and engine
PLUGIN_LIBRARY = "/home/andrey/Jetson_Yolo/tensorrtx/yolov5/build/libmyplugins.so"
engine_file_path = "/home/andrey/Jetson_Yolo/tensorrtx/yolov5/build/yolov5s.engine"

if len(sys.argv) > 1:
    engine_file_path = sys.argv[1]
if len(sys.argv) > 2:
    PLUGIN_LIBRARY = sys.argv[2]

ctypes.CDLL(PLUGIN_LIBRARY)

yolov5_wrapper = YoLov5TRT(engine_file_path)


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


camera = VideoCapture(0)

GPIO.setmode(GPIO.BOARD)
inputPin = 15
outputPin = 23
GPIO.setup(inputPin, GPIO.IN)
GPIO.setup(outputPin, GPIO.OUT)
file_counter = 0
counter = 0

for i in range(10):
    # create a new thread to do warm_up
    thread1 = warmUpThread(yolov5_wrapper)
    thread1.start()
    thread1.join()

print('Ready to predict')

if __name__ == '__main__':
    while True:
        x = GPIO.input(inputPin)

        if x == 1:  # Default value on input pin 15
            counter = 0
            GPIO.output(outputPin, 0)  # Led signal off

        if (x == 0) and (counter == 0):  # If sensor active
            time.sleep(0.12)  # Sensor signal delay
            start_time = time.time()
            frame = camera.read()  # stream from camera
            print('Making prediction')

            # create a new thread to do inference
            thread1 = inferThread(yolov5_wrapper, frame[:, :, ::-1])
            thread1.start()
            thread1.join()

            print('Done')
            counter += 1

    camera.release()
    yolov5_wrapper.destroy()
