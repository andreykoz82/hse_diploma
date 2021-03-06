# HSE: OCR for Medical Label Verification
Development of a Low-cost Industrial OCR System with Deep Learning Technology

## Instructions:
### Generate SyntText dataset
1. Clone SynthText repository: `git clone -b python3 https://github.com/ankush-me/SynthText.git`
2. Use the instructions in SynthText repository to download all necessary files
3. Make first run of generator: `python gen.py`
4. Copy modified generator script from `\synthtext\mygen.py` to synthtext folder
5. Run generator: `python mygen.py`
As the result you will have a new folders with images and labels in the `\results` folder

### Train Yolov5 model with synthtext dataset
1. Clone Yolov5 repository: `git clone https://github.com/ultralytics/yolov5`
2. Install requirements: `pip install -qr requirements.txt`
3. Make a train-test split of synthtext dataset using python script `\synthtext\split.py`. This will create new folder `dataset` with train and validation images and labels.
4. Modify `\synthtext\data.yaml` file according to your needs (paths to train and val folders, number of classes, classes names). 
5. Train Yolov5 model (specify path to `data.yaml`): `python train.py --img 640 --batch 16 --epochs 3 --data data.yaml --weights yolov5s.pt`
6. Trained model can be downloaded from the model folder `\models`

### Convert to TensorRT object
1. Clone TensorRTX repository: `git clone https://github.com/wang-xinyu/tensorrtx.git`
2. Generate .wts from pytorch model *.pt:\
`cp {tensorrtx}/yolov5/gen_wts.py {ultralytics}/yolov5`\
`cd {ultralytics}/yolov5`\
`python gen_wts.py -w yolov5s.pt -o yolov5s.wts`\
`// a file 'yolov5s.wts' will be generated.`\
3. Build tensorrtx/yolov5:\
`cd {tensorrtx}/yolov5/`\
`// update CLASS_NUM in yololayer.h if your model is trained on custom dataset`\
`mkdir build`\
`cd build`\
`cp {ultralytics}/yolov5/yolov5s.wts {tensorrtx}/yolov5/build`\
`cmake ..`\
`make`\
`sudo ./yolov5 -s [.wts] [.engine] [s]  // serialize model to plan file`\

### Running on Jetson Nano

![alt text](img/results.png)
