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
3. 
