# Speech_Recognition_System

To clone this repo, please execute:
```
git clone https://github.com/tsaishien-chen/Speech_Recognition_System.git
```

## Requirements
* Python 3.5 or newer

## Directory
* `data/` : 14,224 training speech (1 sec duration clips), which is the subset of Speech Commands Datasets released by TensorFlow: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/dataand and is constituted with 6 command words.
** to download the dataset, run
```
wget --no-check-certificate https://drive.google.com/uc?export=download&id=<FILE_ID>
unzip data.zip
```
* `example_wav/` : an example wav file from each command word
* `model/` : pretrained CNN model to predict word
    
## Execute

### I. Visualize
run
```
python3 visualizer.py example_wav/yes.wav
```
to visualize the specific wav file.

### II. Train
run
```
python3 train.py
```
to train the speech recognition model.

### III. Predict
run
```
python3 predict.py example_wav/yes.wav
```
to predict the class of command for the specific wav file.

