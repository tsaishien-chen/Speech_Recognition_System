# Speech Recognition System

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
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1seJQJ9amLVaTUB-uXB1Jq0ob0cbuSDTJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1seJQJ9amLVaTUB-uXB1Jq0ob0cbuSDTJ" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
```
* `example_wav/` : an example wav file from each command word
* `model/` : pretrained CNN model to predict word
    
## Execute

### Visualize:
to visualize the raw waveform and spectrogram of specific wav file, run
```
python3 visualizer.py example_wav/yes.wav
```

### Train:
to execute data-preprocessing and  train the speech recognition model, run
```
python3 train.py
```

### Predict:
to predict the class of command for the specific wav file, run
```
python3 predict.py example_wav/yes.wav
```
