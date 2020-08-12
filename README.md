# Speech_Recognition_System

To clone this repo, please execute:
```
git clone https://github.com/tsaishien-chen/Speech_Recognition_System.git
```

## Requirements
* Python 3.5 or newer:  

## Directory
* `data/`: 14,224 training speech (1 sec duration clips), which is the subset of Speech Commands Datasets released by TensorFlow: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/dataand and is constituted with 6 command words.
** check next section (Execute) to download the dataset
* `example_wav/`: an example wav file from each command word
* `model/`: pretrained CNN model to predict word
    
## Execute

### I. Visualize
1. Execute:

1. Download the dataset (save all training images into `src/total_train/` and all testing images into `src/test/`).

2. Execute:
```
bash src/Pytorch/test.sh [directory of sample_submission.csv]
```

3. The final result will show up here: `src/Pytorch/submit/result.csv`.
