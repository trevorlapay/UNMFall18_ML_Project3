# UNM CS 429/529 Music Classification Project

by Trevor La Pay, Luke Hanks, and Kyle Jurney

## Usage

### Frequency Intensity Model

To run our best model (frequency intensity), put the following lines of code in `main()` of ML_Project3.py:

```python
auToWav()
training_matrix, label_matrix_hot = loadTrainingDataWav()
model = fitModelNoSplit(training_matrix, label_matrix_hot)
predict(model)
```

### CEPS Model

To run CEPS, put the following lines of code in `main()` of ML_Project3.py:

```python
training_matrix, label_matrix_hot = loadTrainingDataWav(True)
model = fitModelNoSplit(training_matrix, label_matrix_hot)
predict(model, True) 
```

The `predict()` function will output a kaggle-ready file to the root project directory names submit.txt.

### Spectrogram Model(s)

Open and run SpectrogramClassifier.ipynb in Jupyter Notebooks. This does not require .wav files.

### Requirements

Prior to running any code, ensure that all import libraries are installed (keep in mind that 
tensorflow is not currently compatible with Python 3.7 - please use 3.6.)

We require wav versions of all files. To convert all files to wav:

1. Install the [SoX library](http://sox.sourceforge.net/). 
2. Add it to your PATH variable.
3. Ensure that the genres and validation archives are expanded/unpacked/unzipped into the root project directory.
4. Set the python kernel's working directory to the root project directory.
5. Call the function `auToWav()` found in ML_Project3.py.
