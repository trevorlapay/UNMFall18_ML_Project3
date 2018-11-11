# UNM CS 429/529 Music Classification Project

by Trevor La Pay, Luke Hanks, and Kyle Jurney

## Usage

To run our best model (frequency intensity), put the following lines of code in main():

    auToWav()
    training_matrix, label_matrix_hot = loadTrainingDataWav()
    model = fitModelNoSplit(training_matrix, label_matrix_hot)
    predict(model)
    
To run CEPS, put thr following lines of code in main():

    training_matrix, label_matrix_hot = loadTrainingDataWav(True)
    model = fitModelNoSplit(training_matrix, label_matrix_hot)
    predict(model, True) 


### Requirements

Prior to running any code, ensure that all import libraries are installed (keep in mind that 
tensorflow is not currently compatible with Python 3.7 - please use 3.6.)

We require wav versions of all files. Do convert all files to wav, install the sox library, add it to your
PATH variable, and run the following:

    auToWav()
    
Ensure that the genres and validation compressed directories are expanded in the root project directory.

[Luke, I am adding to this as if we are submitting the current version. 
Feel free to discard if you make improvements]



