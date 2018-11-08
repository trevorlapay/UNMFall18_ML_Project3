import scipy
import numpy as np
import os
import pickle
import time
from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import subprocess
import tensorflow
import librosa
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
TRAINING_FILE_NAME = "TrainingData.pkl"
numframes = 660000 # Default to frequency amplitude frames
batch_size = 100
epochs = 25
num_classes = 10
SEED = 42
hop_length = 256

# Loads from training directory. Ceps argument = True if you want to use 13 CEPS features.
def loadTrainingDataWav(ceps = False):
    training_data = []
    labels = []
    for num, genre in enumerate(genres):
        for track_num, filename in enumerate(os.listdir('genres/genres/' + genre)):
            if not filename.endswith((".wav")):
                continue
            if ceps:
                audio_data_list = []
                y, sr = librosa.load('genres/genres/' + genre + "/" + filename)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
                for cep in mfcc:
                    audio_data_list.append(np.mean(cep))
                labels.append(num)
                training_data.append(audio_data_list)

                # Fast Fourier on readframes data to pull out most important features
            else:
                sample_rate, X = scipy.io.wavfile.read('genres/genres/' + genre + "/" + filename)
                audio_data = abs(scipy.fft(X)[:numframes])
                audio_data_list = audio_data.tolist()
                labels.append(num)
                training_data.append(audio_data_list)
    training_matrix = np.array(training_data)
    print(training_matrix.shape)
    # Convert labels to array of binary values (colloquially known as one hot labels)
    label_matrix_hot = to_categorical(np.array(labels))
    return training_matrix, label_matrix_hot

# Loads testing files for preditcion. Ceps = True if you want CEPS features.
def loadTestingDataWav(ceps = False):
    testing_data = []
    for track_num, filename in enumerate(os.listdir('validation/rename/')):
        if not filename.endswith((".wav")):
            continue
        if ceps:
            audio_data_list = []
            y, sr = librosa.load('validation/rename/' + filename)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
            for cep in mfcc:
                audio_data_list.append(np.mean(cep))
            testing_data.append(audio_data_list)
        else:
            sample_rate, X = scipy.io.wavfile.read('validation/rename/' + filename)
            # Fast Fourier on readframes data to pull out most important features
            audio_data = abs(scipy.fft(X)[:numframes])
            audio_data_list = audio_data.tolist()
            testing_data.append(audio_data_list)
    testing_matrix = np.array(testing_data)
    print(testing_matrix.shape)
    return testing_matrix

# Helper for getting filenames for submit file create.
def getTestingFilenames():
    testfiles= []
    for track_num, filename in enumerate(os.listdir('validation/rename/')):
        if not filename.endswith((".wav")):
            continue
        testfiles.append(filename[:-4])
    return testfiles

# Split a given data set into both training and testing sets. Useful when tinkering with model.
def splitTraining(training_matrix, label_matrix_hot):
    (x_train, x_val, y_train, y_val) = train_test_split(training_matrix, label_matrix_hot, test_size=0.3,
                                                        random_state=SEED)
    return x_train, y_train, x_val, y_val

# Load and compile a NN model.
def loadCompileModel(ceps = False):
    if ceps:
        shape = 13
    else:
        shape = numframes
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Flatten(input_shape=(shape, )))
    model.add(Dense(128, activation=tensorflow.nn.relu))
    model.add(Dropout(0.30))
    model.add(Dense(128, activation=tensorflow.nn.relu))
    model.add(Dense(10, activation=tensorflow.nn.softmax))
    model.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model

# Fit our model to data (with split between train and test)
def fitModel(x_train, y_train, x_val, y_val):

    x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
    x_val = tensorflow.keras.utils.normalize(x_val, axis=1)

    model = loadCompileModel()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

# Fit our model to data, no splitting (the "production" mode when generating submit files)
def fitModelNoSplit(x_train, y_train):
    model = loadCompileModel()
    x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    evalSerialize(model, x_train, y_train)

# Serialize weights.
def evalSerialize(model, x_train, y_train):
    # evaluate the model
    scores = model.evaluate(x_train, y_train, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

# Convert au to wav
def auToWav():
    directories = [Path(tpl[0]) for tpl in os.walk(Path('.'))]
    for directory in directories:
        print(str(directory))
        for file in os.listdir(directory):
            if file[-3:] == '.au':
                print("    " + file)
                subprocess.run(["sox", str(directory / file), str(directory / (file[:-3] + ".wav"))])


def spectrograms():
    directories = [Path(tpl[0]) for tpl in os.walk(Path('.'))]
    for directory in directories:
        print(str(directory))
        for file in os.listdir(directory):
            if file[-4:] == '.wav':
                sample_rate, X = wavfile.read(directory/file)
                print("    "+file)
                specgram(X, Fs=sample_rate, xextent=(0,30))
                plt.show()
                break # Remove this to plot all the files, not just the first of each genre.

# Run prediction and generate submit file. Ceps = True if using Ceps features.
def predict(model, ceps=False):
    prediction = model.predict_classes(loadTestingDataWav(ceps))
    predictions = []
    for encoding in prediction:
        predictions.append(genres[encoding])
    zip_predictions = zip(getTestingFilenames(), predictions)
    print(zip_predictions)
    file = open('submit.txt', 'w')
    file.write("id,class\n")
    for pred in zip_predictions:
        file.write(pred[0] + ".au," + pred[1] + '\n')
    file.close()

# Note: pass True to loadTrainingDataWav and as second arg in preict() to run CEPS features.
def main():
    # To generate model files, run the fitModelNoSplit method. This will serialize
    # your weights so you don't need to generate them again.
    # loadTrainingData()
    # auToWav()
    # spectrograms()
    training_matrix, label_matrix_hot = loadTrainingDataWav()
    # fitModelNoSplit(training_matrix, label_matrix_hot)
    fitModelNoSplit(training_matrix, label_matrix_hot)
    model = loadModelFromJSON()
    model.compile(optimizer="nadam", loss="categorical_crossentropy",metrics=['accuracy'])
    predict(model)

# load model from JSON file
def loadModelFromJSON():
    model = loadCompileModel()
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    return model

#%% Define pickle file functions
def savePickle(obj, fileName=None):
    if fileName is None: fileName = nowStr()+"pickleFile.pkl"
    if fileName[-4:] != ".pkl": fileName += ".pkl"
    with open(fileName, 'wb') as pklFile:
        pickle.dump(obj, pklFile)
        pklFile.close()

def loadPickle(fileName="pickleFile.pkl"):
    if fileName[-4:] != ".pkl": fileName += ".pkl"
    obj = None
    with open(fileName, 'rb') as pklFile:
        obj = pickle.load(pklFile)
        pklFile.close()
    return obj
if __name__ == "__main__": main()

def nowStr(): return time.strftime("%Y-%m-%d_%H-%M-%S")