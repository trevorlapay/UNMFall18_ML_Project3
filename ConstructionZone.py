import sunau
import scipy
import numpy as np
import os
import pickle
import time
from scipy.fftpack import fft
from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import subprocess
import tensorflow
from keras.layers import Dense, Dropout
from keras.layers import Conv1D
from keras.models import model_from_json
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
TRAINING_FILE_NAME = "TrainingData.pkl"
numframes = 650000
batch_size = 64
epochs = 30
num_classes = 10
SEED = 42


# Load training data for each file as integer array using sunau.
# Each class gets an integer value (0-9) appended to each list.
# Unpack the files in our project directory as-is for this to work.

def loadTrainingDataAu():
    training_data = []
    try:
        training_data = loadPickle(TRAINING_FILE_NAME)
    except:
        for num, genre in enumerate(genres):
            for filename in os.listdir('genres/genres/' + genre):
                if filename == '.DS_Store':
                    continue
                f = sunau.Au_read('genres/genres/' + genre + '/' + filename)
                # Fast Fourier on readframes data to pull out most important features
                audio_data = fft(np.frombuffer(f.readframes(numframes), dtype=np.int16))
                audio_data_list = np.append(audio_data, [num]).tolist()
                training_data.append(audio_data_list)
        savePickle(training_data, TRAINING_FILE_NAME)
        training_data = loadPickle(TRAINING_FILE_NAME)

    # Sanity dump for values.
    for item in training_data:
        print(item)

def loadTrainingDataWav():
    training_data = []
    labels = []
    for num, genre in enumerate(genres):
        for track_num, filename in enumerate(os.listdir('genres/genres/' + genre)):
            if not filename.endswith((".wav")):
                continue
            sample_rate, X = scipy.io.wavfile.read('genres/genres/' + genre + "/" + filename)
            # Fast Fourier on readframes data to pull out most important features
            audio_data = abs(scipy.fft(X)[:numframes])
            audio_data_list = audio_data.tolist()
            labels.append(num)
            training_data.append(audio_data_list)
    training_matrix = np.array(training_data)
    print(training_matrix.shape)
    # Convert labels to array of binary values (colloquially known as one hot labels)
    label_matrix_hot = to_categorical(np.array(labels))
    return training_matrix, label_matrix_hot

def loadTestingDataWav():
    testing_data = []
    for track_num, filename in enumerate(os.listdir('validation/rename/')):
        if not filename.endswith((".wav")):
            continue
        sample_rate, X = scipy.io.wavfile.read('validation/rename/' + filename)
        # Fast Fourier on readframes data to pull out most important features
        audio_data = abs(scipy.fft(X)[:numframes])
        audio_data_list = audio_data.tolist()
        testing_data.append(audio_data_list)
    testing_matrix = np.array(testing_data)
    print(testing_matrix.shape)
    return testing_matrix

def getTestingFilenames():
    testfiles= []
    for track_num, filename in enumerate(os.listdir('validation/rename/')):
        if not filename.endswith((".wav")):
            continue
        testfiles.append(filename[:-4])
    return testfiles

def splitTraining(training_matrix, label_matrix_hot):
    (x_train, x_val, y_train, y_val) = train_test_split(training_matrix, label_matrix_hot, test_size=0.3,
                                                        random_state=SEED)
    return x_train, y_train, x_val, y_val

def loadCompileModel():
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Flatten(input_shape=(numframes, )))
    model.add(Dense(128, activation=tensorflow.nn.relu))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation=tensorflow.nn.relu))
    model.add(Dense(10, activation=tensorflow.nn.softmax))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model

def fitModel(x_train, y_train, x_val, y_val):

    x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
    x_val = tensorflow.keras.utils.normalize(x_val, axis=1)

    model = loadCompileModel()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

def fitModelNoSplit(x_train, y_train):
    model = loadCompileModel()
    x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    evalSerialize(model, x_train, y_train)

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


def predict(model):
    prediction = model.predict(loadTestingDataWav())
    predictions = []
    for encoding in prediction:
        predictions.append(genres[encoding.tolist().index(1)])
    zip_predictions = zip(getTestingFilenames(), predictions)
    print(zip_predictions)
    file = open('submit.txt', 'w')
    file.write("id,class\n")
    for pred in zip_predictions:
        file.write(pred[0] + ".au," + pred[1] + '\n')
    file.close()


def main():
    # To generate model files, run the fitModelNoSplit method. This will serialize
    # your weights so you don't need to generate them again.
    # loadTrainingData()
    # auToWav()
    # spectrograms()
    training_matrix, label_matrix_hot = loadTrainingDataWav()
    fitModelNoSplit(training_matrix, label_matrix_hot)
    model = loadModelFromJSON()
    model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['accuracy'])
    predict(model)

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