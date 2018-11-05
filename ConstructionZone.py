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
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
TRAINING_FILE_NAME = "TrainingData.pkl"
numframes = 1000
batch_size = 64
epochs = 20
num_classes = 10
music_model = None


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
    testing_data = []
    labels = []
    for num, genre in enumerate(genres):
        for filename in os.listdir('genres/genres/' + genre):
            if not filename.endswith((".wav")):
                continue
            sample_rate, X = scipy.io.wavfile.read('genres/genres/' + genre + "/" + filename)
            # Fast Fourier on readframes data to pull out most important features
            audio_data = abs(scipy.fft(X)[:1000])
            audio_data_list = audio_data.tolist()
            labels.append(num)
            training_data.append(audio_data_list)
    training_matrix = np.array(training_data)
    label_matrix_hot = to_categorical(np.array(labels))
    # check training data shape
    print(training_matrix.shape)
    train_X, valid_X, train_label, valid_label = train_test_split(training_matrix, label_matrix_hot, test_size=0.2,
                                                                  random_state=13)
    createModel()
    fashion_train = music_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                      validation_data=(valid_X, valid_label))



def createModel():
    # I scraped some of this off of a demo page. This is only for getting things off the ground and NOT for use in submittal
    global music_model
    music_model = Sequential()
    music_model.add(Dense(12, input_dim=1000, activation='relu'))
    music_model.add(Dense(8, activation='relu'))
    music_model.add(Dense(10, activation='sigmoid'))
    music_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

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


def main():
    # loadTrainingData()
    # auToWav()
    # spectrograms()
    loadTrainingDataWav()
    createModel()



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