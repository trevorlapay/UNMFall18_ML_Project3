import sunau
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
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
TRAINING_FILE_NAME = "TrainingData.pkl"
numframes = 1000

# Load training data for each file as integer array using sunau.
# Each class gets an integer value (0-9) appended to each list.
# Unpack the files in our project directory as-is for this to work.

def loadTrainingData():
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
    spectrograms()



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