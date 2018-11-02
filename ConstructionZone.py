import sunau
import numpy as np
import os
import pickle
import time

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
                audio_data = np.frombuffer(f.readframes(numframes), dtype=np.int16)
                audio_data_list = np.append(audio_data, [num]).tolist()
                training_data.append(audio_data_list)
        savePickle(training_data, TRAINING_FILE_NAME)
        training_data = loadPickle(TRAINING_FILE_NAME)

    # Sanity dump for values.
    for item in training_data:
        print(item)

def main():
    loadTrainingData()



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