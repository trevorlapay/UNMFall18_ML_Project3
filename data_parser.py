from pathlib import Path
import numpy as np
import time
import pickle
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
IMAGE_HIGHT = 64
IMAGE_WIDTH = 192
IMAGE_CHANNELS = 3
IMAGE_SLICES = 10


def nowStr(): return time.strftime("%Y-%m-%d_%H-%M-%S")


def png_files(path=None, with_paths=False, as_array=False):
    if path is None:
        path = "genres/genres/"
    color_mode = 'grayscale' if IMAGE_CHANNELS == 1 else 'rgb'
    if as_array:
        readFunc = lambda p: img_to_array(load_img(p, color_mode=color_mode))
    else:
        readFunc = lambda p: load_img(p, color_mode=color_mode)
    for im_Path in Path(path).glob('**/*.png'):
        if with_paths:
            yield (im_Path, readFunc(im_Path))
        else:
            yield readFunc(im_Path)

            
def load_png_training_data(validation_split=0):
    num_validation = round(90 * validation_split) * 10 * IMAGE_SLICES
    num_training = 900 * IMAGE_SLICES - num_validation
    x = np.zeros((num_training,
                  IMAGE_HIGHT,
                  IMAGE_WIDTH,
                  IMAGE_CHANNELS))
    y = np.zeros((num_training, 1)).astype(np.int8)
    validation_x = np.zeros((num_validation,
                             IMAGE_HIGHT,
                             IMAGE_WIDTH,
                             IMAGE_CHANNELS))
    validation_y = np.zeros((num_validation, 1)).astype(np.int8)
    if validation_split > 0:
        validation_indexes = [i * IMAGE_SLICES
                              for i in random.sample(range(90), k=round(90*validation_split))]
        validation_indexes = [j for i in validation_indexes for j in range(i,i+IMAGE_SLICES)]
    validation_index = 0
    eyes=[]
    for i, (path, im_arr) in enumerate(png_files(with_paths=True, as_array=True)):
        eyes.append(i)
        if (i%(90*IMAGE_SLICES)) in validation_indexes:
            validation_x[validation_index] = im_arr
            validation_y[validation_index] = genres.index(path.parent.name)
            validation_index += 1
        else:
            if i >= num_training+num_validation:
                print("i = {}\nnum_training = {}\nnum_validation={}\nvalidation_index={}".format(
                        i,
                        num_training,
                        num_validation,
                        validation_index))
                print(validation_indexes)
                break
            x[i-validation_index] = im_arr
            y[i-validation_index] = genres.index(path.parent.name)
    return (x, y), (validation_x, validation_y)


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
