
# coding: utf-8

# # Spectrogram Classifier

# ## Imports

# In[1]:


import tensorflow.python.keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.utils import to_categorical
from IPython.display import display
from tensorflow.keras.callbacks import EarlyStopping

import data_parser


# ## Constants and Utilities

# In[2]:


genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

IMAGE_HIGHT = data_parser.IMAGE_HIGHT
IMAGE_WIDTH = data_parser.IMAGE_WIDTH
IMAGE_CHANNELS = data_parser.IMAGE_CHANNELS
IMAGE_SLICES = data_parser.IMAGE_SLICES

def nowStr(): return time.strftime("%Y-%m-%d_%H-%M-%S")


# In[34]:


def plot_loss_curve(history, title_prefix=''):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training loss', 'Validation Loss'])
    plt.xlabel('Epochs ')
    plt.ylabel('Loss')
    plt.title(title_prefix+'Loss Curves')
    plt.show()

def plot_acc_curve(history, title_prefix=''):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epochs ')
    plt.ylabel('Accuracy')
    plt.title(title_prefix+'Accuracy Curves')
    plt.show()


# ## Collect the Data

# In[4]:


(training_data, training_labels),(val_data, val_labels)=data_parser.load_png_training_data(validation_split=0.15)
training_labels_onehot = to_categorical(training_labels)
val_labels_onehot = to_categorical(val_labels)

for i in range(0,training_data.shape[0],450):
    print(genres[training_labels[i,0]])
    display(array_to_img(training_data[i]))


# ## Binary Model per Genre

# ### Build Model

# In[22]:


genre_models = []
for genre in genres:
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(8,6),
                     strides=(4,2),
                     input_shape=(IMAGE_HIGHT,
                                  IMAGE_WIDTH,
                                  IMAGE_CHANNELS),
                     data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,(1,2),(1,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(64,(2,2),(1,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(2,2),(1,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #TODO: Change `loss='binary_crossentropy'` to something better for 10 class problems.
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    genre_models.append(model)


# ### Train Model

# In[23]:


# new
genre_histories = []
for i, model in enumerate(genre_models):
    print("Training "+genres[i]+" model.")
    history = model.fit(x=training_data,
                        y=training_labels_onehot[:,i],
                        epochs=10,
                        batch_size=100,
                        verbose=1,
                        validation_data=(val_data, val_labels_onehot[:,i]),
                        shuffle=True,
                        class_weight = {0: 1., 1: 9.},
                        callbacks=[EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=2,
                                   verbose=1, mode='auto')])
    genre_histories.append(history)
    plot_loss_curve(history)
    plot_acc_curve(history)
    print(model.evaluate(training_data, training_labels_onehot[:,i]))


# In[35]:


for i, history in enumerate(genre_histories):
    plot_loss_curve(history, title_prefix=genres[i]+' ')
    plot_acc_curve(history, title_prefix=genres[i]+' ')


# In[24]:


for i, model in enumerate(genre_models):
    model.save_weights(nowStr() + '_Spectrogram_Classifier_Weights_genre{}.h5'.format(i))


# ### Test Model

# In[36]:


def classify(instance, by_ordering=False):
    best = (0, 0)
    if by_ordering:
        temp = [(i, model.predict(instance)[0,0]) for i, model in enumerate(genre_models)]
        list.sort(temp, key=lambda x: x[1], reverse=True)
        return temp
    for i, model in enumerate(genre_models):
        temp = model.predict(instance)[0,0]
        if best[1] < temp:
            best = (i, temp)
    return best[0]

def sum_confidence_voting(instances):
    ballots = [classify(instances[i:i+1], by_ordering=True) for i in range(10)]
    totals = {genre: 0 for genre in range(10)}
    for ballot in ballots:
        for genre, conf in ballot:
            totals[genre] += conf
    return max(totals.items(), key=lambda x: x[1])

def ranked_choice_election(instances):
    ballots = [classify(instances[i:i+1], by_ordering=True) for i in range(10)]
    winner = None
    counts = {genre: 0 for genre in range(10)}
    numCounting = 2
    while winner is None:
        counts = {k: 0 for k in counts.keys()}
        for ballot in ballots:
#             print('ballot')
            numCounted = 0
            for genre, conf in ballot:
#                 print("{} with {}".format(genre, conf))
                if genre in counts.keys():
                    numCounted += 1
                    counts[genre] += 1/numCounted
                    if numCounted >= numCounting:
                        break
        if max(counts.values()) >= 8:
            winner = max(counts.items(), key=lambda x: x[1])
        else:
            for k, v in list(counts.items()):
                if v == 0:
                    del counts[k]
            loser = min(counts.items(), key=lambda x: x[1])
#             print("Dropping {}".format(loser))
            del counts[loser[0]]
    return winner
            
        


# In[38]:


correct = 0
incorrect = 0

for i in range(0,training_data.shape[0],IMAGE_SLICES):
    winner = sum_confidence_voting(training_data[i:i+IMAGE_SLICES])
    if i %(9*IMAGE_SLICES) == 0 and i > 0:
        print("{} -> {} with {} votes".format(training_labels[i,0], winner[0], winner[1]))
        print("Accuracy: {}".format(correct/(correct+incorrect)))
    if winner[0] == training_labels[i,0]:
        correct += 1
    else:
        incorrect += 1
print("Accuracy: {}".format(correct/(correct+incorrect)))


# ## Merging Model

# ### Build Model

# In[95]:


merge_model = tensorflow.keras.models.Sequential()
merge_model.add(Dense(10, input_dim=10, activation='relu'))
merge_model.add(Dense(10, activation='softmax'))
merge_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])


# ### Train Model

# In[49]:


def evaluate_confidences(x):
    y = np.zeros((x.shape[0], 10))
    for i in range(x.shape[0]):
        for genre in range(10):
            y[i, genre] = genre_models[genre].predict(x[i:i+1])[0,0]
    return y


# In[50]:


training_confidences = evaluate_confidences(training_data)
val_confidences = evaluate_confidences(val_data)


# In[96]:


merge_history = merge_model.fit(training_confidences,
                                training_labels_onehot,
                                batch_size=10,
                                epochs=10,
                                verbose=1,
                                validation_data=(val_confidences,
                                                 val_labels_onehot),
                               callbacks=[EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=2,
                                   verbose=1, mode='auto')])
plot_loss_curve(merge_history, title_prefix='Merged ')
plot_acc_curve(merge_history, title_prefix='Merged ')


# ### Test Model

# In[121]:


correct = 0
incorrect = 0

for i in range(val_confidences.shape[0]):
    winner = merge_model.predict_classes(val_confidences[i:i+1])[0]
    if i % 9 == 0 and i > 0:
        print("{} -> {}".format(val_labels[i,0], winner))
    if winner == val_labels[i,0]:
        correct += 1
    else:
        incorrect += 1
print("Accuracy: {}".format(correct/(correct+incorrect)))


# In[119]:


np.mean(val_labels)


# ## Unified Model

# ### Build Model

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=8, 
                 kernel_size=2,
                 strides=1,
                 input_shape=(IMAGE_HIGHT,
                              IMAGE_WIDTH,
                              IMAGE_CHANNELS),
                 data_format="channels_last"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(16,2,1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32,2,1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64,2,1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#TODO: Change `loss='binary_crossentropy'` to something better for 10 class problems.
model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
genre_models.append(model)


# ### Train Model

# In[ ]:


history = model.fit(x=training_data,
                    y=training_labels_onehot,
                    epochs=30,
                    batch_size=100,
                    verbose=1,
                    validation_data=(val_data, val_labels_onehot),
                    shuffle=True)
plot_loss_curve(history)
plot_acc_curve(history)


# ### Test Model

# In[ ]:


model.evaluate(training_data, training_labels_onehot)

