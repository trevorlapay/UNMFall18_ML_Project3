#%% Imports and constants

import sys
import os
from pathlib import Path
import subprocess
import queue
import threading
import time
import _thread
import librosa.core
import matplotlib
import matplotlib.pyplot as plt
import itertools

AUDIO_TYPE = ".au"
IMAGE_TYPE = ".png"
NUM_IMAGE_SLICES = 10
PLOT_HIGHT = 1 # inches
PLOT_WIDTH = 30/NUM_IMAGE_SLICES # 1 inch per second
DPI = 32 # dots per inch (1D so don't square it.)
REPLACE_EXISTING = True
PRINT_SKIPS = True
CHECK_FOR_MISSING = True
NUM_THREADS = 8

if len(sys.argv) > 1 and sys.argv[1] is not None:
    audFiles = Path(sys.argv[1]).glob('**/*'+AUDIO_TYPE)
    imgFiles = Path(sys.argv[1]).glob('**/*'+IMAGE_TYPE)
else:
    audFiles = itertools.chain(Path("genres/genres/").glob('**/*'+AUDIO_TYPE),
                               Path("validation/rename/").glob('**/*'+AUDIO_TYPE))
    imgFiles = itertools.chain(Path("genres/genres/").glob('**/*'+IMAGE_TYPE),
                               Path("validation/rename/").glob('**/*'+IMAGE_TYPE))
    # Note: Chain returns a generator that is only good for one iteration.

#%% Threading

class myThread (threading.Thread):
    def __init__(self, threadID, q):
        threading.Thread.__init__(self)
        self.id = threadID
        self.q = q
        self.fig = plt.figure(frameon=False)
        self.fig.set_size_inches(PLOT_WIDTH, PLOT_HIGHT)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.set_aspect('auto')
        self.fig.add_axes(self.ax)
    def run(self):
        print("Starting thread_" + str(self.id))
        process_data(self.id, self.q, self.fig, self.ax)
        print("Exiting thread_" + str(self.id))

def process_data(threadID, q, fig, ax):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            audPath, imgPath = q.get()
            queueLock.release()
            print("thread_"+str(threadID)+" processing " + audPath.name)
            autopng(audPath, imgPath, fig, ax)
        else:
            queueLock.release()
            time.sleep(1)


def autopng(audPath, imgPath, fig, ax):
    samples, sample_rate = librosa.core.load(audPath)
    chunkSize = round(len(samples)/NUM_IMAGE_SLICES)
    sampleChunks = [samples[i:min(i + chunkSize, len(samples)-1)] 
                    for i in range(0, len(samples), chunkSize)]
    for i, chunk in enumerate(sampleChunks):
        if i == NUM_IMAGE_SLICES:
#            print("Too many chunks")
            break
        ax.specgram(chunk, Fs=sample_rate, cmap='jet')
        chunkPath = imgPath.with_name((imgPath.stem[:-2]+'{:02}'+IMAGE_TYPE).format(i))
        fig.savefig(chunkPath, dpi=DPI, bbox_inches='tight', pad_inches=0)
    ax.clear()

#%% Script
exitFlag = 0

queueLock = threading.Lock()
workQueue = queue.Queue(100)
threads = []

# Create new threads
for threadID in range(1, NUM_THREADS+1):
   thread = myThread(threadID, workQueue)
   thread.start()
   threads.append(thread)

# Fill the queue
for audFile in audFiles:
    imgFile = audFile.parent/(audFile.stem+'.00'+IMAGE_TYPE)
    if REPLACE_EXISTING or imgFile not in imgFiles:
#        print(audFile.name +" -> "+ imgFile.name)
        while workQueue.full():
            time.sleep(1)
        queueLock.acquire()
        workQueue.put((audFile, imgFile))
        queueLock.release()
#            break # to only do the first file of the directory
    elif PRINT_SKIPS:
        print(imgFile.name + " already exists. Skipping.")

# Wait for queue to empty
while not workQueue.empty():
   time.sleep(1)

#%%
if CHECK_FOR_MISSING:
    audFiles = list(itertools.chain(Path("genres/genres/").glob('**/*'+AUDIO_TYPE),
                               Path("validation/rename/").glob('**/*'+AUDIO_TYPE)))
    imgFiles = list(itertools.chain(Path("genres/genres/").glob('**/*'+IMAGE_TYPE),
                               Path("validation/rename/").glob('**/*'+IMAGE_TYPE)))
    for audFile in audFiles:
        for i in range(NUM_IMAGE_SLICES):
            imgFile = audFile.parent/(audFile.stem+'.{:02}'.format(i)+IMAGE_TYPE)
            if imgFile not in imgFiles:
                print(str(imgFile)+" is missing.")

#%%

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
   t.join()
print ("Exiting Main Thread")
