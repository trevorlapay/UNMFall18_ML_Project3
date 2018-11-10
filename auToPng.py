
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

AUDIO_TYPE = ".au"
IMAGE_TYPE = ".png"
NUM_IMAGE_SLICES = 10
PLOT_HIGHT = 1 # inches
PLOT_WIDTH = 30/NUM_IMAGE_SLICES # 1 inch per second
DPI = 125 # dots per inch (1D so don't square it.)
REPLACE_EXISTING = False
PRINT_SKIPS = True
NUM_THREADS = 4

pathstr = sys.argv[1] if len(sys.argv) > 1 is not None else "genres/genres/"

exitFlag = 0


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
    sampleChunks = [samples[i:i + chunkSize] 
                    for i in range(0, len(samples), chunkSize)
                    if len(samples[i:i + chunkSize]) == chunkSize]
    for i, chunk in enumerate(sampleChunks):
        ax.specgram(chunk, Fs=sample_rate, cmap='jet')
        chunkPath = imgPath.with_name((imgPath.stem[:-2]+'{:02}'+IMAGE_TYPE).format(i))
        fig.savefig(chunkPath, dpi=DPI, bbox_inches='tight', pad_inches=0)
    ax.clear()

queueLock = threading.Lock()
workQueue = queue.Queue(1000)
threads = []

# Create new threads
for threadID in range(1, NUM_THREADS+1):
   thread = myThread(threadID, workQueue)
   thread.start()
   threads.append(thread)

# Fill the queue
directories = [Path(tpl[0]) for tpl in os.walk(Path(pathstr))]
for directory in directories:
    print(str(directory))
    indent = ' '*(len(str(directory.parent))+1)
    files = os.listdir(directory)
    for audFile in [f for f in files if f[-len(AUDIO_TYPE):] == AUDIO_TYPE]:
        imgFile = audFile[:-len(AUDIO_TYPE)]+'.00'+IMAGE_TYPE
        if REPLACE_EXISTING or imgFile not in files:
            print(indent + audFile +" -> "+ imgFile)
            while workQueue.full():
                time.sleep(1)
            queueLock.acquire()
            workQueue.put((directory/audFile, directory/imgFile))
            queueLock.release()
#            break # to only do the first file of the directory
        elif PRINT_SKIPS:
            print(indent + imgFile + " already exists. Skipping.")
    
    # break to only do the first directory
#    if len([f for f in files if f[-3:] == AUDIO_TYPE]) > 0:
#        break

# Wait for queue to empty
while not workQueue.empty():
   time.sleep(1)

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
   t.join()
print ("Exiting Main Thread")
