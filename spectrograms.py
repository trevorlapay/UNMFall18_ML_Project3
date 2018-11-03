import sys
import os
from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

directories = [Path(tpl[0]) for tpl in os.walk(Path(sys.argv[1]))]
for directory in directories:
    print(str(directory))
    for file in os.listdir(directory):
        if file[-4:] == '.wav':
            sample_rate, X = wavfile.read(directory/file)
            print("    "+file)
            specgram(X, Fs=sample_rate, xextent=(0,30))
            plt.show()
            break # Remove this to plot all the files, not just the first of each genre.
