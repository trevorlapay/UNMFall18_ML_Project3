import os
import random
import sys
from pathlib import Path

import librosa.core
import matplotlib
import matplotlib.pyplot as plt

PLOTS_PER_GENRE = 2

directories = [Path(tpl[0]) for tpl in os.walk(Path(sys.argv[1]))]
allFilesByDir = {directory: [directory / file
                             for file in os.listdir(directory)
                             if file[-3:].lower() == '.au']
                 for directory in directories}
allFilesByDir = {k: v for k, v in allFilesByDir.items() if len(v) != 0}

PLOTS_PER_GENRE = min([len(files) for directory, files in allFilesByDir.items()]
                      + [PLOTS_PER_GENRE])


matplotlib.rc('font', size=6)
fig, ax = plt.subplots(nrows=10, ncols=PLOTS_PER_GENRE,
                       sharex='col', sharey='row',
                       gridspec_kw={'top': 0.95,
                                    'bottom': 0.05,
                                    'left': 0.1,
                                    'right': 0.95,
                                    'hspace': 1,
                                    'wspace': 0.1},
                       figsize=(6.0, 8.0))

for subplot_row, (directory, files) in enumerate(allFilesByDir.items()):
    for subplot_col in range(PLOTS_PER_GENRE):
        rand = random.randrange(0, len(files) - 1)
        samples, sample_rate = librosa.core.load(files[rand])
        print(str(files[rand]))
        ax[subplot_row, subplot_col].specgram(samples, Fs=sample_rate, xextent=(0, 30), cmap='binary')
        ax[subplot_row, subplot_col].set_title((directory.name + " {}").format(rand))

# plt.show()
plt.savefig('Spectrogram Samples', dpi=300)
