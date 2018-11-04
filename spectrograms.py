import sys
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import random
import librosa.core

PLOTS_PER_GENRE = 5

directories = [Path(tpl[0]) for tpl in os.walk(Path(sys.argv[1]))]
allFilesByDir = {folder: [folder/file for file in os.listdir(folder) if file[-3:] == '.au'] for folder in directories}

matplotlib.rc('font', size=6)

fig, ax = plt.subplots(nrows=10, ncols=PLOTS_PER_GENRE, sharex='col', sharey='row',
                       gridspec_kw={'top': 0.95,
                                    'bottom': 0.05,
                                    'left': 0.1,
                                    'right': 0.95,
                                    'hspace': 1,
                                    'wspace': 0.1},
                       figsize=(8.0, 5.0))

for iAxCol in range(PLOTS_PER_GENRE):
    iAxRow = 0
    for folder in directories:
        if iAxCol >= len(allFilesByDir[folder]):
            continue
        iFile = random.randrange(0, len(allFilesByDir[folder]) - 1)
        X, sample_rate = librosa.core.load(allFilesByDir[folder][iFile])
        print(str(allFilesByDir[folder][iFile]))
        ax[iAxRow, iAxCol].specgram(X, Fs=sample_rate, xextent=(0, 30), cmap='jet')
        ax[iAxRow, iAxCol].set_title((folder.name + " {}").format(iFile))
        iAxRow += 1

# plt.show()
plt.savefig('Spectrograms', dpi=300)
