"""Converts .au to .wav file using the sox tool.
IN: Path to a directory containing au files and/or directories containing au files.
OUT: wav files located in the same directory as their original au version.
Run instructions:
python auToWav.py path
Where path contains au files to be converted and/or directoreis containing au files to be converted
"""

import sys
import os
from pathlib import Path
import subprocess

REPLACE_EXISTING = False
PRINT_SKIPS = True

pathstr = sys.argv[1] if len(sys.argv) > 1 is not None else "genres/genres/"

directories = [Path(tpl[0]) for tpl in os.walk(Path(pathstr))]
for directory in directories:
    print(str(directory))
    files = os.listdir(directory)
    for file in [f for f in files if f[-3:] == '.au']:
        if REPLACE_EXISTING or (file[:-3]+".wav") not in files:
            print(' '*(len(str(directory.parent))+1)+file)
            subprocess.run(["sox", str(directory/file), str(directory/(file[:-3]+".wav"))])
#            break
        elif PRINT_SKIPS:
            print(' '*(len(str(directory.parent))+1)+(file[:-3]+".wav")+" already exists. Skipping.")
#    if len([f for f in files if f[-3:] == '.au']) > 0:
#        break
