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

directories = [Path(tpl[0]) for tpl in os.walk(Path(sys.argv[1]))]
for directory in directories:
    print(str(directory))
    for file in os.listdir(directory):
        if file[-3:] == '.au':
            print("    "+file)
            subprocess.run(["sox", str(directory/file), str(directory/(file[:-3]+".wav"))])
