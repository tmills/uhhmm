#!/bin/bash

python3 setup.py build_ext
if [ $# -ne 1 ]; 
    then echo "Running the script requires a config argument; exiting."
    exit
fi
python3 -m pdb scripts/uhhmm-trainer.py $*

