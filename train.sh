#!/bin/bash
./gpusrc/compile_linux.sh
#./gpusrc/compile.sh
#./gpusrc/compile_osc.sh
python3 setup.py build_ext || exit
if [ $# -ne 1 ]; 
    then echo "Running the script requires a config argument; exiting."
    exit
fi
python3 scripts/uhhmm-trainer.py $*

