#!/bin/bash
./gpusrc/compile_osc.sh
python3 setup.py build_ext || exit
if [ $# -ne 1 ]; 
    then echo "Running the script requires a config argument; exiting."
    exit
fi
qsub gpusrc/master.scr
qsub gpusrc/worker.scr

