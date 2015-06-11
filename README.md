UHHMM - Unsupervised Hierarchical Hidden Markov Model

Usage:
python3 scripts/d1trainer.py <config file>
OR
make


There is a sample config file in config/d1train.ini. If you run using make, the first time it runs it will copy that file into myconfig.ini. You can then modify that config file and subsequent runs of make will use it.

The config has an 'io' (input/output) section and a 'params' (machine learning parameters) section. 'io' requires an input file, output directory, and an optional dictionary file.
The input file should contain one sentence per line, with space-separated ints representing tokens.
The output directory is where all output will be stored. This is so multiple runs will be preserved if desired. The first thing the d1trainer.py script will do is copy the config file into the output directory, to promote reproducibility.
The dictionary file should contain a mapping between words and their index. Each line should be <word> <index>. While it is technically optional, it is borderline required for understanding output.

There are scripts in the scripts/ sub-directory to convert token files into int files, creating a dictionary as well.

