UHHMM - Unsupervised Hierarchical Hidden Markov Model

Usage:
python3 scripts/d1trainer.py <config file>
OR
make


There is a sample config file in config/d1train.ini. If you run using make, the first time it runs it will copy that file into myconfig.ini. You can then modify that config file and subsequent runs of make will use it.

The config has an 'io' (input/output) section and a 'params' (machine learning parameters) section. 'io' requires an input file, output directory, a dictionary file, and a working directory (filesystem location) that both the workers and the job distributer have access to.

The input file should contain one sentence per line, with space-separated ints representing tokens (if evaluating for POS tagging, the system accepts space-separated tag/word tokens.

The output directory is where all output will be stored. This is so multiple runs will be preserved if desired. The first thing the d1trainer.py script will do is copy the config file into the output directory, to promote reproducibility.
The dictionary file should contain a mapping between words and their index. Each line should be &lt;word&gt; &lt;index&gt;.

There are scripts in the scripts/ sub-directory to convert token files into int files, creating a dictionary as well.

## Resuming jobs
If a run terminates early or needs to be stopped, they can often be resumed. After every iteration the model files will be pickled into a file named sample.obj. To resume a job rather than running a new one, simply run:
python3 scripts/d1trainer.py <output dir>

where <output dir> is the directory that the run was writing its output to. The system will read in the sample.obj file as well as the config file that was written to that directory to resume inference. This means you can change parameters if you edit that config file. This can be useful (for increasing the number of workers) but could also be abused in ways that would probably break the system (e.g., changing the number of POS tags in the finite system).

## Cluster usage
This software has been written to be highly parallelizable, either on a single node with multiple cores or in a cluster environment with multiple nodes. We use Pyzmq inter-process messaging to pass sentences to workers and receive parses from workers. The main job collects the sampled sentences, recomputes UHHMM model parameters, and then iterates. For cluster environments where there is no shared disk space in the standard places (/tmp), please make sure you have a environment variable TMP, TEMP, or TMPDIR set  -- pythons temp file libraries will use these.
