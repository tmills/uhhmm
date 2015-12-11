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
This software has been written to be highly parallelizable, either on a single node with multiple cores or in a cluster environment with multiple nodes. We use Pyzmq inter-process messaging to pass sentences to workers and receive parses from workers. The main job collects the sampled sentences, recomputes UHHMM model parameters, and then iterates.

### Cluster submission scripts:
To run the code on a cluster, there is an additional parameter that must be included in the config file (cluster_cmd), which tells the software how to submit job arrays for its workers. This command needs to have a parameter %%c where the executable name goes which our code will replace with the call to start the workers (it needs two % because the .ini script will remove the first one). We have tested this code on two different cluster platforms -- LSF and SLURM -- see the examples below.

#### LSF:
cluster_cmd=bsub -e %%J.err -J workers[1-70] -q <queue name> %%c

This command writes its output to a file that uses the job id (%%J), creates a job array with 70 workers (workers[1-70]), and is submitted to the job queue you give it. Note that the num_proces=70 argument is still required.

#### SLURM:
cluster_cmd=sbatch --array=1-30 --mem-per-cpu 3000 -p <partition name> -t 1-00 --wrap=%%c

This command uses 30 workers, requests 3 GB of RAM per worker, is submitted to the partition given (similar to the queues above) and will be killed after one day. 

