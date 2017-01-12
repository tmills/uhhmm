#!/usr/bin/env python3

import logging
from multiprocessing import Process
import multiprocessing
import os, os.path
import signal
import subprocess
import sys
import time
import PyzmqWorker

def start_cluster_workers(work_distributer, cluster_cmd, maxLen, gpu):
    logging.debug("Cluster command is %s" % cluster_cmd)

    cmd_str = 'python3 %s/scripts/workers.py %s %d %d %d %d %s' % (os.getcwd(), work_distributer.host, work_distributer.jobs_port, work_distributer.results_port, work_distributer.models_port, maxLen, gpu)
    submit_cmd = [ cmd_arg.replace("%c", cmd_str) for cmd_arg in cluster_cmd.split()]
    logging.info("Making cluster submit call with the following command: %s" % str(submit_cmd))
    subprocess.call(submit_cmd)
    
def start_local_workers_with_distributer(work_distributer, maxLen, num_workers, gpu, batch_size=10):
    logging.info("Starting workers with maxLen=%d and num_workers=%d" % (maxLen, num_workers) )
    return start_local_workers(work_distributer.host, work_distributer.jobs_port, work_distributer.results_port, work_distributer.models_port, maxLen, num_workers, gpu, batch_size)
    
def start_local_workers(host, jobs_port, results_port, models_port, maxLen, num_workers, gpu, batch_size=10):
    logging.info("Starting %d workers at host %s with jobs_port=%d, results_port=%d, models_port=%d, maxLen=%d" % (num_workers, host, jobs_port, results_port, models_port, maxLen) )
    multiprocessing.set_start_method('spawn')
    processes = []
    logging.info("Worker intializing GPU status: %s" % gpu)
    for i in range(0, num_workers):
        if i > 0:   
            gpu = False
            batch_size = 1
        fs = PyzmqWorker.PyzmqWorker(host, jobs_port, results_port, models_port, maxLen, tid=i, gpu=gpu, batch_size=batch_size, level=logging.getLogger().getEffectiveLevel())
        signal.signal(signal.SIGTERM, fs.handle_sigterm)
        signal.signal(signal.SIGINT, fs.handle_sigint)
        signal.signal(signal.SIGALRM, fs.handle_sigalarm)
        p = Process(target=fs.run)
        processes.append(p)
        p.start()
    
    return processes

def main(args):
    logging.basicConfig(level=logging.INFO)
    
    if len(args) != 1 and len(args) != 6 and len(args) != 7:
        print("ERROR: Wrong number of arguments! Two run modes -- One argument of a file with properties or 6-8 arguments with properties.")
        sys.exit(-1)
        
    if len(args) == 1:
        config_file = args[0] + "/masterConfig.txt"
        while True:
            if os.path.isfile(config_file):
                configs = open(config_file).readlines()
                if len(configs)==2 and 'OK' in configs[1]:
                    logging.info('OSC setup acquired. Starting a worker with ' + config_file)
                    args = configs[0].strip().split(' ')
                    break
            else:
                time.sleep(10)

    num_workers = 1
    if len(args) >= 7:
        num_workers = int(args[6])
    
        processes = start_local_workers(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]), num_workers, bool(args[5]))
    
#         for i in range(0, num_workers):
#             fs = PyzmqWorker.PyzmqWorker(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]), tid=i)
#             signal.signal(signal.SIGINT, fs.handle_sigint)
#             signal.signal(signal.SIGALRM, fs.handle_sigalarm)
#             p = Process(target=fs.run)
#             processes.append(p)
#             ## Call run directly instead of start otherwise we'll have 2n workers    
#             p.start()
    
        for i in range(0, num_workers):
            processes[i].join()

    else:
        start_local_workers(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]), 1, bool(args[5]))

if __name__ == "__main__":
    main(sys.argv[1:])

