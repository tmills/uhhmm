# Task worker
# Connects PULL socket to tcp://localhost:5557
# Collects workloads from ventilator via that socket
# Connects PUSH socket to tcp://localhost:5558
# Sends results to sink via that socket
#
# Author: Lev Givon <lev(at)columbia(dot)edu>
# Modified by Bill Bryce <bryce2(at)illinois(dot)edu>

import sys
import time
import zmq


context = zmq.Context()

# Socket to receive messages on
receiver = context.socket(zmq.PULL)
receiver.connect("tcp://qenu.linguistics.illinois.edu:5557")

# Socket to send messages to
sender = context.socket(zmq.PUSH)
sender.connect("tcp://qenu.linguistics.illinois.edu:5558")

# Process tasks forever
while True:
    dummy_list = receiver.recv_pyobj()

    # Simple progress indicator for the viewer
#    sys.stdout.write('(1)')
#    sys.stdout.flush()

    # Do the work
    time.sleep(len(dummy_list)*0.001)
    dummy_list.append('(V1)')

    # Send results to sink
    sender.send_pyobj(dummy_list)
