import torch
from ViterbiParser import ViterbiParser
from torch.multiprocessing import Process

class worker:
    def __init__(self):
        self.a = torch.zeros(3,4,5)

    def run(self):
        self.a = self.a.cuda()
        self.a.fill_(0)
        self.parser = ViterbiParser()
        print(self.a)
        self.parser.set_models(0)

for i in range(4):
    w = worker()
    p = Process(target=w.run)
    p.start()
# a = torch.zeros(4, 1372, 32)
# b = torch.zeros(4, 1372, 32)
# a = a.cuda()
# c = ViterbiParser()
# c.set_models(a)
# a.fill_(0)