import os
import time
import psutil

def remain_time(begin_time, i, n):
    process_time = (time.time() - begin_time)
    required_time = (process_time / (i+1)) * (n - i)
    if required_time > 3600:
        return '%.3f hours' % (required_time / 3600)
    if required_time > 60:
        return '%.3f mins' % (required_time / 60)
    return '%.3f secs' % required_time

def get_available_memory():
    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def load_sentences(fname):
    with open(fname, encoding='utf-8') as f:
        sents = [sent.split() for sent in f]
    return sents

class Sentences:
    def __init__(self, fname):
        self.fname = fname
        self.length = 0
    def __iter__(self):
        with open(self.fname, encoding='utf-8') as f:
            for sent in f:
                yield sent.split()
    def __len__(self):
        if self.length == 0:
            with open(self.fname, encoding='utf-8') as f:
                for i, _ in enumerate(f):
                    continue
                self.length = (i+1)
        return self.length