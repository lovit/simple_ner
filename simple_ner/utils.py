import datetime
import os
import time
import psutil

def remain_time(begin_time, i, n, initialize_time=0):
    if i == 0: i = 1
    process_time = (time.time() - begin_time)
    required_time = (process_time / i) * (n - i)
    return '[ -%s, +%s ]' % (datetime_format(process_time + initialize_time), datetime_format(required_time))

def datetime_format(t):
    return datetime.timedelta(seconds=int(t))

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

def write_list(fname, l):
    with open(fname, 'w', encoding='utf-8') as f:
        for doc in l:
            f.write('%s\n' % str(doc))

class Sentences:
    def __init__(self, fnames):
        if type(fnames) == str:
            fnames = [fnames]        
        self.fnames = fnames
        self.length = 0
        
    def __iter__(self):
        for fname in self.fnames:
            if not os.path.exists(fname):
                print('%s does not exist' % fname)
                continue
            with open(fname, encoding='utf-8') as f:
                for sent in f:
                    yield sent.split()
                    
    def __len__(self):
        if self.length == 0:
            for fname in self.fnames:
                if not os.path.exists(fname):
                    continue
                with open(fname, encoding='utf-8') as f:
                    for i, _ in enumerate(f):
                        continue
                    self.length += (i+1)
        return self.length