from collections import defaultdict
import pickle
import sys
from .utils import get_process_memory
from .utils import Sentences

class FeatureManager:
    
    def __init__(self, templates=None, feature_begin=-2, feature_end=2):
        self.begin = feature_begin
        self.end = feature_end
        self.templates = templates if templates else self._generate_token_templates()
        self.vocab_to_idx = {}
        self.idx_to_vocab = []
        self.counter = {}

    def _generate_token_templates(self):
        templates = []
        for b in range(self.begin, self.end):
            for e in range(b, self.end+1):
                if (b == 0) or (e == 0):
                    continue
                templates.append((b, e))
        return templates

    def words_to_feature(self, words):
        x =[]
        for i in range(len(words)):
            xi = []
            e_max = len(words)
            for t in self.templates:
                b = i + t[0]
                e = i + t[1] + 1
                if b < 0 or e > e_max:
                    continue
                if b == (e - 1):
                    xi.append(('X[%d]' % t[0], words[b]))
                else:
                    contexts = words[b:e] if t[0] * t[1] > 0 else words[b:i] + words[i+1:e]        
                    xi.append(('X[%d,%d]' % (t[0], t[1]), tuple(contexts)))
            x.append(xi)
        return x

    def words_to_encoded_feature(self, words):
        x = self.words_to_feature(words)
        z = [[self.vocab_to_idx[f] for f in xi if f in self.vocab_to_idx] for xi in x]
        return z

    def scanning_features(self, corpus_fname, pruning_n_sents=1000000, pruning_min_count=5, min_count=50):
        counter = defaultdict(lambda: 0)
        with open(corpus_fname, encoding='utf-8') as f:
            for num_sent, sent in enumerate(f):
                words = sent.strip().split()
                if not words:
                    continue
                    
                x = self.words_to_feature(words)
                for word, xi in zip(sent, x):
                    for feature in xi:
                        counter[feature] += 1

                if (num_sent + 1) % pruning_n_sents == 0:
                    before_size = len(counter)
                    before_memory = get_process_memory()
                    counter = defaultdict(lambda: 0, {f:v for f,v in counter.items() if v >= pruning_min_count})
                    args = (before_size, len(counter), num_sent, before_memory, get_process_memory())
                    sys.stdout.write('\r# features = %d -> %d, (%d sents) %.3f -> %.3f Gb' % args)

                if num_sent % 1000 == 0:
                    args = (len(counter), num_sent, get_process_memory())
                    sys.stdout.write('\r# features = %d, (%d sents) %.3f Gb' % args)

        counter = {f:v for f,v in counter.items() if v >= min_count}
        self.idx_to_vocab = list(sorted(counter.keys(), key=lambda x:counter.get(x, 0), reverse=True))
        self.vocab_to_idx = {vocab:idx for idx, vocab in enumerate(self.idx_to_vocab)}
        self.counter = counter
    
    def transform_rawtext_to_zcorpus(self, rawtext_fname, zcorpus_fname):
        if not vocab_to_idx:
            raise ValueError('You should scan vocabs first')
        with open(rawtext_fname, encoding='utf-8') as fi:
            with open(zcorpus_fname, 'w', encoding='utf-8') as fo:
                for num_sent, sent in enumerate(fi):
                    words = sent.strip().split()
                    if not words:
                        fo.write('\n')
                        continue
                    z = self.words_to_encoded_feature(words)
                    for wi, zi in zip(words, z):
                        features = ' '.join([str(zi_) for zi_ in zi]) if zi else ''
                        fo.write('%s\t%s\n' % (wi, features))
                    if num_sent % 50000 == 0:
                        sys.stdout.write('\rtransforming .... (%d sents) %.3f Gb' % (
                                num_sent, get_process_memory()))
                print('\rtransforming has done')
            
    def save(self, fname):        
        with open(fname, 'wb') as f:
            parameters = {
                'feature_begin': self.begin,
                'feature_end': self.end,
                'templates': self.templates,
                'idx_to_vocab': self.idx_to_vocab,
                'vocab_to_idx': self.vocab_to_idx, 
                'counter': self.counter
            }
            pickle.dump(parameters, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            parameters = pickle.load(f)
        self.begin = parameters['feature_begin']
        self.end = parameters['feature_end']
        self.templates = parameters['templates']
        self.idx_to_vocab = parameters['idx_to_vocab']
        self.vocab_to_idx = parameters['vocab_to_idx']
        self.counter = parameters['counter']
    
    
class ZCorpus:
    def __init__(self, fname):
        self.fname = fname
        self.length = 0
        
    def __len__(self):
        if self.length == 0:
            with open(self.fname, encoding='utf-8') as f:
                for num_row, _ in enumerate(f):
                    continue
                self.length = (num_row + 1)
        return self.length
    
    def __iter__(self):
        with open(self.fname, encoding='utf-8') as f:
            for row in f:
                row = row.strip()
                if ('\t' in row) == False: continue
                word, features = row.split('\t')
                features = features.split()
                yield word, features