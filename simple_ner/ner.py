from collections import defaultdict
from collections import namedtuple
import pickle
import sys
import time
import numpy as np
from .utils import remain_time
from .utils import get_process_memory

class FeatureCountingNER:
    
    def __init__(self, feature_manager=None):
        self.feature_manager = feature_manager
        self.coefficient = {}
        self.coefficient_ = {}
        
        self._usg_pos = None
        self._usg_neg = None
        self._usg_features = None        
        self.positive_features= None
    
    def fit_and_extract_ner(self, zcorpus, ner_seed, wordset, min_count_positive_features=10):
        _ = self.find_positive_features(zcorpus,ner_seed, min_count_positive_features=10)
        self.compute_score_of_features(zcorpus, ner_seed, wordset)        
        return extract_named_entities_from_zcorpus(zcorpus)
        
    def find_positive_features(self, zcorpus, ner_seeds, min_count_positive_features=10):
        self.positive_features = {}
        
        # TODO: file 넣는걸로 바꾸기
        for num_z, (word, features) in enumerate(zcorpus):    
            if num_z % 1000 == 0:
                sys.stdout.write('\r# scanning positive features # = %d, (%.3f %s, %d in %d) %.3f Gb' 
                                 % (len(self.positive_features), (100 * num_z / len(zcorpus)), '%', 
                                    num_z, len(zcorpus), get_process_memory()))

            if (word in ner_seeds) == False:
                continue        
            for feature in features:
                self.positive_features[feature] = self.positive_features.get(feature, 0) + 1

        self.positive_features = {pos_f:v for pos_f, v in self.positive_features.items() if v > min_count_positive_features}
        print('\rscanning positive features was done')
        return self.positive_features
    
    def compute_score_of_features(self, zcorpus, ner_seeds, wordset):
        def proportion_has_seeds(word_dict, ner_seeds):
            n_neg = 0
            n_pos = 0
            for word, freq in word_dict.items():
                if word in ner_seeds:
                    n_pos += freq
                else:
                    n_neg += freq
            return n_pos / (n_pos + n_neg)
        # TODO: corpus -> filename 넣는걸로
        usg_features, usg_pos, usg_neg = self._scan_usage_of_features(zcorpus, ner_seeds, wordset)
        
        begin_time = time.time()
        n = len(usg_pos)

        proportion_positive_features = {}
        for i, (feature, n_positive) in enumerate(usg_pos.items()):
            # Other functions
        #     proportion_positive_features[feature] = proportion_over_word2vec_sim(usage_of_features[feature], min_similarity=0.5)
            proportion_positive_features[feature] = proportion_has_seeds(usg_features[feature], ner_seeds)
            sys.stdout.write('\r(%d in %d) remained %s' % (i+1, n, remain_time(begin_time, i, n)))
        print('\rcomputing score of features was done')
        self.coefficient = proportion_positive_features
        self.coefficient_ = {self.feature_manager.idx_to_vocab[int(f)]:s for f,s in proportion_positive_features.items() }
        self._usg_pos = dict(usg_pos)
        self._usg_neg = dict(usg_neg)
        self._usg_features = {feature:{w:f for w,f in wd.items()} for feature, wd in usg_features.items()}
    
    def _scan_usage_of_features(self, zcorpus, ner_seeds, wordset):
        usage_of_features = defaultdict(lambda: defaultdict(lambda: 0))
        usage_of_positive_features = defaultdict(lambda: 0)

        for num_z, (word, features) in enumerate(zcorpus):
            if num_z % 1000 == 0:
                sys.stdout.write('\r# scanning usage of positive features (%.3f %s, %d in %d) %.3f Gb' 
                                 % ((100 * num_z / len(zcorpus)), '%', 
                                    num_z, len(zcorpus), get_process_memory()))

            if (word in wordset) == False:
                continue
            for feature in features:
                if feature in self.positive_features:
                    usage_of_features[feature][word] += 1
                    if word in ner_seeds:
                        usage_of_positive_features[feature] += 1

        usage_of_negative_features = {feature:(sum(v.values()) - usage_of_positive_features.get(feature, 0)) 
                              for feature, v in usage_of_features.items()}
                
        print('\rscanning usage of positive features was done')
        return usage_of_features, usage_of_positive_features, usage_of_negative_features
        
    def get_coefficient_histogram(self, n_bins=20):        
        heights, centroids = np.histogram(self.coefficient.values(), bins=n_bins)
        for h, c1, c2 in zip(heights, centroids, centroids[1:]):
            print('%.2f ~ %.2f: %.3f' % (c1, c2, h/sum(heights)))
            
    def extract_named_entities_from_zcorpus(self, zcorpus):
        prediction_score = defaultdict(lambda: 0.0)
        prediction_count = defaultdict(lambda: 0.0)

        for num_z, (word, features) in enumerate(zcorpus):
            if num_z % 1000 == 0:
                sys.stdout.write('\r(%d in %d) %.3f %s' % (num_z, len(zcorpus), 100 * num_z / len(zcorpus), '%'))
            if not features: continue
            for feature in features:
                if (feature in self.coefficient) == False:
                    continue
                prediction_score[word] += self.coefficient[feature]
                prediction_count[word] += 1

        prediction_normed_score = {word:score/prediction_count[word] for word, score in prediction_score.items()}
        sorted_scores = sorted(prediction_normed_score.items(), key=lambda x:x[1], reverse=True)
        return sorted_scores
    
    def infer_named_entity_score(self, encoded_features):
        score = 0
        norm = 0
        for f in encoded_features:
            if (f in self.coefficient) == False:
                continue
            score += self.coefficient[f]
            norm += 1
        return (score / norm) if norm > 0 else 0
    
    def save(self, fname):
        with open(fname, 'wb') as f:
            parameters = {
                'usage_of_positive_features': self._usg_pos,
                'usage_of_negative_features': self._usg_neg,
                'usage_of_features': self._usg_features,
                'coefficient': self.coefficient,
                'coefficient_': self.coefficient_,
                
                'feature_manager.feature_begin': self.feature_manager.begin,
                'feature_manager.feature_end': self.feature_manager.end,
                'feature_manager.templates': self.feature_manager.templates,
                'feature_manager.idx_to_vocab': self.feature_manager.idx_to_vocab,
                'feature_manager.vocab_to_idx': self.feature_manager.vocab_to_idx, 
                'feature_manager.counter': self.feature_manager.counter
            }
            pickle.dump(parameters, f)
        
    def load(self, fname):
        with open(fname, 'rb') as f:
            parameters = pickle.load(f)
            self._usg_pos = parameters['usage_of_positive_features']
            self._usg_neg = parameters['usage_of_negative_features']
            self._usg_features = parameters['usage_of_features']
            self.coefficient = parameters['coefficient']
            self.coefficient_ = parameters['coefficient_']
            
            self.feature_manager = FeatureManager()
            self.feature_manager.begin = parameters['feature_manager.feature_begin']
            self.feature_manager.end = parameters['feature_manager.feature_end']
            self.feature_manager.templates = parameters['feature_manager.templates']
            self.feature_manager.idx_to_vocab = parameters['feature_manager.idx_to_vocab']
            self.feature_manager.vocab_to_idx = parameters['feature_manager.vocab_to_idx']
            self.feature_manager.counter = parameters['feature_manager.counter']


class TrainedLogisticRegressionExtractorFromZcorpus:
    def __init__(self, coefficients):
        self._coef = coefficients
        
    def extract(self, zcorpus):        
        Score = namedtuple('Score', 'score frequency')        
        scores = {}
        _norm = {}
        frequency = {}
        for i, (word, features) in enumerate(zcorpus):
            frequency[word] = frequency.get(word, 0) + 1
            _norm[word] = _norm.get(word, 0) + len(features)
            for feature in features:
                scores[word] = scores.get(word, 0) + self._coef[int(feature)]
            if i % 1000 == 0:
                args = (100*(i+1)/len(zcorpus), '%', i+1, len(zcorpus))
                sys.stdout.write('\rextracting ... %.3f %s (%d in %d)' % args)
        print('\rextracting was done')
        scores = {word:Score(score/_norm[word], frequency[word]) for word, score in scores.items()}
        scores = sorted(scores.items(), key=lambda x:x[1].score, reverse=True)
        return scores
    
    def save(self, fname):
        with open(fname, 'wb') as f:
            params = {'coefficient': self._coef}
            pickle.dump(params, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            self._coef = pickle.load(f)