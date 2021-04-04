import os
import math
from utils import *
from tqdm import tqdm


_file = f'[{os.path.basename(__file__)}]'


class NaiveBayesClassifier:
    '''
    label_cnts: {label: # examples with this label, }
    global_word_cnts: {word: its total occurence in training data, }

    '''
    def __init__(self, 
        label_cnts: dict, 
        global_word_cnts: dict,
        label_word_cnts: dict, 
        label_ip_cnts: dict, 
        label_time_cnts: dict,
        idf: dict,
        smooth_factor=0.1,
        num_features=128,
        use_time=False,
        use_ip=False,
        time_weight=1.0,
        ip_weight=1.0):

        # Params
        assert type(label_cnts) is dict
        self.label_cnts = label_cnts
        self.global_word_cnts = dict(global_word_cnts)
        self.label_word_cnts = dict(label_word_cnts)  # {label: {word: cnt, }, }
        self.label_ip_cnts = dict(label_ip_cnts)
        self.label_time_cnts = dict(label_time_cnts)
        self.labels = label_cnts.keys()
        for label in self.labels:
            self.label_word_cnts[label] = dict(self.label_word_cnts[label])
            self.label_time_cnts[label] = dict(self.label_time_cnts[label])
            self.label_ip_cnts[label] = dict(self.label_ip_cnts[label])
            
        self.idf = idf

        self.num_features = num_features
        self.smooth_factor = smooth_factor
        self.use_time = use_time
        self.use_ip = use_ip
        self.time_weight = time_weight
        self.ip_weight = ip_weight

        # self.use_ip = False
        # self.use_time = False
        
        # Reused values
        self.pre_computed = False
        self.vocab_size = len(global_word_cnts)
        self.num_examples = sum(label_cnts.values())
        self.num_words_in_label = {}
        self.num_time_in_label = {}
        self.num_ip_in_label = {}
        self.total_num_time = None
        self.total_num_ip = None

        # 贝叶斯公式中的先验知识
        self.logp_label = {}       # log(P(y))

    def pre_compute(self) -> None:
        """
        计算先验知识，如 log(P(y))，每个 label 下的词数，等
        """
        print(_file, f'Pre-computing... vocab size = {self.vocab_size}')

        self.total_num_time = 0
        self.total_num_ip = 0
        for label in self.labels:
            # self.logp_label_word[label] = {}

            label_cnt = self.label_cnts[label]
            self.logp_label[label] = math.log(label_cnt / self.num_examples)

            self.num_words_in_label[label] = sum(cnt for cnt in self.label_word_cnts[label].values())
            self.num_time_in_label[label] = sum(cnt for cnt in self.label_time_cnts[label].values())
            self.num_ip_in_label[label] = sum(cnt for cnt in self.label_ip_cnts[label].values())
            self.total_num_ip += self.num_ip_in_label[label]
            self.total_num_time += self.num_time_in_label[label]
        
        print(_file, "total number of ip:", self.total_num_ip)
        print(_file, 'total number of time:', self.total_num_time)
        print(_file, "# words in each class:", self.num_words_in_label)
        print(_file, 'P(y):', self.logp_label)
        self.pre_computed = True

    def get_idf(self, t) -> float:
        return get_or(t, self.idf, 0.0)

    def extract_features(self, terms: list) -> float:
        # for tf
        term_cnts = list_to_occ_dict(terms)
        total_terms = len(terms)
        def get_tf(t) -> float:
            return term_cnts[t] / total_terms

        def get_tfidf(t):
            return get_tf(t) * self.get_idf(t)

        terms = [(t, get_tfidf(t)) for t in terms]
        # 仅保留最高 TF-IDF 的 n 个词
        terms = sorted(terms, key=lambda x: x[1], reverse=True)
        terms = [t[0] for t in terms]
        return terms[:self.num_features]

    def classify(self, email) -> int:
        '''
        Return: label (int)
        '''
        if not self.pre_computed:
            self.pre_compute()
        
        words = email['words']
        words = self.extract_features(words)  # 只取部分词作为特征

        prob_label = {}
        for label in self.labels:
            prob_label[label] = self.logp_label[label]
        
        # for word in tqdm(words):
        for word in words:
            for label in self.labels:
                pwc = self.calc_logp_word_label(word, label)
                prob_label[label] += pwc
        
        if self.use_ip:
            ip = email['ip']
            if ip is not None:
                for label in self.labels:
                    prob_label[label] += self.ip_weight * self.calc_logp_ip_label(ip, label)

        if self.use_time:
            time = email['hour']
            if time is not None:
                for label in self.labels:
                    prob_label[label] += self.time_weight * self.calc_logp_time_label(time, label)

        predict = max(prob_label, key=lambda x: prob_label[x])
        return predict

    def calc_logp_word_label(self, word, label) -> float:
        ''' Return log P(w|C) '''
        # P(w | C) = # w in C / # words in C
        word_cnt = get_or(word, self.label_word_cnts[label], 0)
        nom = word_cnt + self.smooth_factor
        denom = self.num_words_in_label[label] + self.vocab_size * self.smooth_factor
        return math.log(nom / denom)

    def calc_logp_ip_label(self, ip: str, label) -> float:
        assert ip is not None
        ip_cnt = get_or(ip, self.label_ip_cnts, 0)
        nom = ip_cnt + self.smooth_factor
        denom = self.label_cnts[label] + self.total_num_ip * self.smooth_factor
        return math.log(nom / denom)

    def calc_logp_time_label(self, time: str, label) -> float:
        assert time is not None
        time_cnt = get_or(time, self.label_time_cnts, 0)
        nom = time_cnt + self.smooth_factor
        denom = self.label_cnts[label] + self.total_num_time * self.smooth_factor
        return math.log(nom / denom)
        


if __name__ == '__main__':
    pass
