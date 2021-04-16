import re
import os
import pickle
import argparse
import random
import math
from tqdm import tqdm
from config import *
from utils import *
from dataloader import DataLoader


_file = '[' + os.path.basename(__file__) + ']'


random.seed(SEED)


def get_words(dataset):
    '''
    Count occurence of each word
    Return (a, b, c)
    a: {word_i: occurence of word_i in entire dataset, ...}
    b: {word_i: occurence of word_i in examples with label=0, }
    c: {word_i: occurence of word_i in examples with label=1, }
    '''
    print(_file, 'Counting words occurences...')
    words_all = {}
    words_0 = {}
    words_1 = {}

    for email in tqdm(dataset):
        label = email['label']
        
        for word in email['words']:
            incr(words_all, word)
            
            if label == 0:
                incr(words_0, word)
            elif label == 1:
                incr(words_1, word)
            else:
                print(f'[{_file}] Invalid label: {label}')
                print(f'[{_file}] Should be 0 or 1')
                exit(0)

    words_all = sorted(words_all.items(), key=lambda x:x[1], reverse=True)
    words_0 = sorted(words_0.items(), key=lambda x:x[1], reverse=True)
    words_1 = sorted(words_1.items(), key=lambda x:x[1], reverse=True)
    return words_all, words_0, words_1


def get_idf(dataset):
    '''
    Pre-compute IDF of each word
    Return: {word: idf(word), }
    '''
    print(_file, 'Computing IDF...')
    cnt = {}  # {t: 含 t 文数)}
    for email in dataset:
        for word in set(email['words']):
            incr(cnt, word, 1)
    idf = {}  # {t: idf(t)}
    D = len(dataset)
    for t in cnt:
        idf[t] = math.log(D / cnt[t])
    return idf


def get_label_cnts(dataset: list) -> dict:
    label_cnts = {}
    for email in dataset:
        incr(label_cnts, email['label'], 1)
    return label_cnts


def get_label_time_cnts(dataset: list) -> dict:
    label_time_cnts = {0: {}, 1: {}}
    for email in dataset:
        label = email['label']
        if email['hour'] is not None:
            incr(label_time_cnts[label], email['hour'], 1)
    return label_time_cnts


def get_label_ip_cnts(dataset: list) -> dict:
    label_ip_cnts = {0: {}, 1: {}}
    for email in dataset:
        label = email['label']
        if email['ip'] is not None:
            incr(label_ip_cnts[label], email['ip'], 1)
    return label_ip_cnts


def process_dev():
    '''
    Process dev dataset, this needs to be run only once
    '''
    loader = DataLoader(FILE_INDEX_DEV)
    pickle_save(loader.data, FILE_DEV_DATASET)
    return loader.data


def preprocess(args):
    dev_dataset = process_dev()
    print(f'Loading data with data_size = {100 * args.data_size}%')
    train_loader = DataLoader(FILE_INDEX_TRAIN, args.data_size)
    train_dataset = train_loader.data

    print(_file, f'# Train ex.: {len(train_dataset)}')

    words_all, words_0, words_1 = get_words(train_dataset)
    idf = get_idf(train_dataset)

    label_cnts = get_label_cnts(train_dataset)
    label_ip_cnts = get_label_ip_cnts(train_dataset)
    label_time_cnts = get_label_time_cnts(train_dataset)
    print(_file, 'Label counts:', label_cnts)
    print(_file, 'Vocab size:', len(words_all))
    # Save
    print(_file, 'Saving pre-processed data to', DIR_PROCESSED)
    if not os.path.exists(DIR_PROCESSED):
        os.makedirs(DIR_PROCESSED)
    pickle_save(train_dataset, FILE_TRAIN_DATASET)
    pickle_save(words_all, FILE_GLOBAL_WORD_CNTS)
    pickle_save(words_0, FILE_WORDS_0)
    pickle_save(words_1, FILE_WORDS_1)
    pickle_save(label_cnts, FILE_LABEL_CNTS)
    pickle_save(label_ip_cnts, FILE_LABEL_IP_CNTS)
    pickle_save(label_time_cnts, FILE_LABEL_TIME_CNTS)
    pickle_save(idf, FILE_IDF)


def gen_index_file(train_file, dev_file, k_fold, shuffle=True):
    lines = []
    with open(FILE_INDEX_FULL) as f:
        lines = f.readlines()
    if shuffle:
        random.shuffle(lines)
    sep = len(lines) // k_fold
    train_lines = lines[sep:]
    dev_lines = lines[:sep]
    for i in range(2):
        with open([train_file, dev_file][i], 'w') as f:
            for line in [train_lines, dev_lines][i]:
                f.write(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_size',
        help='proportion of the dataset to be used',
        type=float,
        default=1.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # gen_index_file(FILE_INDEX_TRAIN, FILE_INDEX_DEV, 5)
    preprocess(args)