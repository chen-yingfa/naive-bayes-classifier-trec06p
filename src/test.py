import os
import random
import argparse
from tqdm import tqdm
from config import *
from classifier import NaiveBayesClassifier
from utils import *
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


random.seed(SEED)


def calc_score(gold: list, predict: list):
    macro_f1 = f1_score(gold, predict, average='macro')
    micro_f1 = f1_score(gold, predict, average='micro')
    acc = accuracy_score(gold, predict)
    recall = recall_score(gold, predict)
    return {
        'acc': acc, 
        'micro_f1': micro_f1, 
        'macro_f1': macro_f1,
        'recall': recall}


def get_processed_data() -> tuple:
    '''
    Loads processed data
    '''
    dev_dataset = pickle_load(FILE_DEV_DATASET)
    global_word_cnts = pickle_load(FILE_GLOBAL_WORD_CNTS)
    word_cnts_0 = pickle_load(FILE_WORDS_0)
    word_cnts_1 = pickle_load(FILE_WORDS_1)
    label_word_cnts = {0: word_cnts_0, 1: word_cnts_1}
    label_cnts = pickle_load(FILE_LABEL_CNTS)
    label_ip_cnts = pickle_load(FILE_LABEL_IP_CNTS)
    label_time_cnts = pickle_load(FILE_LABEL_TIME_CNTS)
    idf = pickle_load(FILE_IDF)
    return (
        dev_dataset, 
        label_cnts,
        global_word_cnts, 
        label_word_cnts, 
        label_ip_cnts,
        label_time_cnts,
        idf)


def test_classifier(classifier, dataset):
    gold = [e['label'] for e in dataset]
    predict = []
    for email in tqdm(dataset):
        output = classifier.classify(email)
        predict.append(output)

    scores = calc_score(gold, predict)
    return scores


def test(args):
    '''
    Tests classifier using dev dataset
    '''
    print('Loading data...')
    processed_data = get_processed_data()
    dev_dataset = processed_data[0]
    label_cnts = processed_data[1]
    global_word_cnts = processed_data[2]
    label_word_cnts = processed_data[3]
    label_ip_cnts = processed_data[4]
    label_time_cnts = processed_data[5]
    idf = processed_data[6]

    print('Initializing classifier with parameters:')
    print('    Smoothing factor:', args.smooth)
    print('    Use IP:', args.use_ip)
    print('    Use time:', args.use_time)
    print('    IP weight:', args.ip_weight)
    print('    Time weight:', args.time_weight)
    
    classifier = NaiveBayesClassifier(
        label_cnts, 
        global_word_cnts, 
        label_word_cnts,
        label_ip_cnts,
        label_time_cnts,
        idf,
        smooth_factor=args.smooth,
        use_time=args.use_time,
        use_ip=args.use_ip,
        time_weight=args.time_weight,
        ip_weight=args.ip_weight,
        )
    print(f'--- Testing ---')
    print(f'# examples: {len(dev_dataset)}')
    print(f'---------------')
    scores = test_classifier(classifier, dev_dataset)
    acc = scores['acc']
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    recall = scores['recall']
    print(f'--- Result ---')
    print(f'accuracy = {acc*100:.3f}')
    print(f'macro_f1 = {macro_f1*100:.3f}')
    print(f'micro_f1 = {micro_f1*100:.3f}')
    print(f'recall = {recall*100:.3f}')
    print(f'--------------')
#     file_result = os.path.join(
#         DIR_RESULT, 
#         f'result_{args.smooth}_{1 if args.use_ip else 0}_{1 if args.use_time else 0}')
#     with open(file_result, 'w') as f:
#         f.write(f'''accuracy = {acc:.5f}
# macro_f1 = {macro_f1:.5f}
# micro_f1 = {micro_f1:.5f}
# recall = {recall:.5f}
# ''')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smooth',
        help='Laplace smoothing factor',
        type=float,
        default=1e-16,
    )
    parser.add_argument(
        '--use_ip',
        help='Whether to use IP address for classification',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_time',
        help='Whether to use time for classification',
        action='store_true',
        default=False,
    )    
    parser.add_argument(
        '--time_weight',
        help='Weight of time',
        type=float,
        default=0.0,
    )  
    parser.add_argument(
        '--ip_weight',
        help='Weight of IP addres',
        type=float,
        default=0.0,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(args) 