import os.path as path

DATA_NAME = 'trec06p'
DIR_DATA      = path.join('..', 'data', DATA_NAME)
DIR_PROCESSED = path.join('..', 'data', 'processed')
DIR_RESULT    = path.join('..', 'result')

FILE_INDEX_FULL  = path.join(DIR_DATA, 'index')
FILE_INDEX_TRAIN = path.join(DIR_DATA, 'train_index.txt')
FILE_INDEX_DEV   = path.join(DIR_DATA, 'dev_index.txt')

# Processed data
FILE_TRAIN_DATASET      = path.join(DIR_PROCESSED, 'train_dataset.pkl')
FILE_DEV_DATASET        = path.join(DIR_PROCESSED, 'dev_dataset.pkl')
FILE_GLOBAL_WORD_CNTS   = path.join(DIR_PROCESSED, 'words_all.pkl')
FILE_WORDS_0            = path.join(DIR_PROCESSED, 'words_0.pkl')
FILE_WORDS_1            = path.join(DIR_PROCESSED, 'words_1.pkl')
FILE_LABEL_CNTS         = path.join(DIR_PROCESSED, 'label_cnts.pkl')
FILE_LABEL_IP_CNTS      = path.join(DIR_PROCESSED, 'label_ip_cnts.pkl')
FILE_LABEL_TIME_CNTS    = path.join(DIR_PROCESSED, 'label_time_cnts.pkl')
FILE_IDF                = path.join(DIR_PROCESSED, 'idf.pkl')


TOKEN_URL       = '[URL]'
TOKEN_EMAIL     = '[EMAIL]'
TOKEN_SYMBOLS   = '[SYMBOLS]'
TOKEN_NUM       = '[NUM]'

SEED = 123