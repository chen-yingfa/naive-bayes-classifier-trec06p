import os
import random
import re
import codecs
from tqdm import tqdm
from config import *
from utils import *


_file = os.path.basename(__file__)


class DataLoader:
    def __init__(self, index_file, data_size=1.0, shuffle=True):
        self.data = []
        self.index_file = index_file
        self.data_size = data_size
        self.shuffle = shuffle

        self.load_data()
        print(f'[{_file}] Loaded {len(self.data)} examples')

    def remove_special_chars(self, s) -> str:
        # remove = '!"()}{?$#@|%'
        remove = '&<>.,:;_^-+=/\\*!"()}{?$#@|%'
        replace_with_space = '&<>.,:;_^-+=/\\*!"()}{?$#@|%'
        for c in remove:
            s = s.replace(c, '')
        for c in replace_with_space:
            s = s.replace(c, ' ')
        return s

    def parse_email(self, filename) -> dict:
        email = {}
        email['words'] = {}
        email['ip'] = None
        email['hour'] = None
        f = codecs.open(filename, 'r', 'utf8', errors='ignore')

        re_url = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        re_only_special_char = re.compile('^[\W_]+$')
        re_email = re.compile('[^@]+@[^@]+\.[^@]+')
        re_special_char = re.compile('\W')

        # When True, everything afterwards is content
        reached_content = False
        for line in f:
            line = line.strip().lower()
            # Parse ip
            if re.match(r'received: from.*', line):
                ip_re = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', line, flags=0)
                email['ip'] = ip_re[0] if len(ip_re) > 0 else None
            # Parse time
            if re.match(r'date: .*', line):
                if not reached_content:
                    date_re = re.findall(r'\d+:', line, flags=0)
                    if len(date_re) > 0:
                        hour = date_re[0].strip(":")
                        email['hour'] = hour
            elif re.match(r'^$', line):
                reached_content = True
            elif not reached_content:
                continue
                
            # Parse content
            # TODO: Use a pretrained tokenizer instead
            else:
                line = re.sub(re_url, TOKEN_URL, line)
                line = re.sub(re_email, TOKEN_EMAIL, line)
                line = re.sub(re_only_special_char, TOKEN_SYMBOLS, line)
                line = self.remove_special_chars(line)

                words = line.split()

                for word in words:
                    # Skip numbers
                    # if not any(c.isalpha() for c in word):
                        # Skip all words without letters
                        # continue
                    # elif cnt_chars(word) < len(word) // 2:
                        # Skip all words where less than half of chars
                        # are special non-letters
                        # continue
                    # Count occurrence of word
                    incr(email['words'], word)
        f.close()
        return email

    def load_labels(self, filename) -> dict:
        """
        Return dict of dict of int. 1 = spam, 0 = ham
        """
        print(f"[{_file}] Loading labels from {filename }")
        labels = {}
        label_to_id = {'ham': 0, 'spam': 1}
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Shuffle and keep part of dataset based on hyperparameters
        if self.shuffle:
            random.shuffle(lines)
        size = int(len(lines) * self.data_size)
        lines = lines[: size]

        for line in lines:
            line = line.strip().split()
            if len(line) < 2:
                continue
            label = label_to_id[line[0]]
            path = line[1]
            path = path.split('/')
            dir_0 = path[2]  
            dir_1 = path[3]

            if dir_0 not in labels:
                labels[dir_0] = {}
            if dir_1 not in labels[dir_0]:
                labels[dir_0][dir_1] = {}
            labels[dir_0][dir_1] = label
        return labels

    def load_data(self) -> None:
        """
        Return list of Emails (dict)
        `index_file`: path to index file
        """
        labels = self.load_labels(self.index_file)

        num_examples = 0
        for x in labels:
            for y in labels[x]:
                num_examples += 1

        print(f'[{_file}] Loading {num_examples} examples...')
        for dir_0 in tqdm(labels):
            for dir_1 in labels[dir_0]:
                label = labels[dir_0][dir_1]
                path = os.path.join(DIR_DATA, 'data', dir_0, dir_1)
                email = self.parse_email(path)
                email['label'] = label
                self.data.append(email)
