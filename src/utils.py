import pickle


def incr(d: dict, key, val=1):
    if key not in d:
        d[key] = val
    else:
        d[key] += val


def get_or(key, d: dict, default_val):
    if key in d:
        return d[key]
    return default_val


def list_to_occ_dict(lis) -> dict:
    d = {}
    for x in lis:
        incr(d, x, 1)
    return d


def cnt_chars(s):
    cnt = 0
    for c in s:
        if c.isalpha():
            cnt += 1
    return cnt


def pickle_save(data, filename):
    pickle.dump(data, open(filename, 'wb'))


def pickle_load(filename):
    return pickle.load(open(filename, 'rb'))
