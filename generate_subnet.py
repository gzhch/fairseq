import random
import pickle

def random_subnet(seed=1):
    random.seed(seed)
    L = 24
    n = 1024 + 1
    names =  ['Q', 'K', 'V', 'O', 'FC1', 'FC2']
    subnets = {k : [] for  k in names}
    for layer in range(L):
        for name in names:
            k = 3 if name != 'FC2' else 6
            n = 1024 if name != 'FC2'  else 4096
            subnets[name].append(random.sample(range(n + 1), k))
    return subnets


def fullnet():
    L = 24
    n = 1024 + 1
    names =  ['Q', 'K', 'V', 'O', 'FC1', 'FC2']
    subnets = {k : [] for  k in names}
    for layer in range(L):
        for name in names:
            n = 1024 if name != 'FC2'  else 4096
            subnets[name].append(random.sample(range(n), n))
    return subnets
    

def save_subnet(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

data = fullnet()
save_subnet(data, '../Data/subnet/fullnet.pkl')
