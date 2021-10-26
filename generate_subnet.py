import random
import pickle

def random_subnet(seed=1):
    random.seed(seed)
    L = 24
    n = 1024
    names =  ['Q', 'K', 'V', 'O', 'FC1', 'FC2']
    subnets = {k : [] for  k in names}
    for layer in range(L):
        for name in names:
            subnets[name].append(random.sample(range(n), 3))
    return subnets


def save_subnet(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

data = random_subnet(1)
save_subnet(data, './subnet/r_1.pkl')