import sys
import re

name = sys.argv[1]

res = []
with open(name, 'r') as f:
    for l in f:
        if 'best_' in l:
            t = l[-10:].strip('\n').split()[-1]
            if t[0] == 'N':
                continue
            res.append(float(t))

best = res[-1]
cnt = 0
for l in res:
    if best - l > 1:
        cnt += 1
    else:
        break
print(best, cnt)