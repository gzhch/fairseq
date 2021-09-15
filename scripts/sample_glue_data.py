import random
data = []
head 
with open('/new_home/zhuocheng/blobs/gzhch/data/glue_data/MNLI/train.tsv', 'r') as f:
    cnt = 0
    for l in f:
        if cnt == 0:
        cnt += 1
        if cnt == 5:
            break