import numpy as np
import csv, glob

for fn in glob.glob('checkpoints/cifar10/*/log.txt'):
    print(fn)
    max_ = 0
    with open(fn, newline='') as f:
        for row in f:
            if 'Lear' in row:continue
            row=row.split('\t')
            max_ = max(float(row[-2]), max_)
    print(100-max_)

#
#         with open('eggs.csv', newline='') as csvfile:
#             ...
#             spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#         ...
#         for row in spamreader:
#             ...
#             print(', '.join(row))
# with open('eggs.csv', 'rb') as csvfile:
# ...     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
# ...     for row in spamreader:
# ...         print ', '.join(row)
# res.dialect
# res.reader
# res.line_num
# res.d
# print(res)
