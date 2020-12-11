# This file will generate a CSV to be exported to the Deep Learning Algorithm
# Input Files: X dimensional glove vector space, comparisons.csv (created from log_parser)
# Output File: CSV 3 * X dimensions wide


import re
import numpy as np
import csv

d = {}

# Set this value to the dimensionality of the Glove Vector
glove_dimension = 50

f = open("codenames\players\glove.6B.50d.txt", encoding="utf-8")
# with open("codenames\players\glove.6B.50d.txt", encoding="utf-8")) as f:
for line in f:
    key = line.split(None, 1)[0]   # Grab the first word
    val = line.replace(key, '')    # Remove the first word
    val = val.strip()            # Remove the leading white space
    # Dictionary is mapped word -> val since each word is unique
    d[(key)] = val
f.close()
# print(d)

count = 0
# arr = np.empty((0,3), int)
data = np.empty(shape=(0, 3), dtype=object)

single_row = np.empty(shape=(0, 3 * glove_dimension), dtype=object)

# ex comparisions line: water,snow,ground
with open("/Users/Derek/Documents/GitHub/Game/codenames/comparisons.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar="'")
    for row in reader:
        # single_row.append(row.split)
        row_0 = d[row[0]]
        row_1 = d[row[1]]
        row_2 = d[row[2]]
        data = np.append(data, np.array([[row_0, row_1, row_2]]), axis=0)
        # data = np.append(data, np.array([row.split]), axis=0)
        # data.vstack(row_0, row_1, row_2)

np.savetxt("/Users/Derek/Documents/GitHub/Game/50d_testing.csv",
           data, fmt='%s', delimiter=" ")
csvfile.close()
