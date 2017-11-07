#!/usr/bin/env python
import os
import re
import sys
import pstats
import cProfile
import time
import cyfrp
import numpy as np

#to handle a log of negative number issue
np.seterr(all='raise')

cwd = os.getcwd()

input = open("data/input.txt", "r")

coords = input.readline().strip()
data = re.findall("\-?[0-9.]+", coords)
ay = float(data[0])
by = float(data[1])
ax = float(data[2])
bx = float(data[3])

reductionFactor = input.readline().strip()
minNcount = input.readline().strip()
minNfrac = input.readline().strip()

kSize = input.readline().strip()
data = re.findall("\-?[0-9.]+", kSize)
minKsize = data[0]
maxKsize = data[1]

decimalPlaces = input.readline().strip()

directory = input.readline().strip()

input.close()

# Array job index
index = sys.argv[1]

filename = cwd + ('/profiles/{}-' + index).format(time.strftime('%y%m%d%a.%H%M%S'))

runStr = 'cyfrp.run("' + directory + '",' + str(index) + ',' + str(ay) + ',' + str(by) + ',' + str(ax) + ',' + str(bx) + \
         ',' + reductionFactor + ',' + minNcount + ',' + minNfrac + ',' + minKsize + ',' + maxKsize + \
         ',' + decimalPlaces + ')'

cProfile.runctx(runStr, globals(), locals(), '{}.prof'.format(filename))
s = pstats.Stats('{}.prof'.format(filename))
s.strip_dirs().sort_stats('time').print_stats(10)

s = pstats.Stats('{}.prof'.format(filename), stream=open('{}.txt'.format(filename),'w'))
s.strip_dirs().sort_stats('time').print_stats()
