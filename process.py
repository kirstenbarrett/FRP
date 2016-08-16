#!/usr/bin/env python

import sys
import re
import cyfrp
import pstats
import cProfile
import time

input = open("data/input.txt", "r")

coords = input.readline()
data = re.findall("\-?[0-9.]+", coords)
ay = float(data[0])
by = float(data[1])
ax = float(data[2])
bx = float(data[3])

reductionFactor = input.readline()
minNcount = input.readline()
minNfrac = input.readline()

kSize = input.readline()
data = re.findall("\-?[0-9.]+", kSize)
minKsize = data[0]
maxKsize = data[1]

decimalPlaces = input.readline()

doProfiling = input.readline()

directory = input.readline()

input.close()

if doProfiling.upper() == "FALSE":

  cyfrp.run(directory, ay, by, ax, bx, reductionFactor, minNcount, minNfrac, minKsize, maxKsize, decimalPlaces)

else:

  filename = 'profiles/{}'.format(time.strftime('%y%m%d%a.%H%M%S'))

  runStr = 'cyfrp.run("' + directory + '",' + str(ay) + ',' + str(by) + ',' + str(ax) + ',' + str(bx) + \
           ',' + reductionFactor + ',' + minNcount + ',' + minNfrac + ',' + minKsize + ',' + maxKsize + \
           ',' + decimalPlaces + ')'

  cProfile.runctx(runStr, globals(), locals(), '{}.prof'.format(filename))
  s = pstats.Stats('{}.prof'.format(filename))
  s.strip_dirs().sort_stats('time').print_stats(10)

  s = pstats.Stats('{}.prof'.format(filename), stream=open('{}.txt'.format(filename),'w'))
  s.strip_dirs().sort_stats('time').print_stats()