#!/usr/bin/env python

import sys
import re
import cyfrp

target = sys.argv[1]

input = open(target + "/input.txt", "r")

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

input.close()
print 'Lat_min={0} Lat_max={1} Lon_min={2} Lon_max={3}'.format(ay, by, ax, bx)
print 'reductionFactor={0} minNcount={1} minNfrac={2} minKsize={3} maxKsize={4}'.format(reductionFactor, minNcount, minNfrac, minKsize, maxKsize)
cyfrp.run(target, 0, ay, by, ax, bx, reductionFactor, minNcount, minNfrac, minKsize, maxKsize)
