#!/usr/bin/env python

import sys
import re
import cyfrp

target = sys.argv[1]

coords = open(target+"/coords.txt", "r")
line = coords.readline()
data = re.findall("\-?[0-9.]+", line)
ay = float(data[0])
by = float(data[1])
ax = float(data[2])
bx = float(data[3])
coords.close()
print 'Lat_min={0} Lat_max={1} Lon_min={2} Lon_max={3}'.format(ay, by, ax, bx)
cyfrp.run(target, 0, ay, by, ax, bx)
