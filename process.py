#!/usr/bin/env python

import sys
import re
import cyfrp

target = sys.argv[1]
coords = open(target+"/coords.txt", "r")
line = coords.readline()
data = re.findall("[0-9.]+", line)
ay = data[0]
by = data[1]
ax = data[2]
bx = data[3]
coords.close()

cyfrp.run(target, 0, ay, by, ax, bx)