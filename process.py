#!/usr/bin/env python

import sys
import re
import cyfrp

target = sys.argv[1]
coords = open(target + "/coords.txt", "r")
line = coords.readline()
data = re.findall("[0-9.]+", line)
coords.close()

cyfrp.run(target, 0, data[0], data[1], data[2], data[3])