#!/usr/bin/env python

import sys
import math as m
import cyfrp
import time as t

start = t.clock()

if len(sys.argv)>1:
    jobrank = int(sys.argv[1]) - 1
else:
    jobrank = 0

if len(sys.argv)>2:
    numjobs = int(sys.argv[2])
else:
    numjobs = 1

def procarea(x0,y0,x1,y1):
    xlist = []
    ylist = []
    for x in range(x0,x1):
        for y in range(y0,y1):
            if x/3 == x/3.0 and y/3 == y/3.0:
                xlist.append(x)
                ylist.append(y)
    return xlist,ylist

# Fixed parameters
minLat = 62
maxLat = 68.6
minLon = -162
maxLon = -140

tabsize = 0.1 # Level of overlap between tiles, as fraction of shortest side of tile

# Inferred parameters
ntilex = m.sqrt(numjobs)
ntiley = numjobs/ntilex
tilex = abs(maxLon-minLon)/ntilex
tiley = abs(maxLat-minLat)/ntiley
tabsize *= min(tilex, tiley)

# Job specific paramters
yi = jobrank//ntilex
xi = int(ntilex*(jobrank/float(ntilex)-yi))

ax = minLon+xi*tilex
ay = minLat+yi*tiley
bx = ax+tilex
by = ay+tiley

if xi>0:
    ax -= tabsize
if yi>0:
    ay -= tabsize

# Analyse tile
print "(Lat =",ay,"Lon =",ax,") to (Lat =",by,"Lon =",bx,")"
cyfrp.run("data", jobrank, ay, by, ax, bx)

end = t.clock()
print 'Time elapsed {} seconds'.format(end-start)