import os
import csv

hdr = '"FRPline" ' \
    '"FRPsample" ' \
    '"FRPlats" ' \
    '"FRPlons" ' \
    '"FRPT21" ' \
    '"FRPT31" ' \
    '"FRPMeanT21" ' \
    '"FRPMeanT31" ' \
    '"FRPMeanDT" ' \
    '"FRPMADT21" ' \
    '"FRPMADT31" ' \
    '"FRP_MAD_DT" ' \
    '"FRPpower" ' \
    '"FRP_AdjCloud" ' \
    '"FRP_AdjWater" ' \
    '"FRP_NumValid" ' \
    '"FRP_confidence" ' \
    '"HDF_File"'

dirlist = [x for x in os.listdir('.') if ".csv" in x]

newFil = open('COMBINED.csv', 'w')
newFil.write('#' + hdr + '\n')

for fil in dirlist:
    with open(fil, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row = str(row)
            if '#' in row:
                continue
            row = row.replace('\\t', '\t').replace('[', '').replace(']', '').replace("'", "") + '\t' + fil + '\n'
            newFil.write(row)
newFil.close()