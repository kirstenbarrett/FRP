#!/usr/bin/python
from scipy import ndimage
import numpy as np
import os
from osgeo import gdal

os.chdir('/Users/kirsten/Documents/data/MODIS/Boundary_swaths/')
filList = os.listdir('.')
filNam = 'MOD021KM.A2004178.2120.005.'
bands = ['BAND1','BAND2','BAND7','BAND21','BAND22','BAND31','BAND32','landmask','SolarZenith','SolarAzimuth','SensorZenith','SensorAzimuth','LAT','LON']

#READ IN ALL REFLECTANCE AND EMITTED BANDS
allArrays = {}
for b in bands:
    fullFilName = filNam + b + '.tif'
    ds = gdal.Open(fullFilName)
    data = np.array(ds.GetRasterBand(1).ReadAsArray())
#    data = data[1472:1546,566:656] #BOUNDARY FIRE AREA
    data = data[1105:2029,84:1065] #BOREAL AK AREA
    
    if b == 'BAND21' or b == 'BAND22' or b == 'BAND31' or b == 'BAND32':
#        data = np.int_(np.rint(data))
        data = data
    if b == 'BAND1' or b == 'BAND2' or b == 'BAND7':
        b = b + 'x1k'
        data = np.int_(np.rint(data*1000))

    allArrays[b] = data

[nRows,nCols] = np.shape(allArrays['BAND21'])



#DAY/NIGHT FLAG
dayFlag = np.zeros((nRows,nCols),dtype=np.int)
dayFlag[np.where(allArrays['SolarZenith'] < 8500)] = 1

waterFlag = -1
cloudFlag = -2

#CREATE WATER MASK
waterMask = np.zeros((nRows,nCols),dtype=np.int)
waterMask[np.where(allArrays['landmask']!=1)] = waterFlag

#CREATE CLOUD MASK (SET DATATYPE)
cloudMask =np.zeros((nRows,nCols),dtype=np.int)
cloudMask[((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>900)] = cloudFlag
cloudMask[(allArrays['BAND32']<265)] = cloudFlag
cloudMask[((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>700)&(allArrays['BAND32']<285)] = cloudFlag

#MASK CLOUDS AND WATER FROM INPUT BANDS
b21CloudWaterMasked = np.copy(allArrays['BAND21'])
b21CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
b21CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

b22CloudWaterMasked = np.copy(allArrays['BAND22'])
b22CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
b22CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

b31CloudWaterMasked = np.copy(allArrays['BAND31'])
b31CloudWaterMasked [np.where(waterMask == waterFlag)] = waterFlag
b31CloudWaterMasked [np.where(cloudMask == cloudFlag)] = cloudFlag

deltaT = np.abs(allArrays['BAND21'] - allArrays['BAND31'])
deltaTCloudWaterMasked = np.copy(deltaT)
deltaTCloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
deltaTCloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

##########################
##AFTER ALL THE DATA HAVE BEEN READ IN
##########################

minNcount = 8
minNfrac = 0.25
minKsize = 5
maxKsize = 21
b21saturationVal = 450 #???
reductionFactor= 1
increaseFactor = 1+(1-reductionFactor)

bgFlag = -3
bgMask = np.zeros((nRows,nCols),dtype=np.int)

with np.errstate(invalid='ignore'):
    bgMask[np.where((dayFlag == 1) & (allArrays['BAND21'] > (325*reductionFactor)) & (deltaT > (20*reductionFactor)))] = bgFlag
    bgMask[np.where((dayFlag == 0) & (deltaT > (310*reductionFactor))& (deltaT >(10*reductionFactor)))] = bgFlag

    b21bgMask = np.copy(b21CloudWaterMasked)
    b21bgMask[np.where(bgMask == bgFlag)] = bgFlag

    b22bgMask = np.copy(b22CloudWaterMasked)
    b22bgMask[np.where(bgMask == bgFlag)] = bgFlag

    b31bgMask = np.copy(b31CloudWaterMasked)
    b31bgMask[np.where(bgMask == bgFlag)] = bgFlag

    deltaTbgMask = np.copy(deltaTCloudWaterMasked)
    deltaTbgMask[np.where(bgMask == bgFlag)] = bgFlag
#############################
#ALL REQUIRED FUNCTION DEFS
#############################

def makeFootprint(kSize):
    fpZeroLine = (kSize-1)/2
    fpZeroColStart = fpZeroLine-1
    fpZeroColEnd = fpZeroColStart+3
    fp = np.ones((kSize,kSize),dtype = 'int_')
    fp[fpZeroLine,fpZeroColStart:fpZeroColEnd] = -5

    return fp

def meanFilt(kernel,kSize,minKsize,maxKsize):
    bgMean = -4
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2,0)))):
        fpMask = makeFootprint(kSize)
        kernel[np.where(fpMask < 0)] = -5
        nghbrs = kernel[np.where(kernel > 0)]
        nghbrCnt = len(nghbrs)
        
        if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
             bgMean = np.mean(nghbrs)
    return bgMean



def runFilt(band,filtFunc,minKsize,maxKsize):
    filtBand = band
    kSize = minKsize
    bandFilts = {}

    while kSize <=  maxKsize:
        filtName = 'bandFilt'+str(kSize)
        filtBand = ndimage.generic_filter(filtBand, filtFunc, size = kSize, extra_arguments= (kSize,minKsize,maxKsize))
        bandFilts[filtName] = filtBand
        kSize += 2

    bandFilt = bandFilts['bandFilt'+str(minKsize)]
    kSize = minKsize + 2

    while kSize <= maxKsize:
        bandFilt[np.where(bandFilt == -4)] = bandFilts['bandFilt'+str(kSize)][np.where(bandFilt == -4)]
        kSize += 2
        
    return bandFilt

    
############################
####B21 MEAN FILTER
############################

b21meanFilt = runFilt(b21bgMask,meanFilt,minKsize,maxKsize) 
b21minusBG = np.copy(b21CloudWaterMasked) - np.copy(b21meanFilt)

##TEST FOR SATURATION IN BAND 21
if (np.nanmax(b21CloudWaterMasked) > b21saturationVal):

    b22meanFilt = runFilt(b22bgMask,meanFilt,minKsize,maxKsize)
    b22minusBG = np.copy(b22CloudWaterMasked)  - np.copy(b22meanFilt)

    with np.errstate(invalid='ignore'):
        b21minusBG[(b21CloudWaterMasked >= b21saturationVal)] = b22minusBG[(b21CloudWaterMasked >= b21saturationVal)]

####POTENTIAL FIRE TEST
potFire = np.zeros((nRows,nCols),dtype=np.int)
with np.errstate(invalid='ignore'):
    potFire[(dayFlag == 1)&(allArrays['BAND21']>(310*reductionFactor))&(deltaT>(10*reductionFactor))&(allArrays['BAND2x1k']<(300*increaseFactor))] = 1
    potFire[(dayFlag == 0)&(allArrays['BAND21']>(305*reductionFactor))&(deltaT>(10*reductionFactor))] = 1

# ABSOLUTE THRESHOLD TEST (Kaufman et al. 1998) FOR REMOVING SUNGLINT
absValTest = np.zeros((nRows,nCols),dtype=np.int)
with np.errstate(invalid='ignore'):
    absValTest[(dayFlag == 1) & (allArrays['BAND21']>(360*reductionFactor))] = 1
    absValTest[(dayFlag == 0) & (allArrays['BAND21']>(305*reductionFactor))] = 1

#COMBINE TESTS
#DAYTIME "TENTATIVE FIRES"
fireLocTentative = potFire   
fireLocTentativeDay = potFire

dayFires = np.zeros((nRows,nCols),dtype=np.int)
with np.errstate(invalid='ignore'):
    dayFires[(dayFlag == 1)&((absValTest == 1)|(fireLocTentativeDay ==1))] = 1


#NIGHTTIME DEFINITE FIRES
nightFires = np.zeros((nRows,nCols),dtype=np.int)
with np.errstate(invalid='ignore'):
    nightFires[((dayFlag == 0)&((fireLocTentative == 1)|absValTest == 1))] = 1


###COMBINE ALL MASKS
allFires = dayFires+nightFires

b21firesAllMask = allFires*allArrays['BAND21']
b21bgAllMask = allFires*b21meanFilt

b21maskEXP = b21firesAllMask.astype(float)**8
b21bgEXP = b21bgAllMask.astype(float)**8

frpMW = 4.34 * (10**(-19)) * (b21maskEXP-b21bgEXP) #AREA TERM HERE

frpMWabs = frpMW*potFire #APPLY ABSOLUTE TEMP THRESHOLD

##################
##AREA CALCULATION
##################

##S = (I-hp)/H
##
##where:
##
##I is the zero-based pixel index
##hp is 1/2 the total number of pixels (zero-based)
##    (for MODIS each scan is 1354 "1km" pixels, 1353 zero-based, so hp = 676.5)
##H is the sensor altitude divided by the pixel size
##    (for MODIS altitude is approximately 700km, so for "1km" pixels, H = 700/1)

I = np.indices((nRows,nCols))[1]
hp = 676.6
H = 700

S = (I-hp)/H

##Compute the zenith angle:
Z = np.arcsin(1.111*np.sin(S))

##Compute the Along-track pixel size:
Pn = 1 #Pixel size in km at nadir
Pt = Pn*9*np.sin(Z-S)/np.sin(S)

##Compute the Along-scan pixel size:
Ps = Pt/np.cos(Z)

areaKmSq = Pt * Ps

frpMwKmSq = frpMWabs/areaKmSq

with np.errstate(invalid='ignore'):
    FRPlats = allArrays['LAT'][(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
    FRPlons =allArrays['LON'][(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
    Area = areaKmSq[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
    FRP = frpMWabs[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
    FrpArea = frpMwKmSq[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]

exportCSV = np.column_stack([FRPlons,FRPlats,Area,FRP,FrpArea])

np.savetxt(filNam+'frp_boreal_FRP_no_context_tests_filts_1124.csv', exportCSV, delimiter=",")




