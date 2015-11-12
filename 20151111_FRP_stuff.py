from scipy import ndimage
import numpy as np
import os
from osgeo import gdal
from collections import namedtuple

os.chdir('/Users/kirsten/Documents/data/MODIS/Boundary_swaths/20151110_new')
filList = os.listdir('.')
filNam = 'MOD021KM.A2004178.2120.005.'
bands = ['BAND1','BAND2','BAND7','BAND21','BAND22','BAND31','BAND32','landmask','SolarZenith','SolarAzimuth','SensorZenith','SensorAzimuth','LAT','LON']

#READ IN AL REFLECTANCE AND EMITTED BANDS
allArrays = {}
for b in bands:
    fullFilName = filNam + b + '.tif'
    ds = gdal.Open(fullFilName)
    data = np.array(ds.GetRasterBand(1).ReadAsArray())
    data = data[1283:1292,570:579] #TESTING DATA

    if b == 'BAND21' or b == 'BAND22' or b == 'BAND31' or b == 'BAND32':
        data = np.int_(np.rint(data))
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

#CHANGES LANDMASK TO BINARY (SET DATATYPE) "LAND" = 1
waterMask = np.zeros((nRows,nCols),dtype=np.int)
waterMask[np.where(allArrays['landmask']!=1)] = waterFlag

#CREATE CLOUD MASK (SET DATATYPE) "CLOUD" = 0
b1plus2gt900=np.zeros((nRows,nCols),dtype=np.int)
b1plus2gt900[np.where((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>900)] = cloudFlag

b32lt265 = np.ones((nRows,nCols),dtype=np.int)
b32lt265[np.where(allArrays['BAND32']<265)] = 0

##MASK CLOUDS AND WATER
b1plus2gt700=np.zeros((nRows,nCols),dtype=np.int)
b1plus2gt700[np.where((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>700)] = cloudFlag

b32lt285=np.zeros((nRows,nCols),dtype=np.int)
b32lt285[np.where(allArrays['BAND32']<285)]=cloudFlag

temp=np.zeros((nRows,nCols),dtype=np.int)
temp[np.where(b1plus2gt700+b32lt265)==0] = cloudFlag

cloudMask = np.zeros((nRows,nCols),dtype=np.int)
cloudMask[np.where((b1plus2gt900 == cloudFlag) & (b32lt285 == cloudFlag) & (temp == cloudFlag))] = cloudFlag

b21CloudWaterMasked = np.copy(allArrays['BAND21'])
b21CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
b21CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

b22CloudWaterMasked = np.copy(allArrays['BAND22'])
b22CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
b22CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

b31CloudWaterMasked = np.copy(allArrays['BAND31'])
b31CloudWaterMasked [np.where(waterMask == waterFlag)] = waterFlag
b31CloudWaterMasked [np.where(cloudMask == cloudFlag)] = cloudFlag

deltaT = np.copy(allArrays['BAND21']) - np.copy(allArrays['BAND31'])
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

bgFlag = -3
bgMask = np.zeros((nRows,nCols),dtype=np.int)
bgMask[np.where((dayFlag == 1) & (allArrays['BAND21'] >325))] = bgFlag
bgMask[np.where((dayFlag == 1) & (deltaT >20))] = bgFlag
bgMask[np.where((dayFlag == 0) & (deltaT >310))] = bgFlag
bgMask[np.where((dayFlag == 0) & (deltaT >10))] = bgFlag

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
            bgMean = np.int_(np.rint(np.mean(nghbrs)))
            
    return bgMean

def MADfilt(kernel,kSize,minKsize,maxKsize):
    bgMAD = -4
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2,0)))):
        fpMask = makeFootprint(kSize)
        kernel[np.where(fpMask < 0)] = -5
        nghbrs = kernel[np.where(kernel > 0)]
        nghbrCnt = len(nghbrs)
        
        if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
            bgMAD = np.int(np.rint(np.mean(abs(centerVal - nghbrs))))
            
    return bgMAD


def nValidFilt(kernel,kSize,minKsize,maxKsize): #USE BG mask files
    nghbrCnt = -4
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-3,0)))):
        fpMask = makeFootprint(kSize)
        kernel[np.where(fpMask < 0)] = -5
        nghbrs = kernel[np.where(kernel > 0)]
        nghbrCnt = len(nghbrs)
            
    return nghbrCnt

def nRejectBGfireFilt(kernel,kSize,minKsize,maxKsize): 
    nRejectBGfire = -4
    kernel = kernel.reshape((kSize,kSize))
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -4))):
        nRejectBGfire = len(kernel[np.where(kernel == -3)])
            
    return nRejectBGfire

def nRejectWaterFilt(kernel,kSize,minKsize,maxKsize): 
    nRejectWater = -4
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -4))):
        nRejectWater= len(kernel[np.where(kernel == -1)])
            
    return nRejectWater

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

def adjWater(kernel):
    nghbors = kernel[range(0,4)+range(5,9)]
    waterNghbors = kernel[np.where(nghbors == 1)]   
    nWaterNghbr = len(waterNghbors)
    return nWaterNghbr
    


############################
####B21 MEAN FILTER
############################

b21meanFilt = runFilt(b21bgMask,meanFilt,minKsize,maxKsize) 
b21minusBG = np.copy(b21CloudWaterMasked) - np.copy(b21meanFilt) #introduces weird flags, but all < 0

##TEST FOR SATURATION IN BAND 21
if (np.nanmax(b21CloudWaterMasked) > b21saturationVal):

    b22meanFilt = runFilt(b22bgMask,meanFilt,minKsize,maxKsize)
    b22minusBG = np.copy(b22CloudWaterMasked)  - np.copy(b22meanFilt)

    b21minusBG[(b21CloudWaterMasked >= b21saturationVal)] = b22minusBG[(b21CloudWaterMasked >= b21saturationVal)]

####POTENTIAL FIRE TEST
potFire = np.zeros((nRows,nCols),dtype=np.int)
potFire[(dayFlag == 1)&(allArrays['BAND21']>310)] = 1
potFire[(dayFlag == 1)&(deltaT>10)] = 1
potFire[(dayFlag == 0) & (allArrays['BAND21']>320)] = 1

# ABSOLUTE THRESHOLD TEST (Kaufman et al. 1998) FOR REMOVING SUNGLINT
##absValTest = np.zeros((nRows,nCols),dtype=np.int)
##absValTest[(dayFlag == 1) & (allArrays['BAND21']>360) & (deltaT > 10) & (allArrays['BAND2x1k']<300)] = 1
##absValTest[(dayFlag == 0) & (allArrays['BAND21']>305)] = 1

#########################################
#CONTEXT TESTS (GIGLIO ET AL 2003)
#########################################


####CONTEXT FIRE TEST 2:
deltaTmeanFilt = runFilt(deltaTbgMask,meanFilt,minKsize,maxKsize) 

####deltaT MAD Filtering
deltaTMADFilt = runFilt(deltaTbgMask,MADfilt,minKsize,maxKsize) 
deltaTMADfire = np.zeros((nRows,nCols),dtype=np.int)
deltaTMADfire[abs(deltaT)>(abs(deltaTmeanFilt) + (3.5*deltaTMADfire))] = 1


####CONTEXT FIRE TEST 3
deltaTfire = np.zeros((nRows,nCols),dtype=np.int)
deltaTfire[np.where(abs(deltaT) > (abs(deltaTmeanFilt) + 6))] = 1 

####CONTEXT FIRE TEST 4
B21fire = np.zeros((nRows,nCols),dtype=np.int)
b21MADfilt = runFilt(b21bgMask,MADfilt,minKsize,maxKsize) 
B21fire[(b21CloudWaterMasked > (b21meanFilt + (3*b21MADfilt)))] = 1


#POTENTIAL FIRE TEST 5
b31meanFilt = runFilt(b31bgMask,meanFilt,minKsize,maxKsize)
b31MADfilt = runFilt(b31bgMask,MADfilt,minKsize,maxKsize) 

B31fire = np.zeros((nRows,nCols),dtype=np.int)
B31fire[(b31CloudWaterMasked > (b31meanFilt + b31MADfilt - 4))] = 1

###CONTEXT FIRE TEST 6
rejectedBGmask = np.zeros((nRows, nCols),dtype=np.int)
rejectedBGmask[(bgMask == bgFlag)] = 1
rejB21bgFires = np.copy(b21CloudWaterMasked)

rejB21bgFires[(bgMask != bgFlag)] = bgFlag

b21rejMADfilt = runFilt(rejB21bgFires,MADfilt,minKsize,maxKsize)

B21rejFire = np.zeros((nRows,nCols),dtype=np.int)
B21rejFire[(b21rejMADfilt>=5)] = 1

############################################
####DESERT BOUNDARY TESTS
############################################

#COMBINE TESTS
#DAYTIME "TENATIVE FIRES"
fireLocTentative = deltaTMADfire*deltaTfire*B21fire
fireLocB31andB21refFire = np.zeros((nRows,nCols),dtype=np.int)
fireLocB31andB21refFire[np.where((B21rejFire == 1)|(B31fire == 1))]= 1
fireLocTentativeDay = potFire*fireLocTentative*fireLocB31andB21refFire

dayFires = np.zeros((nRows,nCols),dtype=np.int)
dayFires[np.where((dayFlag == 1)&(fireLocTentativeDay == 1))] = 1

#NIGHTTIME DEFINITE FIRES
nightFires = np.zeros((nRows,nCols),dtype=np.int)
nightFires[np.where((dayFlag == 0)&(fireLocTentative == 1))] = 1

###########################################
#####ADDITIONAL DAYTIME TESTS
##############################################

#sunglint rejection CHECK ALL UNITS of inputs to thresholds (e.g. thetaG, BAND1x1k)

relAzimuth = allArrays['SensorAzimuth']-allArrays['SolarAzimuth']
cosThetaG = (np.cos(allArrays['SensorZenith'])*np.cos(allArrays['SolarZenith']))- (np.sin(allArrays['SensorZenith'])*np.sin(allArrays['SolarZenith'])*np.cos(relAzimuth))
thetaG = np.arccos(cosThetaG)
thetaG = (thetaG/3.141592)*180
#thetaG = ((thetaG+3.141592)/(2*3.141592))*360 #NOT SURE WHICH ONE TO USE

#SUNGLINT TEST 8
sgTest8 = np.zeros((nRows,nCols),dtype=np.int)
sgTest8[np.where(thetaG < 2)] = 1

#SUNGLINT TEST 9
sgTest9 = np.zeros((nRows,nCols),dtype=np.int)
sgTest9[np.where((thetaG<8)&(allArrays['BAND1x1k']>100)&(allArrays['BAND2x1k']>200)&(allArrays['BAND7x1k']>120))] = 1

#SUNGLINT TEST 10
waterLoc = np.zeros((nRows,nCols),dtype=np.int)
waterLoc[np.where(waterMask == waterFlag)] = 1
nWaterAdj = ndimage.generic_filter(waterLoc, adjWater, size = 3)
nRejectedWater = runFilt(waterMask,nRejectWaterFilt,minKsize,maxKsize)
nRejectedWater[np.where(nRejectedWater<0)] = 0
                           
sgTest10 = np.zeros((nRows,nCols),dtype=np.int)
sgTest10[np.where((thetaG<12) & ((nWaterAdj+nRejectedWater)>0))] = 1

#desert boundary rejection

nValid = runFilt(b21bgMask,nValidFilt,minKsize,maxKsize)
nRejectedBG = runFilt(bgMask,nRejectBGfireFilt,minKsize,maxKsize)
nRejectedBG[np.where(nRejectedBG<0)] = 0

#DESERT BOUNDARY TEST 11
dbTest11 = np.zeros((nRows,nCols),dtype=np.int)
dbTest11[np.where(nRejectedBG>(0.1*nValid))] = 1

#DB TEST 12
dbTest12 = np.zeros((nRows,nCols),dtype=np.int)
dbTest12[(nRejectedBG>=4)] = 1

#DB TEST 13
dbTest13 = np.zeros((nRows,nCols),dtype=np.int)
dbTest13[np.where(allArrays['BAND2x1k']>150)] = 1

#DB TEST 14
#ON REJECTED PIXELS MEAN T4
b21rejBG = np.copy(b21CloudWaterMasked)
b21rejBG[np.where(bgMask != bgFlag)] = bgFlag
b21rejMeanFilt = runFilt(b21rejBG,meanFilt,minKsize,maxKsize)
dbTest14 = np.zeros((nRows,nCols),dtype=np.int)
dbTest14[(b21rejMeanFilt<345)] = 1

#DB TEST 15
dbTest15 = np.zeros((nRows,nCols),dtype=np.int)
dbTest15[(b21rejMADfilt>=3)] = 1

#DB TEST 16
dbTest16 = np.zeros((nRows,nCols),dtype=np.int)
dbTest16[(b21CloudWaterMasked<(b21rejMeanFilt+(6*b21rejMADfilt)))] = 1

dbAll = dbTest11*dbTest12*dbTest13*dbTest14*dbTest15*dbTest16
#CHUCK OUT ANYTHING THAT FULFILLS ALL DESERT BOUNDARY CRITERIA

#coastal false alarm rejection
####

##SHOULD YIELD ~176 FIRES FOR 2004178.2120          
#OUTPUT DF

b21maskEXP = b21firesAllMask.astype(float)**8
b21bgEXP = b21bgAllMask.astype(float)**8

frpMW = 4.34 * (10**(-19)) * (b21maskEXP-b21bgEXP)#AREA TERM HERE

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

##inds=np.where(frpMwKmSq>0)
##FRPlats = allArrays['LAT'][inds]
##FRPlons =allArrays['LON'][inds]
##FrpInds = frpMwKmSq[inds]
#exportCSV = np.column_stack([FRPlons,FRPlats,FrpInds])
#np.savetxt("frpMwKm_2004178_2120.csv", exportCSV, delimiter=",")



