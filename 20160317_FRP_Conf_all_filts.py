#!/usr/bin/python
from scipy import ndimage
import numpy as np
import os
from osgeo import gdal
from pyproj import Proj, transform
import datetime
from scipy.stats import gmean


os.chdir('/media/ubuntu/TLINGIT/data/Remote_Sensing/MODIS/Giglio_test')
filList = os.listdir('.')

bands = ['BAND1','BAND2','BAND7','BAND21','BAND22','BAND31','BAND32','LANDMASK','SolarZenith','SolarAzimuth','SensorZenith','SensorAzimuth','LAT','LON']

#AK BOREAL EXTENT
minX = -511738.931
minY = 1176158.734
maxX = 672884.463
maxY = 2117721.949

nProjRows = np.int_(np.rint((maxY-minY)/1000))
nProjCols = np.int_(np.rint((maxX-minX)/1000))
minNcount = 8
minNfrac = 0.25
minKsize = 5
maxKsize = 21
b21saturationVal = 450 #???
reductionFactor= 1
increaseFactor = 1+(1-reductionFactor)
waterFlag = -1
cloudFlag = -2
bgFlag = -3
resolution = 5
datsWdata = []
datIter = 0 #START AFTER MISSING DATA
#i = 0

datList=[]
filNamList = []

for fil in filList:
    if fil[-3:] == 'tif' and 'KM' in fil:
        filNam = fil[0:27]
        csvNam = filNam + 'iterByDate_conf_wakelin.csv'
        if (filNam not in filNamList) and (csvNam not in filList):
            filNamList.append(filNam)
            datTim = fil.split('.')[1].replace('A','') + fil.split('.')[2]
            dateTime = datetime.datetime.strptime(datTim, "%Y%j%H%M")
            if dateTime not in datList:
                datList.append(dateTime)

del filNam
datList.sort()

#############################
#ALL REQUIRED FUNCTION DEFS
#############################
def adjCloud(kernel):
    nghbors = kernel[range(0,4)+range(5,9)]
    cloudNghbors = kernel[np.where(nghbors == 1)]   
    nCloudNghbr = len(cloudNghbors)
    return nCloudNghbr

def adjWater(kernel):
    nghbors = kernel[range(0,4)+range(5,9)]
    waterNghbors = kernel[np.where(nghbors == 1)]   
    nWaterNghbr = len(waterNghbors)
    return nWaterNghbr

def makeFootprint(kSize):
    fpZeroLine = (kSize-1)/2
    fpZeroColStart = fpZeroLine-1
    fpZeroColEnd = fpZeroColStart+3
    fp = np.ones((kSize,kSize),dtype = 'int_')
    fp[fpZeroLine,fpZeroColStart:fpZeroColEnd] = -5
    return fp


#RETURN NUMBER OF NON-BACKGROUND FIRE, NON-CLOUD, NON-WATER NEIGHBORS
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

#RETURN NUMBER OF NEIGHBORS REJECTED AS BACKGROUND
def nRejectBGfireFilt(kernel,kSize,minKsize,maxKsize):
    nRejectBGfire = -4
    kernel = kernel.reshape((kSize,kSize))
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]

    if (((kSize == minKsize) | (centerVal == -4))):
        nRejectBGfire = len(kernel[np.where(kernel == -3)])

    return nRejectBGfire

#RETURN NUMBER OF NEIGHBORS REJECTED AS WATER
def nRejectWaterFilt(kernel,kSize,minKsize,maxKsize):
    nRejectWater = -4
    kernel = kernel.reshape((kSize,kSize))

    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]

    if (((kSize == minKsize) | (centerVal == -4))):
        nRejectWater= len(kernel[np.where(kernel == -1)])

    return nRejectWater

#RETURN NUMBER OF 'UNMASKED WATER' NEIGHBORS
def nUnmaskedWaterFilt(kernel,kSize,minKsize,maxKsize):
    nUnmaskedWater = -4
    kernel = kernel.reshape((kSize,kSize))

    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]

    if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-3,0)))):
        nUnmaskedWater= len(kernel[np.where(kernel == -6)])

    return nUnmaskedWater

def rampFn(band,rampMin,rampMax):
    conf = 0
    confVals = []
    for bandVal in band:
        if rampMin < bandVal < rampMax:
            conf = (bandVal-rampMin)/(rampMax-rampMin)
        if bandVal >= rampMax: #I THINK THIS SHOULD BE GREATER THAN!!!
            conf = 1
        confVals.append(conf)
    #masked values (-3) return conf of 0
    return np.asarray(confVals)

#RUNS FILTERS ON PROGRESSIVELY LARGER KERNEL SIZES, COMBINES RESULTS FROM SMALLEST KSIZE
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


def wakelinMeanMADFilter(band,maxKsize,minKsize):
    
    # Add boundary for largest known tile size (maxKsize)
    bSize = (maxKsize-1)/2 
    bandMatrix = np.pad(band,((bSize,bSize),(bSize,bSize)),mode='symmetric')

    bandFiltsMean2 = {}
    bandFiltsMAD2 = {}
    kSize = minKsize 
    i,j = np.shape(band)

    # Loop through dataset
    while kSize <=  maxKsize:
        
        bandMADFilt2_tmp  = np.full([i,j], -4.0)
        bandMeanFilt2_tmp = np.full([i,j], -4.0)
        
        halfK = (kSize-1)/2
        for x in range(bSize,i+bSize):
            for y in range(bSize,j+bSize):

                xmhk = x-halfK
                xphk = x+halfK+1
                ymhk = y-halfK
                yphk = y+halfK+1

                # Must copy kernel otherwise it is a reference to original array - hence original is changed!
                kernel = bandMatrix[xmhk:xphk:1,ymhk:yphk:1].copy()
                centerVal = bandMatrix[x,y]


                if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2,0)))):
                    fpMask = makeFootprint(kSize)
                    kernel[np.where(fpMask < 0)] = -5
                    nghbrs = kernel[np.where(kernel > 0)]
                    nghbrCnt = len(nghbrs)
                    
                    if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
                        bgMean = np.mean(nghbrs)
                        meanDists = np.abs(nghbrs - bgMean)
                        bgMAD = np.mean(meanDists)

                        # Remember - Results matrix is smaller than padded dataset by bSize in all directions
                        xmb = x-bSize
                        ymb = y-bSize
                        bandMADFilt2_tmp[xmb,ymb] = bgMAD
                        bandMeanFilt2_tmp[xmb,ymb] = bgMean

                        
        filtNameMean2 = 'bandFiltMean'+str(kSize)
        bandFiltsMean2[filtNameMean2] = bandMeanFilt2_tmp
        filtNameMAD2 = 'bandFiltMAD'+str(kSize)
        bandFiltsMAD2[filtNameMAD2] = bandMADFilt2_tmp

        kSize += 2



    bandFiltMean2 = bandFiltsMean2['bandFiltMean'+str(minKsize)]
    bandFiltMAD2 = bandFiltsMAD2['bandFiltMAD'+str(minKsize)]
    kSize = minKsize + 2

    while kSize <= maxKsize:
        bandFiltMean2[np.where(bandFiltMean2 == -4)] = bandFiltsMean2['bandFiltMean'+str(kSize)][np.where(bandFiltMean2 == -4)]
        bandFiltMAD2[np.where(bandFiltMAD2 == -4)] = bandFiltsMAD2['bandFiltMAD'+str(kSize)][np.where(bandFiltMAD2 == -4)]
        kSize += 2

    return bandFiltMean2,bandFiltMAD2


def wakelinMeanFilter(band,maxKsize,minKsize):
    
    # Add boundary for largest known tile size (maxKsize)
    bSize = (maxKsize-1)/2 
    bandMatrix = np.pad(band,((bSize,bSize),(bSize,bSize)),mode='symmetric')

    bandFiltsMean2 = {}
    kSize = minKsize 
    i,j = np.shape(band)

    # Loop through dataset
    while kSize <=  maxKsize:
        bandMeanFilt2_tmp = np.full([i,j], -4.0)
        halfK = (kSize-1)/2
        for x in range(bSize,i+bSize):
            for y in range(bSize,j+bSize):

                xmhk = x-halfK
                xphk = x+halfK+1
                ymhk = y-halfK
                yphk = y+halfK+1

                # Must copy kernel otherwise it is a reference to original array - hence original is changed!
                kernel = bandMatrix[xmhk:xphk:1,ymhk:yphk:1].copy()
                centerVal = bandMatrix[x,y]


                if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2,0)))):
                    fpMask = makeFootprint(kSize)
                    kernel[np.where(fpMask < 0)] = -5
                    nghbrs = kernel[np.where(kernel > 0)]
                    nghbrCnt = len(nghbrs)
                    
                    if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
                        bgMean = np.mean(nghbrs)
                        meanDists = np.abs(nghbrs - bgMean)

                        # Remember - Results matrix is smaller than padded dataset by bSize in all directions
                        xmb = x-bSize
                        ymb = y-bSize
                        bandMeanFilt2_tmp[xmb,ymb] = bgMean

                        
        filtNameMean2 = 'bandFiltMean'+str(kSize)
        bandFiltsMean2[filtNameMean2] = bandMeanFilt2_tmp
        kSize += 2

    bandFiltMean2 = bandFiltsMean2['bandFiltMean'+str(minKsize)]
    kSize = minKsize + 2

    while kSize <= maxKsize:
        bandFiltMean2[np.where(bandFiltMean2 == -4)] = bandFiltsMean2['bandFiltMean'+str(kSize)][np.where(bandFiltMean2 == -4)]
        kSize += 2

    return bandFiltMean2

def wakelinMADFilter(band,maxKsize,minKsize):
    
    # Add boundary for largest known tile size (maxKsize)
    bSize = (maxKsize-1)/2 
    bandMatrix = np.pad(band,((bSize,bSize),(bSize,bSize)),mode='symmetric')

    bandFiltsMAD2 = {}
    kSize = minKsize 
    i,j = np.shape(band)

    # Loop through dataset
    while kSize <=  maxKsize:
        
        bandMADFilt2_tmp  = np.full([i,j], -4.0)
        
        halfK = (kSize-1)/2
        for x in range(bSize,i+bSize):
            for y in range(bSize,j+bSize):

                xmhk = x-halfK
                xphk = x+halfK+1
                ymhk = y-halfK
                yphk = y+halfK+1

                # Must copy kernel otherwise it is a reference to original array - hence original is changed!
                kernel = bandMatrix[xmhk:xphk:1,ymhk:yphk:1].copy()
                centerVal = bandMatrix[x,y]


                if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2,0)))):
                    fpMask = makeFootprint(kSize)
                    kernel[np.where(fpMask < 0)] = -5
                    nghbrs = kernel[np.where(kernel > 0)]
                    nghbrCnt = len(nghbrs)
                    
                    if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
                        bgMean = np.mean(nghbrs)
                        meanDists = np.abs(nghbrs - bgMean)
                        bgMAD = np.mean(meanDists)

                        # Remember - Results matrix is smaller than padded dataset by bSize in all directions
                        xmb = x-bSize
                        ymb = y-bSize
                        bandMADFilt2_tmp[xmb,ymb] = bgMAD

                        
        filtNameMAD2 = 'bandFiltMAD'+str(kSize)
        bandFiltsMAD2[filtNameMAD2] = bandMADFilt2_tmp

        kSize += 2


    bandFiltMAD2 = bandFiltsMAD2['bandFiltMAD'+str(minKsize)]
    kSize = minKsize + 2

    while kSize <= maxKsize:
        bandFiltMAD2[np.where(bandFiltMAD2 == -4)] = bandFiltsMAD2['bandFiltMAD'+str(kSize)][np.where(bandFiltMAD2 == -4)]
        kSize += 2

    return bandFiltMAD2


##########################################################

while datIter < len(datList):
    t = datList[datIter]

    #GET REQUIRED TIFS
    julianDay = str(t.timetuple().tm_yday)
    jZeros = 3-len(julianDay)
    julianDay = '0'*jZeros+julianDay
    yr = str(t.year)
    hr = str(t.hour)
    if len(hr) < 2:
        hr = '0'+hr
    mint = str(t.minute)
    if len(mint) < 2:
        mint = '0'+mint
    datNam = str(t.year)+julianDay+'.'+hr+mint

    for filNamCandidate in filNamList:
        if datNam in filNamCandidate:
            filNam = filNamCandidate
    
    #READ IN ALL REFLECTANCE AND EMITTED BANDS
    allArrays = {}
    for b in bands:
        fullFilName = filNam + b + '.tif'
        ds = gdal.Open(fullFilName)
        data = np.array(ds.GetRasterBand(1).ReadAsArray())
        data = data[400:600,400:600] #BOUNDARY y2004 d180 21h 10m
 #       data = data[1472:1546,566:656] #BOUNDARY FIRE AREA
#      data = data[1105:2029,84:1065] #BOREAL AK AREA
        
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

    #CREATE WATER MASK
    waterMask = np.zeros((nRows,nCols),dtype=np.int)
    waterMask[np.where(allArrays['LANDMASK']!=1)] = waterFlag

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

####################################################################################
#### MEAN AND MAD FILTERS (MAD NEEDED FOR CONFIDENCE ESTIMATION)
####################################################################################

    b21meanFilt,b21MADfilt = wakelinMeanMADFilter(b21bgMask,maxKsize,minKsize)
    b31meanFilt,b31MADfilt = wakelinMeanMADFilter(b21bgMask,maxKsize,minKsize)
    deltaTmeanFilt, deltaTMADFilt = wakelinMeanMADFilter(deltaTbgMask, maxKsize, minKsize)

    ###FOR CONTEXT FIRE TEST 6
    rejB21bgFires = np.copy(b21CloudWaterMasked)
    with np.errstate(invalid='ignore'):
        rejB21bgFires[(bgMask != bgFlag)] = bgFlag #PROCESS BG PIXELS
    b21rejMeanFilt, b21rejMADfilt = wakelinMeanMADFilter(rejB21bgFires, maxKsize, minKsize)

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


    #########################################
    #CONTEXT TESTS
    #########################################

    deltaTMADfire = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        deltaTMADfire[deltaT>(deltaTmeanFilt + (3.5*deltaTMADFilt))] = 1

    ####CONTEXT FIRE TEST 3
    deltaTfire = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        deltaTfire[np.where(deltaT > (deltaTmeanFilt + 6))] = 1

    ####CONTEXT FIRE TEST 4
    b21minusBG = np.copy(b21CloudWaterMasked) - np.copy(b21meanFilt)

    ##TEST FOR SATURATION IN BAND 21
    if (np.nanmax(b21CloudWaterMasked) > b21saturationVal):

        b22meanFilt = wakelinMeanFilter(b22bgMask,maxKsize,minKsize)
        b22minusBG = np.copy(b22CloudWaterMasked)  - np.copy(b22meanFilt)

        with np.errstate(invalid='ignore'):
            b21minusBG[(b21CloudWaterMasked >= b21saturationVal)] = b22minusBG[(b21CloudWaterMasked >= b21saturationVal)]

    B21fire = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        B21fire[(b21CloudWaterMasked > (b21meanFilt + (3*b21MADfilt)))] = 1

    ####CONTEXT  FIRE TEST 5
    B31fire = np.zeros((nRows,nCols),dtype=np.int)
    B31fire[(b31CloudWaterMasked > (b31meanFilt + b31MADfilt - 4))] = 1

    ###CONTEXT FIRE TEST 6
    B21rejFire = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        B21rejFire[(b21rejMADfilt>5)] = 1

    #COMBINE TESTS TO CREATE "TENTATIVE FIRES"
    fireLocTentative = deltaTMADfire*deltaTfire*B21fire

 #   fireLocB31andB21rejFire = np.zeros((nRows,nCols),dtype=np.int)
#    with np.errstate(invalid='ignore'):
 #       fireLocB31andB21rejFire[np.where((B21rejFire == 1)|(B31fire == 1))]= 1
#    fireLocTentativeDay = potFire*fireLocTentative*fireLocB31andB21rejFire
    fireLocTentativeDay = potFire*fireLocTentative

    dayFires = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        dayFires[(dayFlag == 1)&((absValTest == 1)|(fireLocTentativeDay ==1))] = 1

    #NIGHTTIME DEFINITE FIRES (NO FURTHER TESTS)
    nightFires = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        nightFires[((dayFlag == 0)&((fireLocTentative == 1)|absValTest == 1))] = 1

    ###########################################
    #####ADDITIONAL DAYTIME TESTS ON TENTATIVE FIRES
    ##############################################

    #SUNGLINT REJECTION
    relAzimuth = allArrays['SensorAzimuth']-allArrays['SolarAzimuth']
    cosThetaG = (np.cos(allArrays['SensorZenith'])*np.cos(allArrays['SolarZenith']))- (np.sin(allArrays['SensorZenith'])*np.sin(allArrays['SolarZenith'])*np.cos(relAzimuth))
    thetaG = np.arccos(cosThetaG)
    thetaG = (thetaG/3.141592)*180

    #SUNGLINT TEST 8
    sgTest8 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        sgTest8[np.where(thetaG < 2)] = 1

    #SUNGLINT TEST 9
    sgTest9 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        sgTest9[np.where((thetaG<8)&(allArrays['BAND1x1k']>100)&(allArrays['BAND2x1k']>200)&(allArrays['BAND7x1k']>120))] = 1

    #SUNGLINT TEST 10
    waterLoc = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        waterLoc[np.where(waterMask == waterFlag)] = 1
    nWaterAdj = ndimage.generic_filter(waterLoc, adjWater, size = 3)
    nRejectedWater = runFilt(waterMask,nRejectWaterFilt,minKsize,maxKsize)
    with np.errstate(invalid='ignore'):
        nRejectedWater[np.where(nRejectedWater<0)] = 0

    sgTest10 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        sgTest10[np.where((thetaG<12) & ((nWaterAdj+nRejectedWater)>0))] = 1

    sgAll = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        sgAll[(sgTest8 == 1) | (sgTest9 == 1) | (sgTest10 == 1)] = 1

    #DESERT BOUNDARY REJECTION

    nValid = runFilt(b21bgMask,nValidFilt,minKsize,maxKsize)

    nRejectedBG = runFilt(bgMask,nRejectBGfireFilt,minKsize,maxKsize)
    with np.errstate(invalid='ignore'):
        nRejectedBG[np.where(nRejectedBG<0)] = 0

    #DESERT BOUNDARY TEST 11
    dbTest11 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        dbTest11[np.where(nRejectedBG>(0.1*nValid))] = 1

    #DB TEST 12
    dbTest12 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        dbTest12[(nRejectedBG>=4)] = 1

    #DB TEST 13
    dbTest13 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        dbTest13[np.where(allArrays['BAND2x1k']>150)] = 1

    #DB TEST 14
    dbTest14 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        dbTest14[(b21rejMeanFilt<345)&(b21rejMeanFilt != -4)] = 1

    #DB TEST 15
    dbTest15 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        dbTest15[(b21rejMADfilt<3)&(b21rejMeanFilt != -4)] = 1

    #DB TEST 16
    dbTest16 = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        dbTest16[(b21CloudWaterMasked<(b21rejMeanFilt+(6*b21rejMADfilt)))&(b21rejMADfilt != -4)&(b21rejMeanFilt != -4)] = 1

    #REJECT ANYTHING THAT FULFILLS ALL DESERT BOUNDARY CRITERIA
    dbAll = dbTest11*dbTest12*dbTest13*dbTest14*dbTest15*dbTest16

    #COASTAL FALSE ALARM REJECTION
    with np.errstate(invalid='ignore'):
        ndvi = (allArrays['BAND2x1k']+allArrays['BAND1x1k'])/(allArrays['BAND2x1k']+allArrays['BAND1x1k'])
    unmaskedWater = np.zeros((nRows,nCols),dtype=np.int)
    uwFlag = -6
    with np.errstate(invalid='ignore'):
        unmaskedWater[((ndvi<0) & (allArrays['BAND7x1k']<50)&(allArrays['BAND2x1k']<150))] = -6
        unmaskedWater[(bgMask == bgFlag)] = bgFlag
    Nuw = runFilt(unmaskedWater,nUnmaskedWaterFilt,minKsize,maxKsize)
    rejUnmaskedWater = np.zeros((nRows,nCols),dtype=np.int)
    with np.errstate(invalid='ignore'):
        rejUnmaskedWater[(absValTest == 0) & (Nuw>0)] = 1

    #COMBINE ALL MASKS
    allFires = dayFires+nightFires #ALL POTENTIAL FIRES
    with np.errstate(invalid='ignore'): #REJECT SUNGLINT, DESERT BOUNDARY, COASTAL FALSE ALARMS
        allFires[(sgAll == 1) | (dbAll == 1) | (rejUnmaskedWater == 1)] = 0


    if np.max(allFires) > 0:
        datsWdata.append(datList[datIter])

        b21firesAllMask = allFires*allArrays['BAND21']
        b21bgAllMask = allFires*b21meanFilt

        b21maskEXP = b21firesAllMask.astype(float)**8
        b21bgEXP = b21bgAllMask.astype(float)**8

        frpMW = 4.34 * (10**(-19)) * (b21maskEXP-b21bgEXP) #AREA TERM HERE

        frpMWabs = frpMW*potFire #APPLY ABSOLUTE TEMP THRESHOLD
        
        #########################
        #DETECTION CONFIDENCE
        #########################
        cloudLoc = np.zeros((nRows,nCols),dtype=np.int)
        with np.errstate(invalid='ignore'):
            cloudLoc[np.where(cloudMask == cloudFlag)] = 1
        nCloudAdj = ndimage.generic_filter(cloudLoc, adjCloud, size = 3)

        waterLoc = np.zeros((nRows,nCols),dtype=np.int)
        with np.errstate(invalid='ignore'):
            waterLoc[np.where(waterMask == waterFlag)] = 1
        nWaterAdj = ndimage.generic_filter(waterLoc, adjWater, size = 3)

        #Fire Detection Confidence 17
        z4 = b21minusBG/b21MADfilt

        #Fire Detection Confidence 18
        zDeltaT = (deltaTbgMask-deltaTmeanFilt)/deltaTMADFilt

        with np.errstate(invalid='ignore'):
            firesNclouds = nCloudAdj[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
            firesZ4 = z4[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
            firesZdeltaT = zDeltaT[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
            firesB21bgMask = b21bgMask[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
            firesNwater = nWaterAdj[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]

        #Fire Detection Confidence 19 (SHOULD BE DIFFERENT IN NIGHT AND DAY)
        C1 = rampFn(firesB21bgMask, 310, 340)

        #Fire Detection Confidence 20
        C2 = rampFn(firesZ4, 2.5, 6)

        #Fire Detection Confidence 21
        C3 = rampFn(firesZdeltaT, 3, 6)

        #Fire Detection Confidence 22
        C4 = 1-rampFn(firesNclouds, 0, 6)
        ##ZERO CLOUDS = ZERO CONFIDENCE????

        #Fire Detection Confidence 23
        C5 = 1-rampFn(firesNwater, 0, 6)

        confArray = np.row_stack((C1,C2,C3,C4,C5))
        detnConf = gmean(confArray, axis = 0)

        ##############################################
        

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
            FRPx = np.where((allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900))[1]
            FRPy = np.where((allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900))[0]
            FRPlats = allArrays['LAT'][(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
            FRPlons =allArrays['LON'][(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
            Area = areaKmSq[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
            FRP = frpMWabs[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
            FrpArea = frpMwKmSq[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]

        inProj = Proj(init='epsg:4326') #GEOGRAPHIC WGS84
        outProj = Proj(init='esri:102006') #AK ALBERS EQUAL AREA CONIC

        FRPxProj,FRPyProj = transform(inProj,outProj,FRPlons,FRPlats)
        hrs = np.array(np.repeat(int(hr),len(FRP)))
        mints = np.array(np.repeat(int(mint),len(FRP)))
        js = np.array(np.repeat(int(julianDay),len(FRP)))
        yrs = np.array(np.repeat(int(yr),len(FRP)))
                
        exportCSV = np.column_stack([FRPx,FRPy,FRPxProj,FRPyProj,FRPlats,FRPlons,Area,FRP,FrpArea,hrs,mints,js,yrs,detnConf])
 #       np.savetxt(filNam+'frp_all_filts.csv', exportCSV, delimiter=",")

    datIter += 1
