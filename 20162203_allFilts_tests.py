from scipy import ndimage
import numpy as np
import os
from osgeo import gdal

#Geometric settings for sampling
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


#OPEN INPUT BANDS
os.chdir('/smb/kb308_uol.le.ac.uk_root/Research/ZAB/kb308/MODIS/FPR_files_2004_180to185')
filList = os.listdir('.')
filNam = 'MOD021KM.A2004182.0730.005.'
bands = ['BAND1','BAND2','BAND7','BAND21','BAND22','BAND31','BAND32','LANDMASK','SolarZenith','SolarAzimuth','SensorZenith','SensorAzimuth','LAT','LON']

allArrays = {}
for b in bands:
    fullFilName = filNam + b + '.tif'
    ds = gdal.Open(fullFilName)
    data = np.array(ds.GetRasterBand(1).ReadAsArray())
#    data = data[1472:1546,566:656] #BOUNDARY FIRE AREA
#    data = data[1105:2029,84:1065] #BOREAL AK AREA
    
    if b == 'BAND21' or b == 'BAND22' or b == 'BAND31' or b == 'BAND32':
#        data = np.int_(np.rint(data))
        data = data
    if b == 'BAND1' or b == 'BAND2' or b == 'BAND7':
        b = b + 'x1k'
        data = np.int_(np.rint(data*1000))

    allArrays[b] = data

[nRows,nCols] = np.shape(allArrays['BAND21'])

###DAY/NIGHT FLAG
##dayFlag = np.zeros((nRows,nCols),dtype=np.int)
##dayFlag[np.where(allArrays['SolarZenith'] < 8500)] = 1
##
###CREATE WATER MASK
##waterMask = np.zeros((nRows,nCols),dtype=np.int)
##waterMask[np.where(allArrays['LANDMASK']!=1)] = waterFlag
##
###CREATE CLOUD MASK (SET DATATYPE)
##cloudMask =np.zeros((nRows,nCols),dtype=np.int)
##cloudMask[((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>900)] = cloudFlag
##cloudMask[(allArrays['BAND32']<265)] = cloudFlag
##cloudMask[((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>700)&(allArrays['BAND32']<285)] = cloudFlag
##
###MASK CLOUDS AND WATER FROM INPUT BANDS
##b21CloudWaterMasked = np.copy(allArrays['BAND21'])
##b21CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
##b21CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag
##
##b22CloudWaterMasked = np.copy(allArrays['BAND22'])
##b22CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
##b22CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag
##
##b31CloudWaterMasked = np.copy(allArrays['BAND31'])
##b31CloudWaterMasked [np.where(waterMask == waterFlag)] = waterFlag
##b31CloudWaterMasked [np.where(cloudMask == cloudFlag)] = cloudFlag
##
##deltaT = np.abs(allArrays['BAND21'] - allArrays['BAND31'])
##deltaTCloudWaterMasked = np.copy(deltaT)
##deltaTCloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
##deltaTCloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag
##
############################
####AFTER ALL THE DATA HAVE BEEN READ IN
############################
##
##bgMask = np.zeros((nRows,nCols),dtype=np.int)
##
##with np.errstate(invalid='ignore'):
##    bgMask[np.where((dayFlag == 1) & (allArrays['BAND21'] > (325*reductionFactor)) & (deltaT > (20*reductionFactor)))] = bgFlag
##    bgMask[np.where((dayFlag == 0) & (deltaT > (310*reductionFactor))& (deltaT >(10*reductionFactor)))] = bgFlag
##
##b21bgMask = np.copy(b21CloudWaterMasked)
##b21bgMask[np.where(bgMask == bgFlag)] = bgFlag
##
##b22bgMask = np.copy(b22CloudWaterMasked)
##b22bgMask[np.where(bgMask == bgFlag)] = bgFlag
##
##b31bgMask = np.copy(b31CloudWaterMasked)
##b31bgMask[np.where(bgMask == bgFlag)] = bgFlag
##
##deltaTbgMask = np.copy(deltaTCloudWaterMasked)
##deltaTbgMask[np.where(bgMask == bgFlag)] = bgFlag
##
######################################################################################
###### MEAN AND MAD FILTERS (MAD NEEDED FOR CONFIDENCE ESTIMATION)
######################################################################################
##
##b21meanFilt,b21MADfilt = wakelinMeanMADFilter(b21bgMask,maxKsize,minKsize)
##b31meanFilt,b31MADfilt = wakelinMeanMADFilter(b21bgMask,maxKsize,minKsize)
##deltaTmeanFilt, deltaTMADFilt = wakelinMeanMADFilter(deltaTbgMask, maxKsize, minKsize)
##
