#Just needs multi-dimensional footprint
#and the sunglint mask...
from scipy import ndimage
import numpy as np
import os
from osgeo import gdal
from collections import namedtuple

os.chdir('/Users/kirsten/Documents/data/MODIS/Boundary_swaths')
filList = os.listdir('.')
filNam = 'MOD021KM.A2004178.2120.005.'
bands = ['BAND1','BAND2','BAND21','BAND22','BAND31','BAND32','landmask','SolarZenith','LAT','LON']

#READ IN AL REFLECTANCE AND EMITTED BANDS
allArrays = {}
for b in bands:
    fullFilName = filNam + b + '.tif'
    ds = gdal.Open(fullFilName)
    data = np.array(ds.GetRasterBand(1).ReadAsArray())

    if b == 'BAND21' or b == 'BAND22' or b == 'BAND31' or b == 'BAND32':
        data = np.int_(np.rint(data))
    if b == 'BAND1' or b == 'BAND2':
        b = b + 'x1k'
        data = np.int_(np.rint(data*1000))

    allArrays[b] = data

[nRows,nCols] = np.shape(allArrays['BAND21'])

#DAY/NIGHT FLAG
dayFlag = np.zeros((nRows,nCols),dtype=np.int)
dayFlag[np.where(allArrays['SolarZenith'] < 8500)] = 1

#CHANGES LANDMASK TO BINARY (SET DATATYPE) "LAND" = 1
waterMask = np.ones((nRows,nCols),dtype=np.int)
waterMask[np.where(allArrays['landmask']>1)] = 0

#CREATE CLOUD MASK (SET DATATYPE) "CLOUD" = 0
b1plus2gt900=np.ones((nRows,nCols),dtype=np.int)
b1plus2gt900[np.where((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>900)] = 0

b32lt265 = np.ones((nRows,nCols),dtype=np.int)
b32lt265[np.where(allArrays['BAND32']<265)] = 0

##MASK CLOUDS AND WATER
b1plus2gt700=np.ones((nRows,nCols),dtype=np.int)
b1plus2gt700[np.where((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>700)] = 0

b32lt285=np.ones((nRows,nCols),dtype=np.int)
b32lt285[np.where(allArrays['BAND32']<285)]=0

temp=np.ones((nRows,nCols),dtype=np.int)
temp[np.where(b1plus2gt700+b32lt265)==0] = 0

cloudMask = b1plus2gt900*b32lt285*temp

maskWaterCloud = waterMask*cloudMask


b21mask = allArrays['BAND21']*maskWaterCloud
b22mask = allArrays['BAND22']*maskWaterCloud
b31mask = allArrays['BAND31']*maskWaterCloud

deltaT = allArrays['BAND21'] - allArrays['BAND31'] + 1 #ADD ONE SO ALL REAL ZEROS ARE MAINTAINED
deltaT *= maskWaterCloud

##########################
##AFTER ALL THE DATA HAVE BEEN READ IN
##########################

minNcount = 8
minNfrac = 0.25
minKsize = 5
maxKsize = 21
b21saturationVal = 450 #???

bgMask = np.ones((nRows,nCols))
bgMask[np.where((dayFlag == 1) & (b21mask >325))] = 0
bgMask[np.where((dayFlag == 1) & (deltaT >20))] = 0
bgMask[np.where((dayFlag == 0) & (deltaT >310))] = 0
bgMask[np.where((dayFlag == 0) & (deltaT >10))] = 0

b21bgMask = b21mask*bgMask
deltaTbgMask = deltaT *bgMask

###############################
##ALL REQUIRED FUNCTION DEFS
###############################

def makeFootprint(kSize):
    fpZeroLine = (kSize-1)/2
    fpZeroColStart = fpZeroLine-1
    fpZeroColEnd = fpZeroColStart+3
    fp = np.ones((kSize,kSize),dtype = 'int_')
    fp[fpZeroLine,fpZeroColStart:fpZeroColEnd] = 0

    return fp

def meanFilt(kernel):
    bgMean = -1
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -1)) & (centerVal != 0)):
        fpMask = makeFootprint(kSize)
        kFpMask = kernel * fpMask
        nghbrCnt = np.count_nonzero(kFpMask)
        
        if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
            bgMean = np.int_(np.rint(np.mean(kFpMask[np.nonzero(kFpMask)])))
            
    return bgMean

def MADfilt(kernel):
    bgMAD = -1
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -1)) & (centerVal != 0)):
        fpMask = makeFootprint(kSize)
        kFpMask = kernel * fpMask
        nghbrCnt = np.count_nonzero(kFpMask)
        
        if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
            bgMAD = np.int(np.rint(np.mean(abs(centerVal - kFpMask[np.nonzero(kFpMask)]))))
            
    return bgMAD

def nValidFilt(kernel):
    nghbrCnt = -1
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -1)) & (centerVal != 0)):
        fpMask = makeFootprint(kSize)
        kFpMask = kernel * fpMask
        nghbrCnt = np.count_nonzero(kFpMask)
            
    return nghbrCnt

def nRejectFilt(kernel):
    nReject = -1
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if (((kSize == minKsize) | (centerVal == -1)) & (centerVal != 0)):
        fpMask = makeFootprint(kSize)
        kFpMask = kernel * fpMask
        nghbrCnt = np.count_nonzero(kFpMask)
        nReject = np.size(np.where(kFpMask == 0))
            
    return nReject

####################################################################

##########################
##B21 MEAN FILTER
##########################
b21meanFilts = {}
meanFiltBand = b21bgMask #INITIALIZE WITH UNFILTERED BAND
kSize = minKsize

while kSize <=  maxKsize:
    filtName = 'b21meanFilt'+str(kSize)
    meanFiltBand = ndimage.generic_filter(meanFiltBand, meanFilt, kSize)
    b21meanFilts[filtName] = meanFiltBand
    kSize += 2

b21meanFilt = b21meanFilts['b21meanFilt'+str(minKsize)]
kSize = minKsize + 2

while kSize <= maxKsize:
    b21meanFilt[np.where(b21meanFilt == -1)] = b21meanFilts['b21meanFilt'+str(kSize)][np.where(b21meanFilts == -1)]
    kSize += 2

b21minusBG = b21mask - b21meanFilt

###TEST FOR SATURATION IN BAND 21 (INEFFICIENT TO RE-CALCULATE FOR ALL PIXELS)
####if (np.nanmax(b21mask) > b21saturationVal):
####    fireCubeArr = b22mask,deltaT,dayFlag
####    b22bg = ndimage.generic_filter(fireCubeArr, BGcontext, footprint=maxFootprint, extra_arguments = ('mean',cubeDepth,0,minKsize,maxKsize))
####    b22minusBG = b22mask - b22bg
####
####    b21minusBG[(b21mask >= b21SaturationValue)] = b22minusBG[(b21mask >= b21SaturationValue)]

##POTENTIAL FIRE TEST
potFire = np.zeros((nRows,nCols))
potFire[(dayFlag == 1)&(allArrays['BAND21']>310)] = 1
potFire[(dayFlag == 1)&(deltaT>10)] = 1
potFire[(dayFlag == 0) & (allArrays['BAND21']>320)] = 1

# ABSOLUTE THRESHOLD TEST (Kaufman et al. 1998) FOR REMOVING SUNGLINT
absValTest = np.zeros((nRows,nCols))
absValTest[(dayFlag == 1) & (allArrays['BAND21']>360) & (deltaT > 10) & (allArrays['BAND2x1k']<300)] = 1
absValTest[(dayFlag == 0) & (allArrays['BAND21']>305)] = 1

#########################################
#CONTEXT TESTS (GIGLIO ET AL 2003)
#########################################


##CONTEXT FIRE TEST 2:
deltaTmeanFilts = {}
meanFiltBand = deltaTbgMask #INITIALIZE WITH UNFILTERED BAND
kSize = minKsize

while kSize <=  maxKsize:
    filtName = 'deltaTmeanFilt'+str(kSize)
    meanFiltBand = ndimage.generic_filter(meanFiltBand, meanFilt, kSize)
    deltaTmeanFilts[filtName] = meanFiltBand
    kSize += 2

deltaTmeanFilt = deltaTmeanFilts['deltaTmeanFilt'+str(minKsize)]
kSize = minKsize + 2

while kSize <= maxKsize:
    deltaTmeanFilt[np.where(deltaTmeanFilt == -1)] = deltaTmeanFilts['deltaTmeanFilt'+str(kSize)][np.where(deltaTmeanFilt == -1)]
    kSize += 2
    
##deltaT MAD Filtering

deltaTMADFilts = {}
MADFiltBand = deltaTbgMask #INITIALIZE WITH UNFILTERED BAND
kSize = minKsize

while kSize <=  maxKsize:
    filtName = 'deltaTMADFilt'+str(kSize)
    MADFiltBand = ndimage.generic_filter(MADFiltBand, MADfilt, kSize)
    deltaTMADFilts[filtName] = MADFiltBand
    kSize += 2

deltaTMADFilt = deltaTMADFilts['deltaTMADFilt'+str(minKsize)]
kSize = minKsize + 2

while kSize <= maxKsize:
    deltaTMADFilt[np.where(deltaTMADFilt == -1)] = deltaTMADFilts['deltaTMADFilt'+str(kSize)][np.where(deltaTMADFilt == -1)]
    kSize += 2


deltaTMADFilt *= maskWaterCloud
deltaTMADfire = np.zeros((nRows,nCols))
deltaTMADfire[abs(deltaT)>(abs(deltaTmeanFilt) + (3.5*deltaTMADfire))] = 1
deltaTMADFilt *= maskWaterCloud #TWICE??

#CONTEXT FIRE TEST 3
deltaTfire = np.zeros((nRows,nCols))
deltaTfire[np.where(abs(deltaT) > (abs(deltaTmeanFilt)) + 7))] = 1 #CHANGED THRESHOLD FROM 6 to 7 b/c deltaT = deltaT + 1 to avoid masking zeros

#CONTEXT FIRE TEST 4
B21fire = np.zeros((nRows,nCols))

b21MADFilts = {}
MADFiltBand = b21bgMask #INITIALIZE WITH UNFILTERED BAND
kSize = minKsize

while kSize <=  maxKsize:
    filtName = 'b21MADFilt'+str(kSize)
    MADFiltBand = ndimage.generic_filter(MADFiltBand, MADfilt, kSize)
    b21MADFilts[filtName] = MADFiltBand
    kSize += 2

b21MADFilt = b21MADFilts['b21MADFilt'+str(minKsize)]
kSize = minKsize + 2

while kSize <= maxKsize:
    b21MADFilt[np.where(b21MADFilt == -1)] = b21MADFilts['b21MADFilt'+str(kSize)][np.where(b21MADFilt == -1)]
    kSize += 2

B21fire[np.where(b21mask > (b21bg + (3*b21MADFilt)))] = 1

#POTENTIAL FIRE TEST 5

b31meanFilts = {}
meanFiltBand = b31bgMask #INITIALIZE WITH UNFILTERED BAND
kSize = minKsize

while kSize <=  maxKsize:
    filtName = 'b31meanFilt'+str(kSize)
    meanFiltBand = ndimage.generic_filter(meanFiltBand, meanFilt, kSize)
    b31meanFilts[filtName] = meanFiltBand
    kSize += 2

b31meanFilt = b31meanFilts['b31meanFilt'+str(minKsize)]
kSize = minKsize + 2

while kSize <= maxKsize:
    b31meanFilt[np.where(b31meanFilt == -1)] = b31meanFilts['b31meanFilt'+str(kSize)][np.where(b31meanFilts == -1)]
    kSize += 2


b31MADFilts = {}
MADFiltBand = b31bgMask #INITIALIZE WITH UNFILTERED BAND
kSize = minKsize

while kSize <=  maxKsize:
    filtName = 'b31MADFilt'+str(kSize)
    MADFiltBand = ndimage.generic_filter(MADFiltBand, MADfilt, kSize)
    b31MADFilts[filtName] = MADFiltBand
    kSize += 2

b31MADFilt = b31MADFilts['b31MADFilt'+str(minKsize)]
kSize = minKsize + 2

while kSize <= maxKsize:
    b31MADFilt[np.where(b31MADFilt == -1)] = b31MADFilts['b31MADFilt'+str(kSize)][np.where(b31MADFilt == -1)]
    kSize += 2

B31fire = np.zeros((nRows,nCols))
B31fire[np.where(b31mask > (b31meanFilt + b31MADFilt - 4))] = 1 ##MUST BE DAYTIME ONLY

#CONTEXT FIRE TEST 6

rejectedMask = np.zeros((nRows, nCols))
rejectedMask[np.where(bgMask == 0)] = 1
rejB21 = b21mask*rejectedMask
#MADfilter should only look at those actually rejected from calculation, no minimum n of neighbors
def MADrejFilt(kernel):
    bgMAD = -1
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if ((((kSize == minKsize) | (centerVal == -1)) & (centerVal != 0))):
        fpMask = makeFootprint(kSize)
        kFpMask = kernel * fpMask
        nghbrCnt = len(np.where(kFpMask == 0)[1]) #NUMBER OF VALID NEIGHBORS
        
        if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
            bgMAD = np.int(np.rint(np.mean(abs(centerVal - kFpMask[np.nonzero(kFpMask)]))))
            
    return bgMAD

b21rejMADFilts = {}
MADFiltBand = rejB21 #INITIALIZE WITH REJECTED VALS
kSize = minKsize

while kSize <=  maxKsize:
    filtName = 'b21rejMADFilt'+str(kSize)
    MADFiltBand = ndimage.generic_filter(MADFiltBand, MADfilt, kSize)
    b21rejMADFilts[filtName] = MADFiltBand
    kSize += 2

b21rejMADFilt = b21rejMADFilts['b21rejMADFilt'+str(minKsize)]
kSize = minKsize + 2

while kSize <= maxKsize:
    b21rejMADFilt[np.where(b21rejMADFilt == -1)] = b21rejMADFilts['b21rejMADFilt'+str(kSize)][np.where(b21rejMADFilt == -1)]
    kSize += 2

B21rejFire = np.zeros((nRows,nCols))
B21rejFire[np.where(b21rejMADFilt>=5)] = 1

##########################################
##DESERT BOUNDARY TESTS
##########################################
