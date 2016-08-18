# Open source fire detection algorithm

The algorithm uses MODIS version 005 data and the Giglio 2003 defaults for values in thresholding and masking.

## Executing on [ALICE](http://www2.le.ac.uk/offices/itservices/ithelp/services/hpc/alice). 

Log into Alice and then on the command line

    chmod +x process.py
    module load python/2.7.9 gdal/1.11.4 R
    make
    qsub q.sub