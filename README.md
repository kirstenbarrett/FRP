# Open source fire detection algorithm

The algorithm uses MODIS version 006 data and the Giglio 2016 defaults for values in thresholding and masking.

## Executing on [ALICE](http://www2.le.ac.uk/offices/itservices/ithelp/services/hpc/alice). 

Log into Alice and then on the command line

    chmod +x process.py
    module load python/2.7.9 gdal/1.11.4 R
    make
    qsub q.sub
    
# TODO

Incorporate [meanMadFilt valid neighbours](https://github.com/kirstenbarrett/FRP/commit/531276ba7482d6e6111ee22df5d1f53f59e9a543).

Incorporate [cloud masking thresholds](https://github.com/kirstenbarrett/FRP/commit/33ada1ec87cdcf5a704a873dcfc9d13ada3b2f0d).