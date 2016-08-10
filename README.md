# FRP Cython

## Executing on [ALICE](http://www2.le.ac.uk/offices/itservices/ithelp/services/hpc/alice). 

Log into Alice and then on the command line

    chmod +x process.py
    module load python/2.7.9 gdal/1.11.4 R
    make
    qsub q.sub