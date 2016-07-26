## FTP Download of HDFs

To programatically download HDFs via FTP use `hdf_ftp.py` program. For usage instructions use option `-h` on the command line.

## Use on ALICE

Log into ALICE (remember to use the -X option on ssh) and load the following modules

    module load python/2.7.9
    module load gdal
    module load R
    
Make the code

    make

Then submit a job

    qsub frp_alice.sub

Once this has finished, run the post processing script 

    R
    source("postproc.r")

# Serial profiling

Ensure that you have Cython, GDAL, numpy and scipy available. Type 

    make

and then 

    ./doprof.py
