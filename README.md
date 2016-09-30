# Open source fire detection algorithm

The algorithm uses MODIS version 005 data and the Giglio 2003 defaults for values in thresholding and masking.

# Executing on [ALICE](http://www2.le.ac.uk/offices/itservices/ithelp/services/hpc/alice). 

Log into Alice and then on the command line

    chmod +x process.py
    module load python/2.7.9 gdal/1.11.4 R
    make
    qsub q.sub
    
## Array Jobs
    
Currently the `q.sub` file is set up to handle the entire job as a series of jobs collectively known as an array job.

The number of jobs that are synchronously executed is defined by the `-t` flag in the `q.sub` file. This value is then passed to the file `process.py` denoted by the variable `$PSB_ARRAYID`.

### Example

`#PBS -t 0-6`

`./process.py $PBS_ARRAYID`
    
The above would run 7 synchronous jobs numbered 0 through to 6. The path defined for the data [directory](data/README.md) in the `input.txt` file would have to contain seven sub directories, consequently one for each job number.
 
The directory structure for such a job is shown below;

    data-directory/
        0/
        1/
        2/
        3/
        4/
        5/
        6/
   
Each sub-directory would contain data that is proportionally equal in execution time.   
   
## Running single jobs

It is still very easy to run single jobs from a single data directory, whilst maintaining the ability to run array jobs when needed.

One would simply remove the array job flag from the `q.sub` file and manually pass in a value where previously the variable `$PSB_ARRAYID` would have been.

### Example
 
`./process.py 0`
 
 The above would run only a single job numbered 0. The path defined for the data [directory](data/README.md) in the `input.txt` file would have to contain one sub directory named 0.
 
 The directory structure for such a job is shown below;
 
    data-directory/
        0/
 
 ## ALICE specific terminal commands
 
 In order to delete or cancel a running job (for array jobs include the `[]`).
 
     qdel <job-id>
     
 Get the status of a job (C = Cancelled, R = Running)
     
     qstat <job-id>
     
 For job testing and quick bug fixing use the development queue (this will over-ride any queue defined in the `q.sub` file)
     
     qsub q.sub -q devel