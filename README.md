# Open source fire detection algorithm

The algorithm uses MODIS version 005 data and the Giglio 2003 defaults for values in thresholding and masking.

## Downloading of HDF orders

In order to get HDF data one has to fill out an order form at <https://ladsweb.nascom.nasa.gov/data/search.html>. 

Upon ordering you will receive a number which is required to download the order.

If you have wget available you can simply run;

    wget -r -N -c ladsweb.nascom.nasa.gov/orders/{ORDER NUMBER}

You can also download HDFs via FTP using the `hdf_ftp.py` program.
 
It can be used as both a standalone module, or you can use it directly when executing `frp.py`. 

For help please run `python hdf_ftp.py -h`.

## Validation of HDF orders

In order to validate the HDFs in your downloaded order one can use the `cksum.py` script.

For help please run `python cksum.py -h`.

## Running FRP on OSX

Install **command line tools** `xcode-select -–install`

Install **homebrew** `ruby -e “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)”`

Install **python** `brew install python`

Install **osgeo** `brew install gdal --with-complete --with-unsupported --with-hdf4`

Install **scipy** `pip install scipy`

Install **numpy** `pip install numpy`
