## FTP Download of HDFs

To programatically download HDFs via FTP use `hdf_ftp.py` program. For usage instructions use option `-h` on the command line.

## How to setup

### Mac

1) Install **homebrew** using the terminal command `ruby -e “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)”`. This requires xcode command line tools which you can get by using the terminal command `xcode-select –install`.

2) Install **python** using the terminal command `brew install python`. _Ensure python is version 2.7.11 (the highest stable build)_.

    This will also install **pip** which is needed to install the required python dependencies. 
    Ensure it is fully updated using the terminal command `pip install --upgrade pip`.

3) Install python dependencies;

    **osgeo** using the terminal command `brew install gdal --with-complete --with-unsupported --with-hdf4`.

    **scipy** using the terminal command `pip install scipy`.

    **numpy** using the terminal command `pip install numpy`.


## How to execute

### Mac

In order to run the project ensure that you use the homebrew version of python which can be found at `/usr/local/Cellar/python/2.7.11/bin/python`, _(assuming you did not change the default install directory)_.



