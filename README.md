## FTP Download of HDFs

To programatically download HDFs via FTP use the `hdf_ftp.py` program. 

You can order specific HDFs by filling out the form which is located at the following URL <https://ladsweb.nascom.nasa.gov/data/search.html>.

The order number which is obtained from the form is required as a program argument. You can check all current orders in your browser by using the following URL <ftp://ladsweb.nascom.nasa.gov/orders/>.

For program usage instructions use option `-h` on the command line.

## Running FRP on OSX

Install **command line tools** `xcode-select –install`

Install **homebrew** `ruby -e “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)”`

Install **python** `brew install python`

Install **osgeo** `brew install gdal --with-complete --with-unsupported --with-hdf4`

Install **scipy** `pip install scipy`

Install **numpy** `pip install numpy`
