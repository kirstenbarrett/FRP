import argparse
import subprocess
import os

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Order checksum file", type=str, required=True)
args = parser.parse_args()

# Read the checksum file
with open(args.file) as f:
  validated = f.readlines()

# Filter the checksum file
filtered = []
for v in validated:
  if ".hdf" in v:
    filtered.append(v)
validated = filtered

# Invalid and missing lists
invalid = []
missing = []

# For each entry in the validated list
for v in validated:

  # Get the checksum, size and filename
  parts = v.split()
  vChecksum = parts[0]
  vSize = parts[1]
  vFile = parts[2]

  # The file is not present
  if not os.path.isfile(vFile):

    # Add it to the missing list
    missing.append(vFile)
    continue

  # Get the cksum output of the downloaded file
  output = subprocess.check_output("cksum " + vFile, shell=True)
  parts = output.split()
  checksum = parts[0]
  size = parts[1]

  # The checksum and size match - file validated
  if checksum == vChecksum and size == vSize:

    print vFile + " verified"

  # The the file must be invalid
  else:

    invalid.append(vFile)

# Output any results
print "\n" + str(len(invalid)) + " corrupt files found - please redownload and rerun"

for f in invalid:
  print f

print "\n" + str(len(missing)) + " missing files detected - please download and rerun"

for f in missing:
  print f