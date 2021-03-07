#!/bin/bash

set -e

TENSOR_NAMES=(
  "chicago-crime"
  "lbnl-network"
  "nips"
  "uber-pickups"
)

# The tensor format from FROSTT is not self describing. Prepend
# we'll prepend information about the dimensions to make it so.
# This array should stay in line with the other arrays. In particular
# it should contain the size of each dimension in the tensor, followed
# by the number of non-zeros.
TENSOR_DIMS=(
  "6186 24 77 32 5330673"
  "1605 4198 1631 4209 868131 1698825"
  "2482 2862 14036 17 3101609"
  "183 24 1140 1717 3309490"
)

TENSOR_URLS=(
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/chicago-crime/comm/chicago-crime-comm.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/lbnl-network/lbnl-network.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nips/nips.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/uber-pickups/uber.tns.gz"
)

mkdir -p data/FROSTT

for i in ${!TENSOR_URLS[@]}; do
    name=${TENSOR_NAMES[$i]}
    dim=${TENSOR_DIMS[$i]}
    url=${TENSOR_URLS[$i]}
    outdir="data/FROSTT/$name"
    if [ -d "$outdir" ]; then
        continue
    fi
    echo "Downloading tensor $name to $outdir"
    mkdir "$outdir"
    # Write the dimension to the file.
    echo "$dim" > "$outdir/tensor.frostt"
    # Append the rest of the data to the file.
    curl $url | gzip -d -c >> "$outdir/tensor.frostt"
done
