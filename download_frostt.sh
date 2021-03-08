#!/bin/bash

set -e

TENSOR_NAMES=(
  "chicago-crime"
  "lbnl-network"
  "nips"
  "uber-pickups"
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
    url=${TENSOR_URLS[$i]}
    out="data/FROSTT/$name.tns"
    if [ -f "$out" ]; then
        continue
    fi
    echo "Downloading tensor $name to $out"
    curl $url | gzip -d -c > "$out"
done
