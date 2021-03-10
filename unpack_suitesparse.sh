#!/bin/bash

cd data/suitesparse/

for f in *.tar.gz; do
    tar -xvf "$f" --strip=1
done
