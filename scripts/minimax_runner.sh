#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

source /data/scratch/rohany/array-programming-benchmarks/venv/bin/activate

out=minmax-bench/numpy

mkdir -p "$out"
mkdir -p "data/minmax"

jsonout="$out/statistics-numpy.json"

LANKA=ON NUMPY_JSON="$jsonout" make python-bench BENCHES="numpy/minmax.py::bench_pydata_minmax" 
