#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

source /data/scratch/rohany/array-programming-benchmarks/venv/bin/activate

out=image-bench/

mkdir -p "$out"
mkdir -p "data/image/tensors"

jsonout="$out/image-bench-statistics-all.json"

LANKA=ON NUMPY_JSON="$jsonout" make python-bench BENCHES="numpy/image.py::bench_edge_detection_statistics"
