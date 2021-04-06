#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

source venv/bin/activate

sspath=data/suitesparse
# out=suitesparse-ufunc-bench/taco
out=suitesparse-ufunc-bench/numpy

mkdir -p "$out"

while read line; do
	matrix="$sspath/$line/$line.mtx"
	# csvout="$out/result-$line.csv"
	# LANKA=ON SUITESPARSE_TENSOR_PATH="$matrix" TACO_OUT="$csvout" make -j8 taco-bench BENCHES="bench_suitesparse_ufunc"
	jsonout="$out/result-$line.json"
	LANKA=ON SUITESPARSE_TENSOR_PATH="$matrix" NUMPY_JSON="$jsonout" make python-bench BENCHES="numpy/ufuncs.py::bench_pydata_suitesparse_ufunc_sparse"
done <$1

# for path in $sspath/*; do
#     if [ ! -d $path ]; then
#         continue
#     fi
#     name="$(cut -d'/' -f3 <<< "$path")"
#     matrix="$path/$name.mtx"
# 
#     csvout="$out/result-$name.csv"
# 
#     LANKA=ON SUITESPARSE_TENSOR_PATH="$matrix" TACO_OUT="$csvout" make -j8 taco-bench BENCHES="bench_suitesparse_ufunc"
# done
