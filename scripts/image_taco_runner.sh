#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

out=image-bench/taco

mkdir -p "$out"

for i in {1..98} 
do
	csvout="$out/result-taco-img$i.csv"
	LANKA=ON IMAGE_NUM="$i" TACO_TENSOR_PATH="data/" TACO_OUT="$csvout" make -j8 taco-bench BENCHES="bench_image"
done
