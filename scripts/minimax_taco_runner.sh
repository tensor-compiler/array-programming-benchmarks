#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

out=minmax-bench/taco

mkdir -p "$out"

for i in {1..5..2} 
do
	csvout="$out/result-taco-minmax$i.csv"
	LANKA=ON MINMAX_ORDER="$i" TACO_CONCRETIZE_HACK=1 TACO_TENSOR_PATH="data/" TACO_OUT="$csvout" make -j8 taco-bench BENCHES="bench_minimax"
done
