#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

out=frostt-ufunc-bench/taco/

mkdir -p "$out"

for tensor in "nips" "uber-pickups" "chicago-crime" "enron" "nell-2" "vast"; do
	csvout="$out/result-$tensor.csv"
	LANKA=ON TACO_OUT="$csvout" make -j8 taco-bench BENCHES="bench_frostt_ufunc/$tensor*"
done
