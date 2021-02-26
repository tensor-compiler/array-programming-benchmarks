#ifndef TACO_BENCH_BENCH_H
#define TACO_BENCH_BENCH_H

#include "benchmark/benchmark.h"

// Register a benchmark with the following options:
// * Millisecond output display
// * 10 data points
// * Reporting of avg/stddev/median
// * Wall-clock time, rather than CPU time.
#define TACO_BENCH(bench)         \
  BENCHMARK(bench)                \
  ->Unit(benchmark::kMillisecond) \
  ->Repetitions(10)               \
  ->Iterations(5)                 \
  ->ReportAggregatesOnly(true)    \
  ->UseRealTime()

// TACO_BENCH_ARG is similar to TACO_BENCH but allows for passing
// of an arbitrarily typed argument to the benchmark function.
// TODO (rohany): Make this take in only 1 argument.
// TODO (rohany): Don't specify the time here, but do it at the command line.
#define TACO_BENCH_ARG(bench, name, arg) \
  BENCHMARK_CAPTURE(bench, name, arg)    \
  ->Unit(benchmark::kMicrosecond)        \
  ->Repetitions(10)                      \
  ->Iterations(5)                        \
  ->ReportAggregatesOnly(true)           \
  ->UseRealTime()

#endif //TACO_BENCH_BENCH_H
