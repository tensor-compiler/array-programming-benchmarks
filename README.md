# Setup

Create a virtualenv (`python3 -m venv venv`) and activate it. Install requirements with `pip install -r requirements.txt`. Add benchmarks as python functions prefixed with `test_` and have the function take in the `benchmark` fixture. An example is in `numpy/windowing.py`. 

To run the benchmarks, use `make python-bench` to run all `python` benchmarks. Use the `BENCHES` flag to choose a particular set of benchmarks to run.

To write C++ benchmarks, we use the `google/benchmark` repository. Look at the example of `taco/windowing.cpp` for an example benchmark. Run `make taco-bench` to compile the `taco` benchmarks.
