BENCHES := ""
BENCHFLAGS := #"--benchmark-group-by=func"

# Pytest Specific Flags
#IGNORE := numpy/image.py
IGNORE += taco 
IGNORE_FLAGS := $(addprefix --ignore=,$(IGNORE)) 

benches_name := $(patsubst %.py,%,$(BENCHES))
benches_name := $(subst /,_,$(benches_name))
benches_name := $(subst *,_,$(benches_name))
NUMPY_JSON ?= results/numpy/$(benches_name)benches_$(shell date +%Y_%m_%d_%H%M%S).json
NUMPY_JSON := $(NUMPY_JSON)

# Taco Specific Flags
TACO_OUT = results/taco/$(benches_name)benches_$(shell date +%Y_%m_%d_%H%M%S).csv

# Set GRAPHBLAS=ON if compiling GraphBLAS benchmarks.
ifeq ($(GRAPHBLAS),)
GRAPHBLAS := "OFF"
endif
# Set OPENMP=ON if compiling TACO with OpenMP support.
ifeq ($(OPENMP),)
OPENMP := "OFF"
endif
# Set LANKA=ON if compiling on the MIT Lanka cluster.
ifeq ($(LANKA),)
LANKA := "OFF"
endif

ifeq ("$(LANKA)","ON")
CMD := OMP_PROC_BIND=true LD_LIBRARY_PATH=taco/build/lib/:$(LD_LIBRARY_PATH) numactl -C 0,2,4,6,8,10,24,26,28,30,32,34 -m 0 taco/build/taco-bench $(BENCHFLAGS)
else
CMD := LD_LIBRARY_PATH=taco/build/lib/:$(LD_LIBRARY_PATH) taco/build/taco-bench $(BENCHFLAGS)
endif

export TACO_TENSOR_PATH = data/

# To group benchmark output by benchmark, use BENCHFLAGS=--benchmark-group-by=func.
# To additionally group by a parameterized value, add on ",param:<paramname>" to the
# command above.
python-bench: results numpy/*.py
	echo $(benches_name)
	-pytest $(IGNORE_FLAGS) --benchmark-json=$(NUMPY_JSON) $(BENCHFLAGS) $(BENCHES) 
	python numpy/converter.py --json_name $(NUMPY_JSON)

# Separate target to run the python benchmarks with numpy-taco cross validation logic.
validate-python-bench: numpy/*.py validation-path
	pytest $(IGNORE_FLAGS) $(BENCHFLAGS) $(BENCHES) 
	
.PHONY: convert-csv-all
convert-csv-all:
	python numpy/converter.py --all

taco-bench: taco/build/taco-bench
ifeq ($(BENCHES),"")
	$(CMD) --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)" --benchmark_repetitions=10 --benchmark_counters_tabular=true
else
	$(CMD) --benchmark_filter="$(BENCHES)" --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)" --benchmark_repetitions=10 --benchmark_counters_tabular=true
endif

# Separate target to run the TACO benchmarks with numpy-taco cross validation logic.
validate-taco-bench: taco/build/taco-bench validation-path
ifeq ($(BENCHES),"")
	$(CMD) --benchmark_repetitions=1
else
	$(CMD) --benchmark_filter="$(BENCHES)" --benchmark_repetitions=1
endif

.PHONY: validation-path
validation-path:
ifeq ($(VALIDATION_OUTPUT_PATH),)
	$(error VALIDATION_OUTPUT_PATH is undefined)
endif

taco/build/taco-bench: results check-and-reinit-submodules taco/benchmark/googletest
	mkdir -p taco/build/ && cd taco/build/ && cmake -DOPENMP=$(OPENMP) -DGRAPHBLAS=$(GRAPHBLAS) -DLANKA=$(LANKA) ../ && $(MAKE) taco-bench 

taco/benchmark/googletest: check-and-reinit-submodules
	if [ ! -d "taco/benchmark/googletest" ] ; then git clone https://github.com/google/googletest taco/benchmark/googletest; fi

.PHONY: results
results:
	mkdir -p results/taco
	mkdir -p results/numpy

.PHONY: check-and-reinit-submodules
check-and-reinit-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi
