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

# Taco Specific Flags
TACO_OUT = results/taco/$(benches_name)benches_$(shell date +%Y_%m_%d_%H%M%S).csv

GRAPHBLAS := "OFF"
OPENMP := "OFF"

export TACO_TENSOR_PATH = data/

# To group benchmark output by benchmark, use BENCHFLAGS=--benchmark-group-by=func.
# To additionally group by a parameterized value, add on ",param:<paramname>" to the
# command above.
python-bench: results numpy/*.py
	echo $(benches_name)
	pytest $(IGNORE_FLAGS) --benchmark-json=$(NUMPY_JSON) $(BENCHFLAGS) $(BENCHES)@
	make convert-csv
	
.PHONY: convert-csv
convert-csv:
	py.test-benchmark compare --csv=$(patsubst %.json,%.csv,$(NUMPY_JSON)) $(NUMPY_JSON)

taco-bench: taco/build/taco-bench
ifeq ($(BENCHES),"")
	LD_LIBRARY_PATH=taco/build/lib/:$(LD_LIBRARY_PATH) taco/build/taco-bench $(BENCHFLAGS) --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)"

else
	LD_LIBRARY_PATH=taco/build/lib/:$(LD_LIBRARY_PATH) taco/build/taco-bench $(BENCHFLAGS) --benchmark_filter="$(BENCHES)" --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)"
endif

taco/build/taco-bench: results check-and-reinit-submodules taco/benchmark/googletest
	mkdir -p taco/build/ && cd taco/build/ && cmake -DOPENMP=$(OPENMP) -DGRAPHBLAS=$(GRAPHBLAS) ../ && $(MAKE) taco-bench 

taco/benchmark/googletest: check-and-reinit-submodules
	if [ ! -d "taco/benchmark/googletest" ] ; then git clone https://github.com/google/googletest taco/benchmark/googletest; fi

.PHONY: csvs
results:
	mkdir -p results/taco
	mkdir -p results/numpy

.PHONY: check-and-reinit-submodules
check-and-reinit-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi
