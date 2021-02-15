BENCHES := ""
BENCHFLAGS :=

# To group benchmark output by benchmark, use BENCHFLAGS=--benchmark-group-by=func.
python-bench: numpy/*.py
	pytest --ignore=taco $(BENCHFLAGS) $(BENCHES)

taco-bench: taco/build/taco-bench
ifeq ($(BENCHES),"")
	taco/build/taco-bench $(BENCHFLAGS)
else
	taco/build/taco-bench $(BENCHFLAGS) --benchmark_filter="$(BENCHES)"
endif

taco/build/taco-bench: check-and-reinit-submodules taco/build/Makefile
	$(MAKE) -C taco/build taco-bench

taco/build/Makefile: taco/benchmark/googletest
	mkdir -p taco/build
	cd taco/build/ && cmake ../

taco/benchmark/googletest: check-and-reinit-submodules
	if [ ! -d "taco/benchmark/googletest" ] ; then git clone https://github.com/google/googletest taco/benchmark/googletest; fi

.PHONY: check-and-reinit-submodules
check-and-reinit-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi
