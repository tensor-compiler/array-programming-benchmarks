BENCHES := ""
BENCHFLAGS :=

python-bench: numpy/*.py
	pytest $(BENCHFLAGS) $(BENCHES)

taco-bench: build-taco-bench
ifeq ($(BENCHES),"")
	taco/build/taco-bench $(BENCHFLAGS)
else
	taco/build/taco-bench $(BENCHFLAGS) --benchmark_filter="$(BENCHES)"
endif

build-taco-bench: check-and-reinit-submodules taco/build taco/build/taco-bench
	$(MAKE) -C taco/build taco-bench

taco-bench-build: taco/build
	mkdir taco/build
	cd taco/build
	cmake ../
	cd ../../

bench-gtest: taco/benchmark/googletest/README.md
	git clone https://github.com/google/googletest taco/benchmark/googletest

.PHONY: check-and-reinit-submodules
check-and-reinit-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi
