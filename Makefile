BENCHES :=
BENCHFLAGS :=

python-bench: numpy/*.py
	pytest $(BENCHFLAGS) $(BENCHES)

