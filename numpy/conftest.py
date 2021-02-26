import pytest
@pytest.fixture
def tacoBench(benchmark):
    def f(func):
        # Take statistics based on 10 rounds.
        benchmark.pedantic(func, rounds=10, iterations=5)
        # How do i set please use 10 rounds...
        # benchmark(func)
    return f
