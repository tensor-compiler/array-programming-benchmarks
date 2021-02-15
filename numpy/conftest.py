import pytest
@pytest.fixture
def tacoBench(benchmark):
    def f(func):
        # Take statistics based on 10 rounds.
        benchmark.pedantic(func, rounds=10, iterations=1)
    return f
