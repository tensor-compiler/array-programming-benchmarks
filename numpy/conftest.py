import pytest
@pytest.fixture
def tacoBench(benchmark):
    def f(func):
        # Take statistics based on 10 rounds.
        benchmark.pedantic(func, rounds=10, iterations=5)
        # How do i set please use 10 rounds...
        # benchmark(func)
    return f

def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", dest="plot", default=False, help="Enable plotting for image.py")

@pytest.fixture
def plot(request):
    return request.config.getoption("--plot") 
