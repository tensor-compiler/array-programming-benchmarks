import pytest
@pytest.fixture
def tacoBench(benchmark):
    def f(func, extra_info = None, save_ret_val = False):
        # Take statistics based on 10 rounds.
        if extra_info is not None:
            for k, v in extra_info.items():
                benchmark.extra_info[k] = v
        if save_ret_val:
            benchmark.extra_info["return"] = func()
        benchmark.pedantic(func, rounds=10, iterations=1, warmup_rounds=1)
    return f

def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", dest="plot", default=False, help="Enable plotting for image.py")

@pytest.fixture
def plot(request):
    return request.config.getoption("--plot") 
