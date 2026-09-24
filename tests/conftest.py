import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True)
def limit_test_blas_threads():
    with threadpool_limits(limits=1):
        yield
