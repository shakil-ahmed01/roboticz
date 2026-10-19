from __future__ import annotations

import pytest

from smartwash.ai.estimator import CycleEstimator
from smartwash.ai.registry import ModelBundle


@pytest.fixture(scope="session")
def bundle() -> ModelBundle:
    """Fully trained models (a smaller recommender than the default), shared by all tests."""
    return ModelBundle.train(recommender_samples=6000, fault_cycles=300, seed=0)


@pytest.fixture(scope="session")
def recommender(bundle):
    return bundle.recommender


@pytest.fixture(scope="session")
def detector(bundle):
    return bundle.detector


def copy_of(bundle: ModelBundle) -> ModelBundle:
    """The shared models with fresh calibration, safe to retrain or calibrate in a test."""
    return ModelBundle(bundle.recommender, bundle.detector, CycleEstimator(), dict(bundle.metadata))


@pytest.fixture
def models(bundle) -> ModelBundle:
    return copy_of(bundle)


@pytest.fixture(scope="session")
def model_dir(bundle, tmp_path_factory):
    """The shared models saved to disk, for code that loads them itself."""
    path = tmp_path_factory.mktemp("models")
    copy_of(bundle).save(path)
    return path


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
