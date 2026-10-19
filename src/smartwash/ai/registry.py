"""Saving, loading and (re)training the bundle of models the controller uses."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import sklearn

from .. import __version__
from .anomaly import FaultDetector
from .estimator import CycleEstimator
from .feedback import FeedbackRecord
from .recommender import CycleRecommender
from .training import train_fault_detector, train_recommender

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
RECOMMENDER_FILE = "recommender.joblib"
DETECTOR_FILE = "fault_detector.joblib"
CALIBRATION_FILE = "calibration.json"
METADATA_FILE = "metadata.json"


class IncompatibleModelError(RuntimeError):
    """Saved models were made by a different library or format version and must be retrained."""


def _write_json(path: Path, data: Any) -> None:
    """Write atomically, so a crash or a concurrent writer never leaves a half-written file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_calibration(model_dir: Path) -> CycleEstimator:
    path = Path(model_dir) / CALIBRATION_FILE
    if not path.exists():
        return CycleEstimator()
    return CycleEstimator(json.loads(path.read_text(encoding="utf-8")))


class ModelBundle:
    def __init__(
        self,
        recommender: CycleRecommender,
        detector: FaultDetector,
        estimator: CycleEstimator | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.recommender = recommender
        self.detector = detector
        self.estimator = estimator or CycleEstimator()
        self.metadata = metadata or {}

    @classmethod
    def train(
        cls,
        *,
        recommender_samples: int = 20_000,
        fault_cycles: int = 300,
        feedback: Iterable[FeedbackRecord] = (),
        seed: int = 0,
        estimator: CycleEstimator | None = None,
    ) -> ModelBundle:
        log.info("Training cycle recommender on %d synthetic loads...", recommender_samples)
        recommender = train_recommender(n_samples=recommender_samples, seed=seed, feedback=feedback)
        log.info("Training fault detector on %d simulated cycles...", fault_cycles)
        detector = train_fault_detector(n_cycles=fault_cycles, seed=seed)
        metadata = {
            "format_version": FORMAT_VERSION,
            "smartwash_version": __version__,
            "sklearn_version": sklearn.__version__,
            "trained_at": time.time(),
            "seed": seed,
            "recommender": {"version": recommender.version, **recommender.metrics},
            "fault_detector": detector.metrics,
        }
        return cls(recommender, detector, estimator, metadata)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.recommender, model_dir / RECOMMENDER_FILE, compress=3)
        joblib.dump(self.detector, model_dir / DETECTOR_FILE, compress=3)
        self.save_calibration(model_dir)
        _write_json(model_dir / METADATA_FILE, self.metadata)

    def save_calibration(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_json(model_dir / CALIBRATION_FILE, self.estimator.to_dict())

    @classmethod
    def load(cls, model_dir: str | Path) -> ModelBundle:
        """Load saved models. Only load model files you created: joblib files can run code."""
        model_dir = Path(model_dir)
        metadata_path = model_dir / METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"no trained models in {model_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != FORMAT_VERSION:
            raise IncompatibleModelError("model format changed")
        if metadata.get("sklearn_version") != sklearn.__version__:
            raise IncompatibleModelError(
                f"models were trained with scikit-learn {metadata.get('sklearn_version')}, "
                f"running {sklearn.__version__}"
            )
        recommender = joblib.load(model_dir / RECOMMENDER_FILE)
        detector = joblib.load(model_dir / DETECTOR_FILE)
        return cls(recommender, detector, load_calibration(model_dir), metadata)

    @classmethod
    def load_or_train(
        cls,
        model_dir: str | Path,
        *,
        auto_train: bool = True,
        feedback: Iterable[FeedbackRecord] = (),
        **train_kwargs: Any,
    ) -> ModelBundle:
        """Load saved models, training (and saving) a fresh set when there are none."""
        model_dir = Path(model_dir)
        try:
            return cls.load(model_dir)
        except FileNotFoundError:
            if not auto_train:
                raise
            log.info("No trained models in %s; training them now (about a minute).", model_dir)
        except IncompatibleModelError as exc:
            if not auto_train:
                raise
            log.warning("Retraining models: %s.", exc)
        bundle = cls.train(feedback=feedback, estimator=load_calibration(model_dir), **train_kwargs)
        bundle.save(model_dir)
        return bundle

    def retrain_recommender(
        self, feedback: Iterable[FeedbackRecord], *, n_samples: int | None = None, seed: int = 0
    ) -> dict[str, Any]:
        """Retrain the recommender with the latest feedback; returns its evaluation metrics."""
        samples = n_samples or int(
            self.metadata.get("recommender", {}).get("test_samples", 4000) / 0.2
        )
        recommender = train_recommender(n_samples=samples, seed=seed, feedback=feedback)
        self.recommender = recommender
        self.metadata["recommender"] = {"version": recommender.version, **recommender.metrics}
        return self.metadata["recommender"]
