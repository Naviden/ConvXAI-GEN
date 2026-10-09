"""Immutable evidence schema. No model or UI dependencies."""
from dataclasses import dataclass
import hashlib
import math

SCHEMA_VERSION = "2"

@dataclass(frozen=True)
class Feature:
    id: str
    label: str
    score: float

@dataclass(frozen=True)
class Evidence:
    features: tuple[Feature, ...]
    dataset: str
    model: str
    explainer: str
    scope: str
    instance: int | None = None
    target: str = ""
    scale: str = "explainer output (legacy scale unspecified)"

    def __post_init__(self):
        ids = [f.id for f in self.features]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Evidence requires unique, nonempty features")
        if any(not math.isfinite(f.score) for f in self.features):
            raise ValueError("Non-finite attribution")

    def normalised(self):
        total = math.fsum(abs(f.score) for f in self.features)
        if not math.isfinite(total):
            raise ValueError("Attribution norm overflow")
        return {f.id: f.score / total if total else 0.0 for f in self.features}

    def selected(self, k=10):
        if k < 1:
            raise ValueError("top_k must be positive")
        return tuple(sorted(self.features, key=lambda f: (-abs(f.score), f.id))[:k])


def feature_id(label):
    return "f" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
