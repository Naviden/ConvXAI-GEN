"""CSV evidence adapter; no pandas index guessing or silent feature omission."""
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from .cache import ByteLRU, CacheKey, digest, file_hash
from .evidence import Evidence, Feature, feature_id

MODELS = ("random_forest", "xgboost", "neural_network", "svm")
DATASETS = ("iris", "winequality")
MODEL_FILES = {"random_forest": "random_forest", "xgboost": "xgb", "neural_network": "neural_network", "svm": "svm"}
META = {"Istanza", "predicted_class", "target_class", "output_scale", "global_importance"}

class MissingEvidence(ValueError): pass


class CSVAdapter:
    def __init__(self, root, cache=None):
        self.root = Path(root).resolve()
        self.cache = cache or ByteLRU()

    def load(self, dataset, model, explainer="shap", scope="local", instance=None):
        if dataset not in DATASETS or model not in MODELS or explainer not in ("shap", "lime") or scope not in ("local", "global"):
            raise MissingEvidence("Unsupported configuration")
        if scope == "local" and (type(instance) is not int or instance < 0):
            raise MissingEvidence("An explicit nonnegative instance is required")
        path = self.root / "explanations" / dataset / model / f"{explainer}_{scope}_exp.csv"
        aggregate = False
        if not path.exists() and scope == "global" and explainer == "lime":
            path = path.with_name("lime_local_exp.csv")
            aggregate = True
        if not path.exists(): raise MissingEvidence("Generate missing explanation artefacts first")
        model_path = self.root / "models" / dataset / f"{MODEL_FILES[model]}_model_{dataset}.joblib"
        dataset_path = self.root / "DATA" / ("Iris.csv" if dataset == "iris" else "winequality.csv")
        manifest_path = self.root / "explanations" / dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        sources = {"explanation": file_hash(path), "data": file_hash(dataset_path) if dataset_path.exists() else "absent",
                   "manifest": file_hash(manifest_path) if manifest_path.exists() else "legacy"}
        model_hash = file_hash(model_path) if model_path.exists() else "absent"
        recorded = manifest.get(model, {})
        if recorded and ((model_path.exists() and recorded.get("model_hash") != model_hash) or recorded.get("data_hash") != sources["data"] or recorded.get("explanations", {}).get(path.name) != sources["explanation"]):
            raise MissingEvidence("Explanation provenance is stale; regenerate artefacts")
        if recorded and not model_path.exists():
            model_hash = recorded["model_hash"]
        # Content hashes are re-read: old legacy files are supported but explicitly unverified.
        key = CacheKey(dataset, model_hash, explainer, scope, digest(instance), source_hash=digest(sources))
        value, hit = self.cache.get_or_load(dataset + "/" + key.digest,
            lambda: self._read(path, dataset, model, explainer, scope, instance, aggregate))
        return value, key, hit

    @staticmethod
    def _read(path, dataset, model, explainer, scope, instance, aggregate=False):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if len(header) != len(set(header)): raise MissingEvidence("Duplicate CSV columns")
            rows = list(reader)
        if scope == "local":
            rows = [r for r in rows if r.get("Istanza") in (str(instance), f"Instance_{instance}")]
        if not rows or (not aggregate and len(rows) != 1):
            raise MissingEvidence("Expected exactly one evidence row")
        features = []
        for name in header:
            if not name or name.startswith("Unnamed:") or name in META: continue
            try:
                values = [float(r[name]) for r in rows]
            except (ValueError, TypeError):
                raise MissingEvidence("Non-numeric attribution column: " + name)
            if not all(math.isfinite(v) for v in values): raise MissingEvidence("Non-finite score")
            score = math.fsum(abs(v) for v in values) / len(values) if aggregate else values[0]
            features.append(Feature(feature_id(name), name, score))
        target = rows[0].get("target_class", rows[0].get("predicted_class", "")) if scope == "local" else "aggregate"
        scale = rows[0].get("output_scale") or ("mean absolute attribution" if scope == "global" else "legacy explainer output; scale unspecified")
        return Evidence(tuple(features), dataset, model, explainer, scope, instance, target, scale)

    def what_if(self, dataset, model, instance, changes, trust_models=False):
        """Prediction on a modified input, not counterfactual optimisation."""
        if not trust_models: raise MissingEvidence("Enable trusted model loading explicitly")
        if dataset not in DATASETS or model not in MODELS or type(instance) is not int or instance < 0:
            raise MissingEvidence("Invalid what-if request")
        if not isinstance(changes, dict) or not changes: raise MissingEvidence("Specify feature changes")
        import joblib
        from load_data import load_dataset_as_sklearn_format
        data_path = self.root / "DATA" / ("Iris.csv" if dataset == "iris" else "winequality.csv")
        data = load_dataset_as_sklearn_format(str(data_path))
        if instance >= len(data["X_test"]): raise MissingEvidence("Instance outside test split")
        original = data["X_test"].iloc[[instance]].copy()
        changed = original.copy()
        for feature, value in changes.items():
            if feature not in original.columns or type(value) not in (int, float) or not math.isfinite(value):
                raise MissingEvidence("Unknown feature or non-finite value")
            changed[feature] = changed[feature].astype(float)
            changed.loc[:, feature] = value
        path = self.root / "models" / dataset / f"{MODEL_FILES[model]}_model_{dataset}.joblib"
        info = joblib.load(path)
        estimator = info["model"]
        if "scaler" in info:
            original = info["scaler"].transform(original)
            changed = info["scaler"].transform(changed)
        labels = data["target_names"]
        before, after = estimator.predict(original)[0], estimator.predict(changed)[0]
        return {"before": str(labels[int(before)]), "after": str(labels[int(after)]), "changes": changes}
