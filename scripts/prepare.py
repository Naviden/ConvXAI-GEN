"""Rebuild model + SHAP/LIME artefacts with explicit class and scale provenance."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convxai.cache import file_hash
from convxai.adapters import MODEL_FILES


def prepare(dataset, model_name, limit=12, background_size=20, samples=200):
    import joblib
    import numpy as np
    import pandas as pd
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from load_data import load_dataset_as_sklearn_format
    started = time.perf_counter()
    np.random.seed(42)
    data_path = ROOT / 'DATA' / ('Iris.csv' if dataset == 'iris' else 'winequality.csv')
    data = load_dataset_as_sklearn_format(str(data_path))
    train, test = data['X_train'], data['X_test'].iloc[:limit]
    if model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'svm':
        model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    elif model_name == 'neural_network':
        model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(10,5), max_iter=1000, random_state=42))
    else:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    model.fit(train, data['y_train'])
    names = list(train.columns)
    targets = list(data['target_names'])
    def predict(values):
        return model.predict_proba(pd.DataFrame(values, columns=names))
    predicted = model.predict(test).astype(int)
    background = shap.sample(train, min(background_size, len(train)), random_state=42)
    explainer = shap.KernelExplainer(predict, background)
    values = explainer.shap_values(test, nsamples=samples, silent=True)
    if isinstance(values, list): values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[2] != len(targets):
        raise ValueError('Expected SHAP [instances, features, classes] for probability outputs')
    local_shap = np.array([values[i,:,c] for i,c in enumerate(predicted)])
    lime = LimeTabularExplainer(train.to_numpy(), feature_names=names, class_names=targets,
                               mode='classification', discretize_continuous=False, random_state=42)
    local_lime = []
    for i, row in enumerate(test.to_numpy()):
        target = int(predicted[i])
        exp = lime.explain_instance(row, predict, labels=(target,), num_features=len(names), num_samples=max(500, samples))
        weights = dict(exp.local_exp[target])
        local_lime.append([weights.get(j, 0.0) for j in range(len(names))])
    destination = ROOT / 'explanations' / dataset / model_name
    destination.mkdir(parents=True, exist_ok=True)
    for name, scores in [('shap', local_shap), ('lime', np.asarray(local_lime))]:
        frame = pd.DataFrame(scores, columns=names)
        frame['target_class'] = [str(targets[c]) for c in predicted]
        frame['output_scale'] = 'probability attribution' if name == 'shap' else 'local surrogate coefficient (standardised feature)'
        frame['Istanza'] = [f'Instance_{i}' for i in range(len(test))]
        frame.to_csv(destination / f'{name}_local_exp.csv', index=False)
        glob = pd.DataFrame([np.abs(scores).mean(axis=0)], columns=names)
        glob['output_scale'] = f'mean absolute {name} scores over {len(test)} test instances and their predicted classes'
        glob.to_csv(destination / f'{name}_global_exp.csv', index=False)
    model_dir = ROOT / 'models' / dataset
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{MODEL_FILES[model_name]}_model_{dataset}.joblib'
    joblib.dump({'model': model, 'target_names': targets, 'feature_names': names}, model_path)
    manifest_path = destination.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    import importlib.metadata
    manifest[model_name] = {'model_hash': file_hash(model_path), 'data_hash': file_hash(data_path),
        'explanations': {f.name: file_hash(f) for f in sorted(destination.glob('*.csv'))},
        'seed': 42, 'test_size': .25, 'test_instances': len(test), 'background_size': len(background),
        'shap_samples': samples, 'lime_samples': max(500,samples),
        'versions': {p: importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','shap','lime','joblib')}}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    result = {'dataset':dataset, 'model':model_name, 'instances':len(test), 'elapsed_seconds':round(time.perf_counter()-started,3)}
    print(json.dumps(result))
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=('iris','winequality'),default='iris')
    parser.add_argument('--model',choices=tuple(MODEL_FILES),default='random_forest')
    parser.add_argument('--limit',type=int,default=12)
    args=parser.parse_args()
    if args.limit < 1: parser.error('--limit must be positive')
    prepare(args.dataset,args.model,args.limit)
