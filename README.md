# ConvXAI-GEN

Companion implementation for **“ConvXAI-GEN: A Modular LLM-Mediated XAI System with Faithfulness Constraints”**, by Chiara Mezzanzanica and Navid Nobani.

The application retrieves SHAP/LIME evidence, resolves conversational requests, and checks the **actual displayed attribution statements** for feature support, ordering, signed magnitude, and label consistency. Failed LLM outputs are discarded and replaced with a checked template. Numeric guarantees apply to this controlled statement format, not arbitrary natural-language explanations, causal conclusions, or the correctness of the explainer itself.

## Run

Use Python 3.12 for the tested app environment. The dependency-free core also supports Python 3.11+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
streamlit run interface.py
```

No API key is required in the default template mode. The repository includes newly generated Iris evidence for **Random Forest and SVM, instances 0–11**, plus global summaries over those 12 test instances. It includes no pre-trained model binaries. The app generates plots directly from the checked evidence.

Example requests:

- `global SHAP Random Forest`
- `local SHAP SVM instance 0`
- `compare with Random Forest`
- `simpler`

Select English, Italian, Spanish, French, or German in the sidebar. Labels for dataset features stay canonical; surrounding templates and the interface are translated. Offline intent parsing recognises a small documented vocabulary; arbitrary multilingual phrasing requires the optional LLM route. Five-language template coverage is not a cross-language performance evaluation.

## Optional LLM mode

Export environment variables before launching Streamlit; `.env.example` lists them. No credentials are stored in the code and no `.env` file is loaded automatically.

```bash
export CONVXAI_USE_LLM=1
export AZURE_OPENAI_ENDPOINT='https://YOUR-RESOURCE.openai.azure.com/'
export AZURE_OPENAI_API_KEY='YOUR-KEY'
export CONVXAI_MODEL='YOUR-DEPLOYMENT'
```

Alternatively, export `OPENAI_API_KEY` and set `CONVXAI_MODEL` to an available model name, without the Azure variables. The LLM handles moderation and intent parsing and can produce the controlled evidence statements. The statement generator receives the evidence, **not the user's prompt or dialogue history**. Additional prose, missing IDs, changed labels, wrong numbers/signs, and inconsistent ordering fail validation. The default generation ceiling is 300 tokens; truncated answers fall back to the template. Intent parsing uses 500 tokens and moderation 100. All calls use temperature 0 and top-p 1.

Online use sends the question and recent resolved conversation context to the configured provider. Clicking **Listen** sends the latest response to the external gTTS service. These features are optional. No live-provider evaluation is included.

## Reproduce evidence and what-if predictions

```bash
python scripts/prepare.py --dataset iris --model random_forest --limit 12
python scripts/prepare.py --dataset iris --model svm --limit 12
```

The same script supports `winequality`, `xgboost`, and `neural_network`. It trains from the included CSVs, uses a fixed split and seed, explains the **predicted class explicitly**, writes local/global CSVs, and records model/data/explanation hashes and dependency versions in a manifest. Global summaries are the mean absolute attributions for the sampled test instances and their predicted classes, not an estimate over the complete dataset. LIME values are local surrogate coefficients on standardised features; they are not interchangeable with SHAP probability contributions. Execution time is printed for each preparation run.

Generated models stay in the ignored `models/` directory. Enable `CONVXAI_TRUST_MODELS=1` only for models you generated/trust; joblib files are executable serialisations. Then, for example:

```text
what if Random Forest instance 0 {"PetalLengthCm": 1.0}
```

This recomputes before/after predictions. It does not optimise a counterfactual, establish causality, or recommend an intervention. Missing explanation artefacts require running the preparation script; the app does not train or launch expensive explainers on a chat cache miss.

## Check the implementation

Core tests need no third-party packages or network:

```bash
python -m unittest discover -s tests -v
```

With app dependencies installed:

```bash
python -m unittest discover -s tests_integration -v
```

Tests cover numeric/sign/rank violations, ties, zero vectors, feature omission/duplication, multilingual formats, fallback, real CSV loading, provenance invalidation, bounded memory, single-flight caching, provider failures, quotas, and the Streamlit UI. These are software verification tests, not empirical evidence of better faithfulness, usability, latency, or multilingual performance.

## Implementation boundaries

- `convxai/evidence.py` and `constraints.py`: canonical IDs, full-vector L1 normalisation, explicit subset selection, statement parsing, magnitude error, sign and support checks. Rank agreement counts concordant/discordant **non-tied source pairs**; it is not standard Kendall tau-b. Default acceptance requires agreement 1.0 and L2 error at most `1e-8`; displayed values use ten decimal places.
- `adapters.py` and `cache.py`: content-versioned CSV retrieval, provenance checks, 32 MiB serialised-payload / 512-entry LRU, one-hour TTL, single-flight loading and prefix invalidation. This byte budget excludes Python object overhead and in-flight/transient allocations. Plots use a separate 16 MiB cache. Source attributions are shared across languages; visuals are keyed by language. Generated responses are not cached.
- `intent.py`, `service.py`, `locales.py`: UI-independent orchestration, five template sets, explicit instance resolution, model comparisons, and a 20-turn bounded history. Comparisons show each model separately and do not assert cross-model magnitude equivalence.
- `runtime.py`: at most three provider attempts, exponential backoff, 30-second timeout, a circuit breaker, session and process-local quotas, and optional metadata-only audit logs. Token quotas reserve UTF-8 input bytes plus output-token allowance conservatively; this is not an exact tokenizer count. There is no distributed rate limiter, user authentication, moderation-review console, or compliance guarantee.
- `moderation.py`: a small rule filter plus optional LLM classification. Provider/moderation failures request a retry instead of silently treating the input as appropriate. Heuristic injection detection is not a complete security defence.

## Reproducibility and licensing

`requirements.txt` pins the tested direct dependencies. Artefact manifests record the main preparation-library versions. Transitive dependencies and provider behaviour may change; rebuild a virtual environment and retain its full package snapshot for a new experiment. Hosted model access and deployments are supplied by the operator.

Code is released under the [MIT license](LICENSE). Dataset attribution and transformations are documented in [DATA/README.md](DATA/README.md); the code license does not replace the dataset licenses. See [CITATION.cff](CITATION.cff) for citation metadata.
