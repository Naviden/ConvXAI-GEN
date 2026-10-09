"""UI-independent conversation manager: classify -> retrieve -> check -> render."""
from collections import deque
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import re
import time
from .adapters import CSVAdapter, MissingEvidence
from .cache import ByteLRU, digest
from .constraints import PROMPT_VERSION, checked_generate
from .intent import IntentRecognizer, validate_intent
from .locales import text, language
from .moderation import appropriate
from .runtime import Audit, Gateway, Settings, Quota, ServiceUnavailable, RateLimited

# Process-wide budgets and source cache; separate session context per Service.
_SHARED_CACHE = ByteLRU()
_SHARED_VISUALS = ByteLRU(max_bytes=16 * 1024 * 1024, max_entries=256)
_GATEWAY = Gateway()
_TURN_QUOTA = Quota(rpm=300, tpm=1000000)


def blocked_input(question):
    # A heuristic, not a security proof. The generator never receives user text.
    return bool(re.search(r'\b(ignore (?:all |previous |system )*instructions|system prompt|ignora .*istruzioni|ignorez .*instructions|ignoriere .*anweisungen|ignora .*instrucciones)\b', question, re.I))


class Service:
    def __init__(self, root, gateway=None, use_llm=False, settings=None, cache=None, audit=None):
        self.settings = settings or Settings()
        self.gateway = (gateway or _GATEWAY) if use_llm else None
        self.adapter = CSVAdapter(root, cache or _SHARED_CACHE)
        self.intent_recognizer = IntentRecognizer(self.gateway, self.settings.history_turns)
        self.audit = audit or Audit(os.getenv("CONVXAI_AUDIT_PATH"))
        self.quota = Quota(rpm=30, tpm=100000)
        self.context = None
        self.last_evidence = None
        self.pending = None

    def _result(self, response, analysis=None, **extra):
        return {"response": response, "analysis": analysis or {}, "data_loaded": False,
                "plot_displayed": False, "chart_data": None, **extra}

    def process(self, question, dataset="iris", lang="en", explicit=None):
        started = time.perf_counter()
        language(lang)
        if not isinstance(question, str) or len(question) > 8000:
            return self._result(text(lang, "blocked"))
        if self.context != dataset:
            self.intent_recognizer.conversation_history.clear()
            self.last_evidence = self.pending = None
            self.context = dataset
        try:
            self.quota.reserve(len(question.encode()))
            _TURN_QUOTA.reserve(len(question.encode()))
        except RateLimited:
            return self._result(text(lang, "limit"))
        if blocked_input(question):
            return self._result(text(lang, "blocked"), language_warning=True)
        try:
            if not appropriate(question, lang, self.gateway):
                return self._result(text(lang, "blocked"), language_warning=True)
        except Exception:
            return self._result(text(lang, "unavailable"))
        # Classifier inputs stay separate from system instructions. Generation sees evidence only.
        try:
            analysis = validate_intent(explicit) if explicit is not None else self.intent_recognizer.analyze_question(question, lang)
        except Exception:
            return self._result(text(lang, "unavailable"))
        p = analysis["parametri"]
        if self.pending and p.get("modello"):
            saved = self.pending
            saved["parametri"]["modello"] = p["modello"]
            analysis, p = saved, saved["parametri"]
            self.pending = None
        intent = analysis["intento"]
        if intent == "INFORMATIVA": return self._result(text(lang, "help"), analysis)
        if intent == "CHIARIMENTO":
            if self.last_evidence:
                return self._evidence_response(self.last_evidence, lang, analysis, started, False, clarification=True)
            return self._result(text(lang, "clarify"), analysis)
        if not p.get("modello"):
            self.pending = analysis
            return self._result(text(lang, "choose"), analysis, waiting_for_model=True)
        if intent in ("LOCALE", "WHAT_IF") and p.get("istanza") is None:
            return self._result(text(lang, "instance"), analysis)
        if p.get("explainer") == "anchor": return self._result(text(lang, "unsupported"), analysis)
        try:
            if intent == "WHAT_IF":
                result = self.adapter.what_if(dataset, p["modello"], p["istanza"], p.get("changes", {}),
                                             trust_models=os.getenv("CONVXAI_TRUST_MODELS") == "1")
                response = f'{text(lang, "whatif")}: {result["before"]} / {result["after"]}.\n{text(lang, "hypothetical")}\n{json.dumps(result["changes"], ensure_ascii=False)}'
                self._remember(analysis)
                return self._result(response, analysis, data_loaded=True, what_if=result)
            evidence, key, hit = self.adapter.load(dataset, p["modello"], p.get("explainer") or "shap",
                                                   "global" if intent == "GLOBALE" else "local", p.get("istanza"))
            result = self._evidence_response(evidence, lang, analysis, started, hit)
            if p.get("confronto_con"):
                other, _, other_hit = self.adapter.load(dataset, p["confronto_con"], p.get("explainer") or "shap", evidence.scope, evidence.instance)
                comparison = self._evidence_response(other, lang, analysis, started, other_hit)
                # Separate labelled panels; no unverified numeric deltas across different targets/scales.
                result["response"] += "\n\n" + comparison["response"]
                result["comparison_validation"] = comparison["validation"]
                result["comparison_evidence"] = other
            self.last_evidence = evidence
            self._remember(analysis)
            return result
        except (ValueError, OSError, ImportError, IndexError, KeyError):
            return self._result(text(lang, "missing"), analysis)

    def _remember(self, analysis):
        self.intent_recognizer.conversation_history.append({"intent": analysis["intento"],
            "parameters": analysis["parametri"], "followup_type": analysis.get("tipo_followup")})

    def _evidence_response(self, evidence, lang, analysis, started, hit, clarification=False):
        body, check, failures, fallback = checked_generate(evidence, lang, self.settings.top_k,
            generate=self.gateway.statements if self.gateway else None)
        heading = f'{text(lang, "heading")} — {evidence.model} / {evidence.explainer} / {evidence.scope}'
        context = json.dumps({"dataset": evidence.dataset, "instance": evidence.instance, "target": evidence.target, "scale": evidence.scale}, ensure_ascii=False)
        suffix = text(lang, evidence.scope)
        if not any(f.score for f in evidence.features): suffix = text(lang, "zero") + " " + suffix
        if clarification: suffix += "\n" + text(lang, "clarify")
        row = self.audit.record(intent=analysis["intento"], language=lang,
            latency_ms=round((time.perf_counter() - started) * 1000, 3), accepted=check.accepted,
            fallback=fallback, failures=failures, cache_hit=hit, schema="2", prompt_hash=digest(PROMPT_VERSION),
            model=os.getenv("CONVXAI_MODEL", "offline-template"), **(self.gateway.last_usage if self.gateway else {}))
        return self._result(heading + "\n" + context + "\n\n" + body + "\n\n" + suffix, analysis,
            data_loaded=True, validation=asdict(check), failures=failures, fallback=fallback,
            evidence=evidence, checked_body=body, audit=row)


def chart_png(evidence, lang="en", k=10):
    """Generate visuals from the same evidence as the checked text, on demand."""
    key = digest({"evidence": asdict(evidence), "language": lang, "k": k, "schema": "2"})
    def render():
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        selected = evidence.selected(k)
        scores = evidence.normalised()
        fig = Figure(figsize=(7, max(2.5, len(selected) * .35)))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        values = [scores[f.id] for f in selected]
        ax.barh([f.label for f in selected], values, color=["#226EAA" if x >= 0 else "#C6563D" for x in values])
        ax.invert_yaxis()
        ax.set_xlabel(text(lang, "score"))
        ax.set_title(evidence.model + " / " + evidence.scope)
        fig.tight_layout()
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=120)
        return stream.getvalue()
    return _SHARED_VISUALS.get_or_load(key, render)[0]
