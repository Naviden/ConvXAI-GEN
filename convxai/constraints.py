"""Validate the actual displayed statements, never hidden LLM metadata."""
from dataclasses import dataclass
import json
import math
import re
from .locales import text

PROMPT_VERSION = "checked-statements-v2"

@dataclass(frozen=True)
class Check:
    accepted: bool
    reason: str
    tau: float | None = None
    error: float | None = None


def label_for(feature):
    # JSON escaping gives an unambiguous, single-line label grammar.
    return json.dumps(feature.label, ensure_ascii=False)


def statement(position, feature, value, lang):
    return f'{position}. {label_for(feature)} [{feature.id}]: {value:+.10f} {text(lang, "score")}'


def template(evidence, lang="en", k=10):
    scores = evidence.normalised()
    return "\n".join(statement(i, f, scores[f.id], lang) for i, f in enumerate(evidence.selected(k), 1))


def validate(body, evidence, lang="en", k=10, tau_min=1.0, epsilon=1e-8):
    if not -1 <= tau_min <= 1 or epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("Invalid acceptance thresholds")
    chosen = evidence.selected(k)
    expected = {f.id: f for f in chosen}
    scores = evidence.normalised()
    lines = body.splitlines()
    if len(lines) != len(chosen):
        return Check(False, "statement_count")
    pattern = re.compile(r'(\d+)\. ("(?:[^"\\]|\\.)*") \[(f[0-9a-f]{16})\]: ([+-]\d+\.\d{10}) ' + re.escape(text(lang, "score")))
    ids, values = [], {}
    for i, line in enumerate(lines, 1):
        match = pattern.fullmatch(line)
        if not match or int(match[1]) != i:
            return Check(False, "unparseable_statement")
        _, label, fid, number = match.groups()
        if fid not in expected or fid in values:
            return Check(False, "feature_support")
        if label != label_for(expected[fid]):
            return Check(False, "feature_label")
        v = float(number)
        # Values that round to zero are allowed only within declared precision.
        ref = scores[fid]
        if (ref > 0 and v < 0) or (ref < 0 and v > 0) or (ref == 0 and v != 0):
            return Check(False, "sign")
        if ref != 0 and v == 0 and abs(ref) >= 0.5e-10:
            return Check(False, "sign")
        ids.append(fid)
        values[fid] = v
    error = math.sqrt(math.fsum((scores[fid] - values[fid]) ** 2 for fid in ids))
    if error > epsilon:
        return Check(False, "magnitude", error=error)
    # Ignore pairs tied in SOURCE magnitudes (tie-equivalent rank order).
    # Each non-tied pair is concordant or discordant in the displayed ordering.
    concordant = discordant = 0
    for i, fid in enumerate(ids):
        for other in ids[i + 1:]:
            left, right = abs(expected[fid].score), abs(expected[other].score)
            if left > right: concordant += 1
            elif left < right: discordant += 1
    pairs = concordant + discordant
    tau = (concordant - discordant) / pairs if pairs else 1.0
    if tau < tau_min:
        return Check(False, "rank", tau, error)
    return Check(True, "accepted", tau, error)


def checked_generate(evidence, lang="en", k=10, generate=None, retries=1):
    """Bounded generation; rejected free prose is never shown to the user."""
    canonical = template(evidence, lang, k)
    failures = []
    if generate:
        for attempt in range(retries + 1):
            try:
                candidate = generate(canonical, lang, attempt)
                result = validate(candidate, evidence, lang, k)
                if result.accepted:
                    return candidate, result, failures, False
                failures.append(result.reason)
            except Exception:
                failures.append("provider_failure")
    check = validate(canonical, evidence, lang, k)
    if not check.accepted:
        raise ValueError("Template failed its own validation: " + check.reason)
    return canonical, check, failures, True
