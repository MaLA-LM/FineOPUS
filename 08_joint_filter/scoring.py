"""Deterministic joint score shared verbatim by statistics and filtering.

Scores are ranking signals, not calibrated probabilities. Raw input columns
are preserved by the filter; corrected features are computed transiently.
"""
from __future__ import annotations

from functools import lru_cache
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import regex

REASONS = {1: "empty_text", 2: "copy_wrong_script", 4: "lid_wrong_script", 8: "missing_semantics"}
NO_SPACE = {"Hans", "Hant", "Hani", "Jpan", "Kore", "Hang", "Thai", "Laoo", "Khmr", "Mymr"}
LETTERS = regex.compile(r"\p{L}")
WHITESPACE = regex.compile(r"\s+")
FEATURE_COLUMNS = [
    "source_text", "target_text", "similarity_score", "qe_score", "qe_model", "llm_judge_score",
    "qe_src_seen", "qe_tgt_seen", "score_lcs_ratio", "score_levenshtein", "score_term_punct",
    "score_numerals", "score_repeat_src", "score_repeat_trg", "src_max_word_len", "trg_max_word_len",
    "filter_html_src", "filter_html_trg", "filter_regex_src",
] + [f"{side}_predlang_{field}_{model}" for side in ("src", "tgt")
     for field in ("id", "conf") for model in ("glotlid", "conlid")]


def validate_config(c):
    if set(c["weights"]) != {"P", "Q", "H", "L"} or any(v <= 0 for v in c["weights"].values()):
        raise ValueError("weights must contain positive P, Q, H, L values")
    if not 0 < c["score_floor"] < 1 or not 0 <= c["tail_start"] < 1:
        raise ValueError("Invalid score_floor/tail_start")
    for key in ("llm_weight", "weak_threshold", "weak_penalty", "copy_penalty",
                "copy_risk_threshold", "copy_language_threshold", "lid_hard_confidence", "script_min_fraction"):
        if not 0 <= c[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if c["llm_weight"] == 1:
        raise ValueError("llm_weight must be < 1")
    if len(c["volume_keep_rates"]) != len(c["volume_edges"]) + 1:
        raise ValueError("volume rates/edges do not match")
    if sorted(set(c["volume_edges"])) != c["volume_edges"] or any(x <= 0 for x in c["volume_edges"]):
        raise ValueError("volume_edges must be positive, increasing")
    if any(not 0 < r <= 1 for r in c["volume_keep_rates"]):
        raise ValueError("volume_keep_rates must be in (0, 1]")
    if c["lid_hard_min_letters"] < 1 or c["copy_hard_min_letters"] < 1:
        raise ValueError("letter limits must be positive")


def numeric(table, name, default=np.nan):
    if name not in table.column_names:
        return np.full(table.num_rows, default, dtype=float)
    return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=float)


def strings(table, name, default=""):
    if name not in table.column_names:
        return np.full(table.num_rows, default, dtype=object)
    return np.asarray([x or default for x in table[name].to_pylist()], dtype=object)


def script_name(code):
    script = code.rsplit("_", 1)[-1]
    return "Han" if script in {"Hans", "Hant", "Hani"} else script


def compatible(expected, predicted, config):
    if not predicted or "_" not in predicted:
        return None
    aliases = config["language_root_aliases"]
    e, p = expected.split("_", 1)[0], predicted.split("_", 1)[0]
    return aliases.get(e, e) == aliases.get(p, p) and script_name(expected) == script_name(predicted)


@lru_cache(maxsize=128)
def script_pattern(code):
    script = script_name(code)
    names = {"Jpan": ["Han", "Hiragana", "Katakana"], "Kore": ["Hangul", "Han"],
             "Hang": ["Hangul", "Han"]}.get(script, [script])
    try:
        return regex.compile("(?:" + "|".join(r"\p{Script=" + n + "}" for n in names) + ")")
    except regex.error:
        return None  # Unknown script is not evidence of a wrong language.


def wrong_script(text, lang, min_letters, config):
    pattern = script_pattern(lang)
    if pattern is None:
        return False
    letters = LETTERS.findall(text)
    return len(letters) >= min_letters and sum(pattern.fullmatch(c) is not None for c in letters) / len(letters) < config["script_min_fraction"]


def word_features(texts):
    arr = pa.array(texts.tolist(), type=pa.string())
    counts = np.asarray(pc.count_substring_regex(arr, pattern=r"\S+"), dtype=float)
    nonspace = np.asarray(pc.utf8_length(pc.replace_substring_regex(arr, pattern=r"\s+", replacement="")), dtype=float)
    return counts, nonspace / np.maximum(counts, 1)


def extract(table, pair, config):
    src, tgt = pair.split("-")
    texts = [strings(table, "source_text"), strings(table, "target_text")]
    n = table.num_rows
    chars = [np.asarray(pc.utf8_length(pa.array(t.tolist(), type=pa.string())), dtype=float) for t in texts]
    empty = np.asarray([not s.strip() or not t.strip() for s, t in zip(*texts)])
    reasons = empty.astype(np.uint16)
    side_l, side_wrong = [], []
    for side, expected, text in zip(("src", "tgt"), (src, tgt), texts):
        evidence, predictions, confidences = [], [], []
        for model in ("glotlid", "conlid"):
            pred = strings(table, f"{side}_predlang_id_{model}")
            raw_conf = numeric(table, f"{side}_predlang_conf_{model}")
            conf = np.where(np.isfinite(raw_conf), np.clip(raw_conf, 0, 1), 0)
            mapping = {p: compatible(expected, p, config) for p in set(pred)}
            sign = np.asarray([0 if mapping[p] is None else (1 if mapping[p] else -1) for p in pred])
            evidence.append(sign * conf)
            predictions.append(pred)
            confidences.append(conf)
        side_l.append((1 + (evidence[0] + evidence[1]) / 2) / 2)
        candidates = ((predictions[0] == predictions[1]) & (evidence[0] < 0)
                      & (confidences[0] >= config["lid_hard_confidence"])
                      & (confidences[1] >= config["lid_hard_confidence"]))
        bad = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(candidates & ~empty):
            bad[i] = wrong_script(text[i], expected, config["lid_hard_min_letters"], config)
        side_wrong.append(bad)
    reasons[side_wrong[0] | side_wrong[1]] |= 4
    normalized_equal = np.zeros(n, dtype=bool)
    for i, (s, t) in enumerate(zip(*texts)):
        normalized_equal[i] = WHITESPACE.sub(" ", unicodedata.normalize("NFC", s)).strip() == WHITESPACE.sub(" ", unicodedata.normalize("NFC", t)).strip()
    if script_name(src) != script_name(tgt):
        for i in np.flatnonzero(normalized_equal & ~empty):
            if (wrong_script(texts[0][i], src, config["copy_hard_min_letters"], config)
                    or wrong_script(texts[1][i], tgt, config["copy_hard_min_letters"], config)):
                reasons[i] |= 2
    sim, qe = numeric(table, "similarity_score"), numeric(table, "qe_score")
    reasons[~np.isfinite(sim) & ~np.isfinite(qe)] |= 8
    words, averages = zip(*(word_features(t) for t in texts))
    logratio = np.log(np.maximum(chars[0], 1) / np.maximum(chars[1], 1))
    raw = {"logratio": logratio}
    if src.rsplit("_", 1)[-1] not in NO_SPACE and tgt.rsplit("_", 1)[-1] not in NO_SPACE:
        raw["word_logratio"] = np.log(np.maximum(words[0], 1) / np.maximum(words[1], 1))
    for side, lang, avg in zip(("src", "trg"), (src, tgt), averages):
        if lang.rsplit("_", 1)[-1] not in NO_SPACE:
            raw[f"{side}_avg_word"] = avg
            raw[f"{side}_max_word"] = numeric(table, f"{side}_max_word_len")
    raw["repeat"] = np.fmax(numeric(table, "score_repeat_src"), numeric(table, "score_repeat_trg"))
    raw["punct"] = -numeric(table, "score_term_punct") / np.log1p(np.maximum(np.minimum(*chars), 1))
    overlap = np.fmax(numeric(table, "score_lcs_ratio"), numeric(table, "score_levenshtein"))
    overlap[normalized_equal] = 1
    copy = np.clip((overlap - 0.8) / 0.2, 0, 1)
    # Short entities and numeral-only strings carry no copy penalty.
    for i in np.flatnonzero(np.nan_to_num(copy) > 0):
        if min(len(LETTERS.findall(texts[0][i])), len(LETTERS.findall(texts[1][i]))) < 12:
            copy[i] = 0
    digit_present = np.asarray(pc.match_substring_regex(pa.array(texts[0].tolist()), pattern=r"\p{Nd}")) | np.asarray(pc.match_substring_regex(pa.array(texts[1].tolist()), pattern=r"\p{Nd}"))
    numbers = np.where(digit_present, 1 - np.clip(numeric(table, "score_numerals"), 0, 1), 0)
    # The arbitrary user regex has unknown semantics: audit it, never treat True
    # as a defect by itself. HTML flags have known semantics and are included.
    html = np.fmax(numeric(table, "filter_html_src", 0), numeric(table, "filter_html_trg", 0))
    seen = []
    for side in ("src", "tgt"):
        values = table[f"qe_{side}_seen"].to_pylist() if f"qe_{side}_seen" in table.column_names else [None] * n
        seen.append(np.asarray([1 if v is True else (0.5 if v is False else 0.75) for v in values]))
    llm = numeric(table, "llm_judge_score")
    return {"raw": raw, "P": sim, "Q": qe, "J": llm, "L": np.minimum(*side_l),
            "qe_reliability": np.minimum(*seen), "qe_model": strings(table, "qe_model", "unknown"),
            "copy": copy, "numbers": numbers, "html": html, "reasons": reasons,
            "chars_src": chars[0], "chars_tgt": chars[1], "texts": texts,
            "cohort": np.isfinite(llm).astype(np.int8)}


def fit_cdf(values):
    values = np.sort(values[np.isfinite(values)])
    if not len(values):
        return None
    knots = np.unique(np.quantile(values, np.linspace(0, 1, 257)))
    ranks = (np.searchsorted(values, knots, "left") + np.searchsorted(values, knots, "right")) / (2 * len(values))
    return {"x": knots.tolist(), "p": ranks.tolist(), "n": len(values)}


def cdf(values, spec):
    if spec is None:
        return np.full(len(values), 0.5)
    return np.interp(values, spec["x"], spec["p"], left=0, right=1)


def fit_calibration(features, config, mrr):
    ok = features["reasons"] == 0
    # No survivor: no quality baselines are invented from rejected rows.
    raw, centers = {}, {}
    for key, values in features["raw"].items():
        finite = ok & np.isfinite(values)
        if key.endswith("logratio"):
            centers[key] = float(np.median(values[finite])) if finite.any() else 0.0
            values = np.abs(values - centers[key])
        raw[key] = fit_cdf(values[ok])
    models = {}
    for model in sorted(set(features["qe_model"][ok])):
        models[model] = fit_cdf(features["Q"][ok & (features["qe_model"] == model)])
    return {"config": config, "mrr": mrr, "centers": centers, "raw": raw,
            "P": fit_cdf(features["P"][ok]), "Q": fit_cdf(features["Q"][ok]),
            "Q_models": models, "J": fit_cdf(features["J"][ok])}


def score(features, calibration):
    c = calibration["config"]
    n = len(features["P"])
    floor = c["score_floor"]
    p = floor + (1 - floor) * cdf(features["P"], calibration["P"])
    q = np.zeros(n)
    for model in set(features["qe_model"]):
        mask = features["qe_model"] == model
        spec = calibration["Q_models"].get(model) or calibration["Q"]
        q[mask] = floor + (1 - floor) * cdf(features["Q"][mask], spec)
    risks = {}
    for key, values in features["raw"].items():
        spec = calibration["raw"].get(key)
        if spec is None or len(spec["x"]) < 2:
            risks[key] = np.zeros(n)  # Disable unavailable/constant calibration features.
            continue
        if key in calibration["centers"]:
            values = np.abs(values - calibration["centers"][key])
        risks[key] = np.clip((cdf(values, spec) - c["tail_start"]) / (1 - c["tail_start"]), 0, 1)
    def maximum(keys):
        return np.maximum.reduce([np.nan_to_num(risks[k], nan=0) for k in keys]) if keys else np.zeros(n)
    length = maximum([k for k in risks if k.endswith("logratio")])
    structure = np.maximum(maximum([k for k in risks if k.endswith("word") or k == "punct"]),
                           np.nan_to_num(features["html"], nan=0))
    groups = np.stack([length, np.nan_to_num(features["copy"], nan=0),
                       maximum(["repeat"] if "repeat" in risks else []),
                       np.nan_to_num(features["numbers"], nan=0), structure])
    h = np.clip(1 - 0.7 * groups.max(axis=0) - 0.3 * groups.mean(axis=0), floor, 1)
    scores = {"P": p, "Q": q, "H": h, "L": features["L"]}
    mrr = calibration["mrr"].get("MRR")
    pr = 0.75 if mrr is None else (0.5 if mrr < 0.8 else 1.0)
    weights = {k: np.full(n, float(v)) for k, v in c["weights"].items()}
    weights["P"] *= pr * np.isfinite(features["P"])
    weights["Q"] *= features["qe_reliability"] * np.isfinite(features["Q"])
    # Normalize base weights so LLM contributes exactly llm_weight when present.
    denominator = sum(weights.values())
    logscore = sum(weights[k] * np.log(np.clip(np.nan_to_num(v, nan=floor), floor, 1)) for k, v in scores.items()) / denominator
    present = np.isfinite(features["J"])
    j = floor + (1 - floor) * cdf(features["J"], calibration["J"])
    logscore[present] = (1 - c["llm_weight"]) * logscore[present] + c["llm_weight"] * np.log(np.clip(j[present], floor, 1))
    result = np.exp(logscore)
    weak = sum(((v < c["weak_threshold"]) & (weights[k] > 0)).astype(int) for k, v in scores.items()) >= 2
    result[weak] *= c["weak_penalty"]
    conflict = (features["copy"] >= c["copy_risk_threshold"]) & (features["L"] < c["copy_language_threshold"])
    result[conflict] *= c["copy_penalty"]
    return result
