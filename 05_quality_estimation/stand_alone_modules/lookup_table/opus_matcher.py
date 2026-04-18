"""Matching logic: convert direction keys and join OPUS dirs with FLORES lookup."""

# FLORES-200 codes use underscores (e.g. ace_Arab), never hyphens, so the
# single hyphen in the directory name always separates source from target.
DEFAULT_MODEL = "qwen3-4b-instruct-2507"
METRICX_MODEL = "metricx24"
QWEN_PREFIX = "qwen"
HIGH_RESOURCE_SENTENCE_THRESHOLD = 2_000_000
METRICX_QWEN_GAP_THRESHOLD = 0.1


def path_key_to_flores_key(path_key):
    """Convert path-style key to FLORES-style key.

    'abk_Cyrl-abk_Cyrl' -> 'abk_Cyrl->abk_Cyrl'
    """
    src, tgt = path_key.split("-", 1)
    return f"{src}->{tgt}"


def _extract_codes(dir_name):
    """Extract the source and target language codes from a directory name."""
    return tuple(dir_name.split("-", 1))


def _pick_default_model(dir_name, default_strategy, metricx_supported_codes):
    """Choose the assigned model for an unmatched direction."""
    if default_strategy == "qwen3":
        return DEFAULT_MODEL
    if default_strategy == "metricx-24":
        return METRICX_MODEL

    src_code, tgt_code = _extract_codes(dir_name)
    if src_code in metricx_supported_codes or tgt_code in metricx_supported_codes:
        return METRICX_MODEL
    return DEFAULT_MODEL


def _parse_required_float(value, field_name, direction_key):
    """Convert a required numeric value to float or raise."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid float for '{field_name}' in {direction_key}: {value!r}"
        ) from None


def _score_for_assigned_model(entry, model):
    """Return the score for the model ultimately assigned to the OPUS pair."""
    if model == METRICX_MODEL:
        return entry["metricx24_mean"]
    if model == DEFAULT_MODEL:
        return entry["qwen3_4b_instruct_2507_mean"]
    return entry["winner_avg_score"]


def _metricx_supports_direction(dir_name, metricx_supported_codes):
    """Return True when MetricX-24 supports either language in the direction."""
    src_code, tgt_code = _extract_codes(dir_name)
    return src_code in metricx_supported_codes or tgt_code in metricx_supported_codes


def _assign_qwen_pair(
    entry,
    num_sentences,
    direction_key,
    dir_name,
    metricx_supported_codes,
    high_resource_threshold,
):
    """Split high-resource Qwen-winning directions between qwen3-4b and MetricX-24."""
    assigned_model = DEFAULT_MODEL
    metricx_mean = _parse_required_float(
        entry["metricx24_mean"],
        "metricx24_mean",
        direction_key,
    )
    qwen4b_mean = _parse_required_float(
        entry["qwen3_4b_instruct_2507_mean"],
        "qwen3_4b_instruct_2507_mean",
        direction_key,
    )
    support_condition = _metricx_supports_direction(dir_name, metricx_supported_codes)
    gap_condition = abs(metricx_mean - qwen4b_mean) < METRICX_QWEN_GAP_THRESHOLD
    is_high_resource = num_sentences > high_resource_threshold

    if is_high_resource and (support_condition or gap_condition):
        assigned_model = METRICX_MODEL

    return {
        "assigned_model": assigned_model,
        "assigned_score": _score_for_assigned_model(entry, assigned_model),
        "is_high_resource": is_high_resource,
        "support_condition": support_condition,
        "gap_condition": gap_condition,
    }


def _require_runtime(runtime_lookup, model, direction_key):
    """Fetch a required runtime entry or raise."""
    if model not in runtime_lookup:
        raise KeyError(f"Missing runtime for model '{model}' in {direction_key}")
    rate = runtime_lookup[model]
    if rate <= 0:
        raise ValueError(
            f"Runtime for model '{model}' must be > 0 in {direction_key}, got {rate}"
        )
    return rate


def build_opus_rows(
    opus_dirs,
    sentence_counts,
    flores_lookup,
    runtime_lookup,
    default_strategy="both",
    metricx_supported_codes=frozenset(),
    high_resource_threshold=HIGH_RESOURCE_SENTENCE_THRESHOLD,
):
    """Match each OPUS direction against the FLORES lookup.

    Args:
        opus_dirs: sorted list of direction directory names
        sentence_counts: dict {dir_name: num_sentences}
        flores_lookup: dict from load_flores_lookup()
        runtime_lookup: dict from load_model_runtime()
        default_strategy: one of "qwen3", "metricx-24", or "both"
        metricx_supported_codes: set of FLORES language codes supported by MetricX-24
        high_resource_threshold: sentence-count cutoff for qwen->metricx reassignment

    Returns:
        rows: list[dict] ready for CSV output
        matched: list[str] direction keys that matched
        unmatched: list[str] direction keys with no FLORES entry
        stats: dict with report counters for assignment behavior
    """
    rows = []
    matched = []
    unmatched = []
    stats = {
        "matched_qwen_winners": 0,
        "matched_qwen_high_resource": 0,
        "matched_qwen_reassigned_to_metricx": 0,
        "matched_qwen_reassigned_support_only": 0,
        "matched_qwen_reassigned_gap_only": 0,
        "matched_qwen_reassigned_both": 0,
    }

    for dir_name in opus_dirs:
        flores_key = path_key_to_flores_key(dir_name)
        if dir_name not in sentence_counts:
            raise KeyError(f"Missing sentence count for direction '{dir_name}'")
        num_sentences = sentence_counts[dir_name]

        if flores_key in flores_lookup:
            entry = flores_lookup[flores_key]
            winner = entry["winner"]
            score = entry["winner_avg_score"]

            if winner.startswith(QWEN_PREFIX):
                stats["matched_qwen_winners"] += 1
                qwen_assignment = _assign_qwen_pair(
                    entry,
                    num_sentences,
                    flores_key,
                    dir_name,
                    metricx_supported_codes,
                    high_resource_threshold,
                )
                winner = qwen_assignment["assigned_model"]
                score = qwen_assignment["assigned_score"]

                if qwen_assignment["is_high_resource"]:
                    stats["matched_qwen_high_resource"] += 1
                if winner == METRICX_MODEL:
                    stats["matched_qwen_reassigned_to_metricx"] += 1
                    if (
                        qwen_assignment["support_condition"]
                        and qwen_assignment["gap_condition"]
                    ):
                        stats["matched_qwen_reassigned_both"] += 1
                    elif qwen_assignment["support_condition"]:
                        stats["matched_qwen_reassigned_support_only"] += 1
                    elif qwen_assignment["gap_condition"]:
                        stats["matched_qwen_reassigned_gap_only"] += 1

            rate = _require_runtime(runtime_lookup, winner, flores_key)
            est_hours = _est_hours(num_sentences, rate)

            matched.append(dir_name)
            rows.append(
                {
                    "direction_key": dir_name,
                    "winner_model": winner,
                    "winner_avg_score": score,
                    "num_sentences": num_sentences,
                    "rate_per_hour": rate,
                    "est_hours": est_hours,
                }
            )
        else:
            model = _pick_default_model(
                dir_name,
                default_strategy,
                metricx_supported_codes,
            )
            default_rate = _require_runtime(runtime_lookup, model, flores_key)
            est_hours = _est_hours(num_sentences, default_rate)

            unmatched.append(dir_name)
            rows.append(
                {
                    "direction_key": dir_name,
                    "winner_model": model,
                    "winner_avg_score": "unknown",
                    "num_sentences": num_sentences,
                    "rate_per_hour": default_rate,
                    "est_hours": est_hours,
                }
            )

    return rows, matched, unmatched, stats


def _est_hours(num_sentences, rate):
    """Estimated wall-clock hours for scoring the full OPUS direction."""
    return f"{num_sentences / rate:.1f}"
