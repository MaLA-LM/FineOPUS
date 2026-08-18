import argparse
import html
import json
import re
from pathlib import Path
from string import Template


FLUENCY_OPTIONS = [
    ("true", "Yes"),
    ("false", "No"),
]

PARALLELISM_OPTIONS = [
    ("not_parallel", "Not parallel"),
    ("partial", "Partially"),
    ("full", "Fully"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a static HTML annotation interface for sentence fluency and parallelism."
    )
    parser.add_argument(
        "--input_path",
        default="./annotation_samples",
        help="Path to an input JSONL file or a directory of input JSONL files.",
    )
    parser.add_argument(
        "--output_path",
        default="./annotation",
        help="Path to an output HTML file or a directory for generated reports.",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path} line {line_number}: {exc}") from exc
    return rows


def parse_lang_codes_from_filename(filename: str):
    match = re.match(r"^sample_([^_]+)_([^_]+)_([^_]+)_([^_]+)_(.+)\.jsonl$", filename)
    if not match:
        return None, None
    return f"{match.group(1)}_{match.group(2)}", f"{match.group(3)}_{match.group(4)}"


def report_filename_from_input(filename: str):
    if filename.endswith(".jsonl"):
        return f"{filename[:-len('.jsonl')]}.html"
    return f"{filename}.html"


STYLESHEET_NAME = "annotation-interface.css"


STYLESHEET_CONTENT = """        :root {
            --bg: #f5f1e8;
            --panel: rgba(255, 252, 246, 0.94);
            --panel-border: rgba(64, 48, 35, 0.12);
            --text: #201913;
            --muted: #69584d;
            --accent: #0c5c51;
            --shadow: 0 18px 40px rgba(54, 39, 26, 0.08);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            color: var(--text);
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at top left, rgba(12, 92, 81, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 24%),
                linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
        }

        main {
            width: min(1100px, calc(100% - 24px));
            margin: 0 auto;
            padding: 20px 0 32px;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255, 252, 246, 0.95), rgba(246, 239, 228, 0.92));
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
            margin-bottom: 16px;
        }

        .hero h1 {
            margin: 0 0 6px;
            font-family: "IBM Plex Serif", Georgia, serif;
            font-size: clamp(1.6rem, 3vw, 2.2rem);
            line-height: 1.1;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
        }

        .controls {
            background: rgba(255, 252, 246, 0.96);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            display: grid;
            gap: 8px;
            padding: 10px 12px;
            margin-bottom: 14px;
            position: sticky;
            top: 10px;
            z-index: 20;
            backdrop-filter: blur(10px);
        }

        .controls-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 8px 14px;
            align-items: start;
        }

        .control-actions {
            display: flex;
            gap: 6px;
            align-items: center;
            flex-wrap: nowrap;
        }

        .control-metrics {
            display: flex;
            gap: 6px 12px;
            align-items: center;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .control-metric {
            display: inline-flex;
            gap: 4px;
            align-items: baseline;
            white-space: nowrap;
            font-size: 0.78rem;
            color: var(--muted);
        }

        .metric-value {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text);
        }

        .control-status {
            font-size: 0.76rem;
            color: var(--muted);
        }

        .button {
            appearance: none;
            border: 1px solid transparent;
            border-radius: 999px;
            padding: 6px 10px;
            font: inherit;
            font-weight: 600;
            font-size: 0.9rem;
            white-space: nowrap;
            cursor: pointer;
            transition: transform 120ms ease, background 120ms ease;
        }

        .button:hover { transform: translateY(-1px); }

        .button-primary {
            background: var(--accent);
            color: white;
        }

        .button-secondary {
            background: rgba(255, 255, 255, 0.78);
            border-color: rgba(32, 25, 19, 0.12);
            color: var(--text);
        }

        .entries {
            display: grid;
            gap: 12px;
        }

        .entry {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 14px;
        }

        .entry[data-review-state="complete"] {
            border-color: rgba(12, 92, 81, 0.35);
        }

        .entry[data-review-state="in-progress"] {
            border-color: rgba(180, 83, 9, 0.28);
        }

        .entry-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 10px;
        }

        .entry-header h2 {
            margin: 0;
            font-size: 1.05rem;
            font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
            word-break: break-word;
        }

        .badges {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            justify-content: flex-end;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .badge-neutral { background: rgba(148, 163, 184, 0.14); color: #334155; }
        .review-pending { background: rgba(148, 163, 184, 0.16); color: #334155; }
        .review-progress { background: rgba(180, 83, 9, 0.12); color: #9a3412; }
        .review-complete { background: rgba(12, 92, 81, 0.14); color: var(--accent); }

        .entry-content {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 10px;
        }

        .panel,
        .choice-panel {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(64, 48, 35, 0.08);
            border-radius: 14px;
            padding: 10px;
        }

        .panel h3,
        .choice-panel h3 {
            margin: 0 0 6px;
            font-size: 0.8rem;
            /* text-transform: uppercase; */
            letter-spacing: 0.06em;
            color: var(--muted);
        }

        .text-block {
            margin: 0;
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 0.95rem;
        }

        .annotation-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
        }

        .choice-buttons {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .choice-btn {
            appearance: none;
            border: 2px solid rgba(32, 25, 19, 0.14);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.82);
            padding: 9px 14px;
            cursor: pointer;
            font: inherit;
            color: var(--text);
            font-weight: 600;
            min-width: 88px;
            transition: transform 100ms ease, background 100ms ease, border-color 100ms ease;
        }

        .choice-btn:hover {
            transform: translateY(-2px);
            background: rgba(12, 92, 81, 0.07);
            border-color: var(--accent);
        }

        .choice-btn.selected {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }

        @media (max-width: 900px) {
            .annotation-grid { grid-template-columns: 1fr; }
        }

        @media (max-width: 640px) {
            .entry-content { grid-template-columns: 1fr; }
            .controls-top { grid-template-columns: 1fr; }
            .control-metrics { justify-content: flex-start; }
            .entry-header { flex-direction: column; }
            .badges { justify-content: flex-start; }
        }

        body.annotation-index {
            margin: 0;
            padding: 32px 16px;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            color: #201913;
        }

        body.annotation-index main {
            width: min(860px, 100%);
            margin: 0 auto;
            background: rgba(255, 252, 246, 0.94);
            border: 1px solid rgba(64, 48, 35, 0.12);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 18px 40px rgba(54, 39, 26, 0.08);
        }

        body.annotation-index h1 {
            margin-top: 0;
            font-family: "IBM Plex Serif", Georgia, serif;
        }

        body.annotation-index ul {
            list-style: none;
            padding: 0;
            margin: 20px 0 0;
            display: grid;
            gap: 12px;
        }

        body.annotation-index li {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(64, 48, 35, 0.08);
        }

        body.annotation-index a {
            color: #0c5c51;
            text-decoration: none;
            font-weight: 600;
        }

        body.annotation-index a:hover {
            text-decoration: underline;
        }

        body.annotation-index span {
            color: #69584d;
        }
    """


def write_stylesheet(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stylesheet_file = output_dir / STYLESHEET_NAME
    stylesheet_file.write_text(STYLESHEET_CONTENT, encoding="utf-8")
    return stylesheet_file


def inject_stylesheet_link(rendered_html: str, stylesheet_href: str):
    head, remainder = rendered_html.split("<style>", 1)
    _, tail = remainder.split("</style>", 1)
    return f"{head}<link rel=\"stylesheet\" href=\"{html.escape(stylesheet_href)}\">\n{tail}"


def build_choice_buttons(field_name: str, options):
    return "".join(
        (
            f'<button class="choice-btn" data-field="{html.escape(field_name)}" '
            f'data-value="{html.escape(value)}" type="button">{html.escape(label)}</button>'
        )
        for value, label in options
    )


def build_entry_html(
    entry_id: str,
    lang1_text: str,
    lang1_code: str,
    lang2_text: str,
    lang2_code: str,
    model: str,
    corpus: str,
    version: str,
):
    model_badge = (
        f'<span class="badge badge-neutral">{html.escape(model)}</span>' if model else ""
    )
    return f"""
        <article class="entry" data-custom-id="{html.escape(entry_id)}" data-src-code="{html.escape(lang1_code)}" data-tgt-code="{html.escape(lang2_code)}" data-model="{html.escape(model)}" data-corpus="{html.escape(corpus)}" data-version="{html.escape(version)}" data-review-state="pending">
            <div class="entry-header">
                <h2>{html.escape(entry_id)}</h2>
                <div class="badges">
                    {model_badge}
                    <span class="badge review-badge review-pending">incomplete</span>
                </div>
            </div>
            <div class="entry-content">
                <section class="panel">
                    <h3>{html.escape(lang1_code)}</h3>
                    <p class="text-block">{html.escape(lang1_text)}</p>
                </section>
                <section class="panel">
                    <h3>{html.escape(lang2_code)}</h3>
                    <p class="text-block">{html.escape(lang2_text)}</p>
                </section>
            </div>
            <section class="annotation-grid">
                <section class="choice-panel" data-field="src_correct_natural">
                    <h3>Is {html.escape(lang1_code)} language correct and natural?</h3>
                    <div class="choice-buttons">
                        {build_choice_buttons('src_correct_natural', FLUENCY_OPTIONS)}
                    </div>
                </section>
                <section class="choice-panel" data-field="tgt_correct_natural">
                    <h3>Is {html.escape(lang2_code)} language correct and natural?</h3>
                    <div class="choice-buttons">
                        {build_choice_buttons('tgt_correct_natural', FLUENCY_OPTIONS)}
                    </div>
                </section>
                <section class="choice-panel" data-field="parallelism">
                    <h3>Are they parallel?</h3>
                    <div class="choice-buttons">
                        {build_choice_buttons('parallelism', PARALLELISM_OPTIONS)}
                    </div>
                </section>
            </section>
        </article>
    """


HTML_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>$report_title</title>
    <style>
        :root {
            --bg: #f5f1e8;
            --panel: rgba(255, 252, 246, 0.94);
            --panel-border: rgba(64, 48, 35, 0.12);
            --text: #201913;
            --muted: #69584d;
            --accent: #0c5c51;
            --shadow: 0 18px 40px rgba(54, 39, 26, 0.08);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            color: var(--text);
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at top left, rgba(12, 92, 81, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 24%),
                linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
        }

        main {
            width: min(1100px, calc(100% - 24px));
            margin: 0 auto;
            padding: 20px 0 32px;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255, 252, 246, 0.95), rgba(246, 239, 228, 0.92));
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
            margin-bottom: 16px;
        }

        .hero h1 {
            margin: 0 0 6px;
            font-family: "IBM Plex Serif", Georgia, serif;
            font-size: clamp(1.6rem, 3vw, 2.2rem);
            line-height: 1.1;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
        }

        .controls {
            background: rgba(255, 252, 246, 0.96);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            display: grid;
            gap: 8px;
            padding: 10px 12px;
            margin-bottom: 14px;
            position: sticky;
            top: 10px;
            z-index: 20;
            backdrop-filter: blur(10px);
        }

        .controls-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 8px 14px;
            align-items: start;
        }

        .control-actions {
            display: flex;
            gap: 6px;
            align-items: center;
            flex-wrap: nowrap;
        }

        .control-metrics {
            display: flex;
            gap: 6px 12px;
            align-items: center;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .control-metric {
            display: inline-flex;
            gap: 4px;
            align-items: baseline;
            white-space: nowrap;
            font-size: 0.78rem;
            color: var(--muted);
        }

        .metric-value {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text);
        }

        .control-status {
            font-size: 0.76rem;
            color: var(--muted);
        }

        .button {
            appearance: none;
            border: 1px solid transparent;
            border-radius: 999px;
            padding: 6px 10px;
            font: inherit;
            font-weight: 600;
            font-size: 0.9rem;
            white-space: nowrap;
            cursor: pointer;
            transition: transform 120ms ease, background 120ms ease;
        }

        .button:hover { transform: translateY(-1px); }

        .button-primary {
            background: var(--accent);
            color: white;
        }

        .button-secondary {
            background: rgba(255, 255, 255, 0.78);
            border-color: rgba(32, 25, 19, 0.12);
            color: var(--text);
        }

        .entries {
            display: grid;
            gap: 12px;
        }

        .entry {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 14px;
        }

        .entry[data-review-state="complete"] {
            border-color: rgba(12, 92, 81, 0.35);
        }

        .entry[data-review-state="in-progress"] {
            border-color: rgba(180, 83, 9, 0.28);
        }

        .entry-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 10px;
        }

        .entry-header h2 {
            margin: 0;
            font-size: 1.05rem;
            font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
            word-break: break-word;
        }

        .badges {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            justify-content: flex-end;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .badge-neutral { background: rgba(148, 163, 184, 0.14); color: #334155; }
        .review-pending { background: rgba(148, 163, 184, 0.16); color: #334155; }
        .review-progress { background: rgba(180, 83, 9, 0.12); color: #9a3412; }
        .review-complete { background: rgba(12, 92, 81, 0.14); color: var(--accent); }

        .entry-content {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 10px;
        }

        .panel,
        .choice-panel {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(64, 48, 35, 0.08);
            border-radius: 14px;
            padding: 10px;
        }

        .panel h3,
        .choice-panel h3 {
            margin: 0 0 6px;
            font-size: 0.8rem;
            /* text-transform: uppercase; */
            letter-spacing: 0.06em;
            color: var(--muted);
        }

        .text-block {
            margin: 0;
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 0.95rem;
        }

        .annotation-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
        }

        .choice-buttons {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .choice-btn {
            appearance: none;
            border: 2px solid rgba(32, 25, 19, 0.14);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.82);
            padding: 9px 14px;
            cursor: pointer;
            font: inherit;
            color: var(--text);
            font-weight: 600;
            min-width: 88px;
            transition: transform 100ms ease, background 100ms ease, border-color 100ms ease;
        }

        .choice-btn:hover {
            transform: translateY(-2px);
            background: rgba(12, 92, 81, 0.07);
            border-color: var(--accent);
        }

        .choice-btn.selected {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }

        @media (max-width: 900px) {
            .annotation-grid { grid-template-columns: 1fr; }
        }

        @media (max-width: 640px) {
            .entry-content { grid-template-columns: 1fr; }
            .controls-top { grid-template-columns: 1fr; }
            .control-metrics { justify-content: flex-start; }
            .entry-header { flex-direction: column; }
            .badges { justify-content: flex-start; }
        }
    </style>
</head>
<body>
    <main>
        <section class="hero">
            <h1>$report_title</h1>
            <h2>Annotation Instructions</h2>
    			<ul>
				<li> For each review item, you will see:
					<ul>
						<li>A pair of sentences in two languages (codes).</li>
					</ul>
				</li>
				<li> Your task is to review the sentences in both languages.</li>
				<li> Checklist:
					<ul>
						<li>Check if the content of each sentence is in the <b>language denoted by the code</b> and if the sentence is <b>natural and fluent</b>.</li>
						<li>Check if the sentences are <b>parallel</b>, i.e., if they are translations of each other.</li>
					</ul>
				</li>
				<li> Your progress should be autosaved in your browser, but it's not sent back to the server. To be safe, you can save a partially done annotation by clicking <b>Download JSON</b>; you can load it via <b>Load JSON</b> to continue the review. Once you are done with the entire annotation, download the JSON file and send it back.</li>
				</ul>
        </section>

        <section class="controls">
            <div class="controls-top">
                <div class="control-actions">
                    <button id="export-annotations" class="button button-primary" type="button">Download JSON</button>
                    <button id="import-annotations" class="button button-secondary" type="button">Load JSON</button>
                    <button id="clear-annotations" class="button button-secondary" type="button">Reset draft</button>
                    <input id="import-file" type="file" accept="application/json" hidden>
                </div>
                <div class="control-metrics" aria-label="Annotation progress">
                    <span class="control-metric"><span id="completed-count" class="metric-value">0 / $total_entries</span><span>complete</span></span>
                </div>
            </div>
            <div class="control-status">
                <span id="annotation-status" aria-live="polite">Draft autosaves in browser but not to the server. Download the JSON to save progress.</span>
            </div>
        </section>

        <section id="entries" class="entries">
            $entry_cards
        </section>
    </main>

    <script>
        const exportButton = document.getElementById('export-annotations');
        const importButton = document.getElementById('import-annotations');
        const clearButton = document.getElementById('clear-annotations');
        const importInput = document.getElementById('import-file');
        const annotationStatus = document.getElementById('annotation-status');
        const completedCount = document.getElementById('completed-count');
        const entries = Array.from(document.querySelectorAll('.entry'));
        const storageKey = $storage_key_json;
        const inputFileName = $input_file_json;
        const fields = ['src_correct_natural', 'tgt_correct_natural', 'parallelism'];
        let persistTimer = null;

        function normalizeBinaryValue(value) {
            if (value === true || value === 'true' || value === 'yes') return 'true';
            if (value === false || value === 'false' || value === 'no') return 'false';
            return null;
        }

        function normalizeParallelValue(value) {
            if (value === 'not_parallel' || value === 'partial' || value === 'full') return value;
            return null;
        }

        function normalizeFieldValue(field, value) {
            return field === 'parallelism' ? normalizeParallelValue(value) : normalizeBinaryValue(value);
        }

        function exportBinaryField(entry, field) {
            const value = getFieldValue(entry, field);
            if (value === null) return null;
            return value === 'true';
        }

        function getFieldValue(entry, field) {
            const raw = entry.dataset[field];
            return raw ? raw : null;
        }

        function setFieldValue(entry, field, value) {
            if (value === null) {
                delete entry.dataset[field];
            } else {
                entry.dataset[field] = value;
            }
        }

        function syncFieldButtons(entry, field) {
            const panel = entry.querySelector('.choice-panel[data-field="' + field + '"]');
            if (!panel) return;
            const selectedValue = getFieldValue(entry, field);
            for (const button of panel.querySelectorAll('.choice-btn')) {
                button.classList.toggle('selected', button.dataset.value === selectedValue);
            }
        }

        function updateEntryState(entry) {
            const answered = fields.filter((field) => getFieldValue(entry, field) !== null).length;
            if (answered === fields.length) {
                entry.dataset.reviewState = 'complete';
            } else if (answered > 0) {
                entry.dataset.reviewState = 'in-progress';
            } else {
                entry.dataset.reviewState = 'pending';
            }

            const badge = entry.querySelector('.review-badge');
            if (!badge) return;
            if (entry.dataset.reviewState === 'complete') {
                badge.textContent = 'complete';
                badge.className = 'badge review-badge review-complete';
            } else if (entry.dataset.reviewState === 'in-progress') {
                badge.textContent = 'in progress';
                badge.className = 'badge review-badge review-progress';
            } else {
                badge.textContent = 'incomplete';
                badge.className = 'badge review-badge review-pending';
            }
        }

        function refreshEntry(entry) {
            for (const field of fields) syncFieldButtons(entry, field);
            updateEntryState(entry);
        }

        function annotationForEntry(entry) {
            return {
                custom_id: entry.dataset.customId,
                lang1_code: entry.dataset.srcCode,
                lang2_code: entry.dataset.tgtCode,
                model: entry.dataset.model || null,
                corpus: entry.dataset.corpus || null,
                version: entry.dataset.version || null,
                src_correct_natural: exportBinaryField(entry, 'src_correct_natural'),
                tgt_correct_natural: exportBinaryField(entry, 'tgt_correct_natural'),
                parallelism: normalizeFieldValue('parallelism', getFieldValue(entry, 'parallelism')),
                completed: entry.dataset.reviewState === 'complete',
                updated_at: new Date().toISOString(),
            };
        }

        function buildPayload() {
            return {
                schema_version: 2,
                input_file: inputFileName,
                saved_at: new Date().toISOString(),
                annotations: entries.map(annotationForEntry),
            };
        }

        function saveDraft(statusMessage = 'Draft autosaved locally.') {
            const payload = buildPayload();
            try {
                localStorage.setItem(storageKey, JSON.stringify(payload));
                annotationStatus.textContent = statusMessage;
            } catch (_) {
                annotationStatus.textContent = 'Local draft save unavailable. Download the JSON before closing.';
            }
            return payload;
        }

        function schedulePersist() {
            if (persistTimer !== null) window.clearTimeout(persistTimer);
            persistTimer = window.setTimeout(() => {
                persistTimer = null;
                saveDraft();
                updateSummary();
            }, 180);
        }

        function updateSummary() {
            let completed = 0;
            for (const entry of entries) {
                if (entry.dataset.reviewState === 'complete') completed += 1;
            }
            completedCount.textContent = completed + ' / ' + entries.length;
        }

        function applyEntryAnnotation(entry, annotation) {
            for (const field of fields) {
                setFieldValue(entry, field, normalizeFieldValue(field, annotation[field]));
            }
            refreshEntry(entry);
        }

        function applyPayload(payload, statusMessage) {
            const rawAnnotations = Array.isArray(payload)
                ? payload
                : Array.isArray(payload && payload.annotations)
                    ? payload.annotations
                    : [];
            const byId = new Map(
                rawAnnotations
                    .filter((annotation) => annotation && annotation.custom_id)
                    .map((annotation) => [annotation.custom_id, annotation])
            );

            for (const entry of entries) {
                const annotation = byId.get(entry.dataset.customId);
                if (annotation) {
                    applyEntryAnnotation(entry, annotation);
                } else {
                    for (const field of fields) setFieldValue(entry, field, null);
                    refreshEntry(entry);
                }
            }

            annotationStatus.textContent = statusMessage;
            updateSummary();
        }

        function downloadPayload(payload) {
            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            const safeStem = inputFileName.replace(/\\.jsonl$$/i, '').replace(/[^a-z0-9._-]+/gi, '-');
            anchor.href = url;
            anchor.download = (safeStem || 'annotations') + '-annotations.json';
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(url);
        }

        function loadDraft() {
            try {
                const raw = localStorage.getItem(storageKey);
                if (!raw) {
                    updateSummary();
                    return;
                }
                applyPayload(JSON.parse(raw), 'Loaded saved draft from this browser.');
            } catch (_) {
                annotationStatus.textContent = 'Saved draft could not be loaded.';
                updateSummary();
            }
        }

        for (const entry of entries) {
            for (const button of entry.querySelectorAll('.choice-btn')) {
                button.addEventListener('click', () => {
                    const field = button.dataset.field;
                    const value = button.dataset.value;
                    const currentValue = getFieldValue(entry, field);
                    setFieldValue(entry, field, currentValue === value ? null : value);
                    refreshEntry(entry);
                    schedulePersist();
                    updateSummary();
                });
            }
        }

        exportButton.addEventListener('click', () => {
            const payload = saveDraft('Draft saved locally and exported.');
            downloadPayload(payload);
            updateSummary();
        });

        importButton.addEventListener('click', () => importInput.click());

        importInput.addEventListener('change', () => {
            const [file] = importInput.files;
            if (!file) return;
            file.text()
                .then((text) => applyPayload(JSON.parse(text), 'Annotations loaded from file.'))
                .then(() => saveDraft('Annotations loaded from file and saved locally.'))
                .catch(() => { annotationStatus.textContent = 'Could not read that annotations JSON file.'; })
                .finally(() => { importInput.value = ''; });
        });

        clearButton.addEventListener('click', () => {
            for (const entry of entries) {
                for (const field of fields) setFieldValue(entry, field, null);
                refreshEntry(entry);
            }
            try {
                localStorage.removeItem(storageKey);
                annotationStatus.textContent = 'Local draft cleared.';
            } catch (_) {
                annotationStatus.textContent = 'Draft reset in page, but browser storage could not be cleared.';
            }
            updateSummary();
        });

        loadDraft();
    </script>
</body>
</html>
"""
)


def render_report(input_file: Path, output_file: Path):
    rows = read_jsonl(input_file)
    if not rows:
        raise ValueError(f"No data found in {input_file}")

    inferred_lang1_code, inferred_lang2_code = parse_lang_codes_from_filename(input_file.name)

    entry_cards = []
    observed_lang1_code = None
    observed_lang2_code = None
    for index, row in enumerate(rows, start=1):
        # custom_id = str(row.get("id") or row.get("custom_id") or f"item_{index:04d}")
        # custom_id = str(row.get("id"))
        custom_id = f"item_{index:04d}"
        lang1_text = row.get("lang1_text", row.get("en", ""))
        lang2_text = row.get("lang2_text", row.get("target_text", ""))
        lang1_code = row.get("lang1_code", inferred_lang1_code or "SRC")
        lang2_code = row.get("lang2_code", inferred_lang2_code or "TGT")
        model = row.get("model", "")
        corpus = row.get("corpus", "")
        version = row.get("version", "")
        observed_lang1_code = observed_lang1_code or lang1_code
        observed_lang2_code = observed_lang2_code or lang2_code
        entry_cards.append(
            build_entry_html(
                custom_id,
                lang1_text,
                lang1_code,
                lang2_text,
                lang2_code,
                model,
                corpus,
                version,
            )
        )

    title_src = observed_lang1_code or inferred_lang1_code or "src"
    title_tgt = observed_lang2_code or inferred_lang2_code or "tgt"
    report_title = f"{title_src} - {title_tgt} annotation workspace"
    stylesheet_file = write_stylesheet(output_file.parent)
    rendered_html = HTML_TEMPLATE.substitute(
        report_title=html.escape(report_title),
        total_entries=len(entry_cards),
        entry_cards="\n".join(entry_cards),
        storage_key_json=json.dumps(f"annotation-draft::{input_file.name}"),
        input_file_json=json.dumps(input_file.name),
    )
    rendered_html = inject_stylesheet_link(rendered_html, stylesheet_file.name)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(rendered_html, encoding="utf-8")

    return {
        "output_file": output_file,
        "total": len(entry_cards),
        "lang1_code": observed_lang1_code or inferred_lang1_code,
        "lang2_code": observed_lang2_code or inferred_lang2_code,
    }


def render_index(reports, output_dir: Path):
    if not reports:
        return None

    index_file = output_dir / "index.html"
    stylesheet_file = write_stylesheet(output_dir)
    items = []
    for report in reports:
        name = report["output_file"].name
        items.append(
            f'<li><a href="{html.escape(name)}">{html.escape(name)}</a>'
            f'<span>{report["total"]} items</span></li>'
        )

    index_template = Template(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Annotation workspaces</title>
    <style>
        body {
            margin: 0;
            padding: 32px 16px;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            background: linear-gradient(180deg, #fbf7f0 0%, #f5f1e8 100%);
            color: #201913;
        }
        main {
            width: min(860px, 100%);
            margin: 0 auto;
            background: rgba(255, 252, 246, 0.94);
            border: 1px solid rgba(64, 48, 35, 0.12);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 18px 40px rgba(54, 39, 26, 0.08);
        }
        h1 { margin-top: 0; font-family: "IBM Plex Serif", Georgia, serif; }
        ul { list-style: none; padding: 0; margin: 20px 0 0; display: grid; gap: 12px; }
        li {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(64, 48, 35, 0.08);
        }
        a { color: #0c5c51; text-decoration: none; font-weight: 600; }
        a:hover { text-decoration: underline; }
        span { color: #69584d; }
    </style>
</head>
<body>
    <main>
        <h1>Contribute</h1>
        <p>Language pairs listed here are undergoing human evaluation. If you would like to contribute to a new language pair, please contact shaoxiong.ji@utu.fi.</p>
        
        <h1>Annotation workspaces</h1>
        <ul>$items</ul>
    </main>
</body>
</html>
"""
    )
    index_file.write_text(index_template.substitute(items="\n".join(items)), encoding="utf-8")
    index_html = index_file.read_text(encoding="utf-8")
    index_html = index_html.replace('<body>', '<body class="annotation-index">', 1)
    index_html = inject_stylesheet_link(index_html, stylesheet_file.name)
    index_file.write_text(index_html, encoding="utf-8")
    return index_file


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if input_path.is_file():
        input_files = [input_path]
        single_file_mode = True
    elif input_path.is_dir():
        input_files = sorted(input_path.glob("*.jsonl"))
        # input file should start with "shuffled_data_"
        input_files = [f for f in input_files if f.name.startswith("shuffled_data_")]
        single_file_mode = False
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not input_files:
        raise FileNotFoundError("No JSONL files found.")

    reports = []
    for input_file in input_files:
        if single_file_mode and output_path.suffix.lower() == ".html":
            output_file = output_path
        else:
            output_dir = output_path if output_path.suffix.lower() != ".html" else output_path.parent
            output_file = output_dir / report_filename_from_input(input_file.name)

        report = render_report(input_file, output_file)
        reports.append(report)
        print(f"Wrote {output_file}")

    if not single_file_mode:
        index_file = render_index(reports, output_path)
        if index_file is not None:
            print(f"Wrote {index_file}")


if __name__ == "__main__":
    main()
