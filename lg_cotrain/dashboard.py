"""Generate an interactive HTML results dashboard from experiment metrics."""

import argparse
import json
import sys
import statistics
from datetime import datetime
from pathlib import Path

# Support both `python -m lg_cotrain.dashboard` (relative imports)
# and `python lg_cotrain/dashboard.py` (direct execution on Windows/Linux).
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lg_cotrain.data_loading import CLASS_LABELS
    from lg_cotrain.run_all import BUDGETS, SEED_SETS
else:
    from .data_loading import CLASS_LABELS
    from .run_all import BUDGETS, SEED_SETS

# Kept for reference / backward compatibility. Dashboard auto-discovers events.
DEFAULT_EVENTS = [
    "california_wildfires_2018",
    "canada_wildfires_2016",
    "cyclone_idai_2019",
    "hurricane_dorian_2019",
    "hurricane_florence_2018",
    "hurricane_harvey_2017",
    "hurricane_irma_2017",
    "hurricane_maria_2017",
    "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]

# Keep old name as alias for backward compatibility
EVENTS = DEFAULT_EVENTS


def format_event_name(event):
    """Convert 'california_wildfires_2018' to 'California Wildfires 2018'."""
    return event.replace("_", " ").title()


def discover_events(metrics):
    """Extract sorted unique event names from metrics data."""
    return sorted(set(m["event"] for m in metrics))


def _has_metrics(path):
    """Check if *path* contains at least one metrics.json at expected depth."""
    return any(Path(path).glob("*/*/metrics.json"))


def discover_result_sets(results_root):
    """Discover result set sub-folders under *results_root*.

    Scans a 3-level hierarchy: ``model / experiment_type / experiment_name``.
    Returns a nested dict::

        {model: {exp_type: [(exp_name, path), ...]}}

    Empty experiment-type directories are included with an empty list so the
    dashboard can display them as placeholder tabs.
    """
    root = Path(results_root)
    hierarchy = {}

    if not root.is_dir():
        return hierarchy

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith((".", "_")):
            continue
        model_entry = {}
        for type_dir in sorted(model_dir.iterdir()):
            if not type_dir.is_dir() or type_dir.name.startswith((".", "_")):
                continue
            experiments = []
            for exp_dir in sorted(type_dir.iterdir()):
                if not exp_dir.is_dir() or exp_dir.name.startswith((".", "_")):
                    continue
                if _has_metrics(exp_dir):
                    experiments.append((exp_dir.name, str(exp_dir)))
            model_entry[type_dir.name] = experiments
        if model_entry:
            hierarchy[model_dir.name] = model_entry

    return hierarchy


def collect_all_metrics(results_root):
    """Scan *results_root* for all metrics.json files, auto-discovering events.

    Silently skips malformed files.
    """
    metrics = []
    root = Path(results_root)
    for metrics_file in sorted(root.glob("*/*/metrics.json")):
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            metrics.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return metrics


def _count_tsv_labels(path):
    """Read a TSV file and return {class_label: count} using only stdlib csv."""
    import csv
    counts = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if "class_label" not in (reader.fieldnames or []):
                return {}
            for row in reader:
                label = row.get("class_label", "").strip()
                if label:
                    counts[label] = counts.get(label, 0) + 1
    except Exception:
        return {}
    return counts


def collect_data_stats(data_root):
    """Scan data_root/original/{event}/ TSV files and count class_label occurrences.

    Uses seed 1 as the representative for labeled/unlabeled subsets. Returns a nested
    dict: {event: {file_key: {class_label: count}}}. Missing or unreadable files produce
    an empty dict for that key. Returns {} if data_root/original/ does not exist.

    Uses pandas when available for speed; falls back to stdlib csv otherwise.
    """
    root = Path(data_root) / "original"
    if not root.is_dir():
        return {}

    try:
        import pandas as pd
        def _read(path):
            df = pd.read_csv(path, sep="\t", dtype={"tweet_id": str},
                             usecols=["class_label"])
            return df["class_label"].value_counts().to_dict()
    except ImportError:
        _read = _count_tsv_labels

    stats = {}
    for event_dir in sorted(root.iterdir()):
        if not event_dir.is_dir():
            continue
        event = event_dir.name
        file_map = {
            "train": event_dir / f"{event}_train.tsv",
            "dev":   event_dir / f"{event}_dev.tsv",
            "test":  event_dir / f"{event}_test.tsv",
        }
        for budget in BUDGETS:
            file_map[f"labeled_{budget}_set1"]   = event_dir / f"labeled_{budget}_set1.tsv"
            file_map[f"unlabeled_{budget}_set1"] = event_dir / f"unlabeled_{budget}_set1.tsv"

        event_stats = {}
        for key, path in file_map.items():
            if not path.exists():
                event_stats[key] = {}
                continue
            try:
                event_stats[key] = _read(path)
            except Exception:
                event_stats[key] = {}

        stats[event] = event_stats

    return stats


def count_expected_experiments(events=None):
    """Return total expected: len(events) * len(BUDGETS) * len(SEED_SETS)."""
    if events is None:
        events = DEFAULT_EVENTS
    return len(events) * len(BUDGETS) * len(SEED_SETS)


def get_event_class_count(metrics, events=None):
    """Derive class count per event from len(test_per_class_f1).

    Returns dict {event: int}. Defaults to len(CLASS_LABELS) for events
    with no results.
    """
    if events is None:
        events = discover_events(metrics) or DEFAULT_EVENTS
    counts = {}
    for m in metrics:
        event = m["event"]
        if event not in counts:
            per_class = m.get("test_per_class_f1", [])
            counts[event] = len(per_class) if per_class else len(CLASS_LABELS)
    # Fill in missing events
    for event in events:
        if event not in counts:
            counts[event] = len(CLASS_LABELS)
    return counts


def build_pivot_data(metrics, events=None):
    """Build pivot: {event: {budget: {f1_mean, f1_std, err_mean, err_std, count}}}.

    Groups by (event, budget), computes mean/std across seed sets.
    """
    if events is None:
        events = discover_events(metrics) or DEFAULT_EVENTS
    # Group
    groups = {}
    for m in metrics:
        key = (m["event"], m["budget"])
        groups.setdefault(key, []).append(m)

    pivot = {}
    for event in events:
        pivot[event] = {}
        for budget in BUDGETS:
            results = groups.get((event, budget), [])
            f1s = [r["test_macro_f1"] for r in results]
            errs = [r["test_error_rate"] for r in results]
            eces = [r["test_ece"] for r in results if "test_ece" in r]
            entry = {"count": len(results)}
            if len(f1s) >= 2:
                entry["f1_mean"] = statistics.mean(f1s)
                entry["f1_std"] = statistics.stdev(f1s)
                entry["err_mean"] = statistics.mean(errs)
                entry["err_std"] = statistics.stdev(errs)
            elif len(f1s) == 1:
                entry["f1_mean"] = f1s[0]
                entry["f1_std"] = None
                entry["err_mean"] = errs[0]
                entry["err_std"] = None
            else:
                entry["f1_mean"] = None
                entry["f1_std"] = None
                entry["err_mean"] = None
                entry["err_std"] = None
            # ECE (may be absent in older metrics)
            if len(eces) >= 2:
                entry["ece_mean"] = statistics.mean(eces)
                entry["ece_std"] = statistics.stdev(eces)
            elif len(eces) == 1:
                entry["ece_mean"] = eces[0]
                entry["ece_std"] = None
            else:
                entry["ece_mean"] = None
                entry["ece_std"] = None
            pivot[event][budget] = entry
    return pivot


def build_lambda_pivot(metrics, events=None):
    """Build lambda weight pivot: {event: {budget: {l1_mean, l2_mean, count}}}."""
    if events is None:
        events = discover_events(metrics) or DEFAULT_EVENTS
    groups = {}
    for m in metrics:
        key = (m["event"], m["budget"])
        groups.setdefault(key, []).append(m)

    pivot = {}
    for event in events:
        pivot[event] = {}
        for budget in BUDGETS:
            results = groups.get((event, budget), [])
            l1s = [r["lambda1_mean"] for r in results]
            l2s = [r["lambda2_mean"] for r in results]
            entry = {"count": len(results)}
            if l1s:
                entry["l1_mean"] = statistics.mean(l1s)
                entry["l2_mean"] = statistics.mean(l2s)
            else:
                entry["l1_mean"] = None
                entry["l2_mean"] = None
            pivot[event][budget] = entry
    return pivot


def build_overall_means(pivot, events=None):
    """Compute 'Mean (all disasters)' row: average across events per budget."""
    if events is None:
        events = list(pivot.keys())
    overall = {}
    for budget in BUDGETS:
        f1_means = []
        err_means = []
        ece_means = []
        for event in events:
            entry = pivot.get(event, {}).get(budget, {})
            if entry.get("f1_mean") is not None:
                f1_means.append(entry["f1_mean"])
                err_means.append(entry["err_mean"])
            if entry.get("ece_mean") is not None:
                ece_means.append(entry["ece_mean"])
        result = {"f1_mean": None, "err_mean": None, "ece_mean": None}
        if f1_means:
            result["f1_mean"] = statistics.mean(f1_means)
            result["err_mean"] = statistics.mean(err_means)
        if ece_means:
            result["ece_mean"] = statistics.mean(ece_means)
        overall[budget] = result
    return overall


def compute_summary_cards(metrics, events=None):
    """Compute the summary card values."""
    if events is None:
        events = discover_events(metrics) or DEFAULT_EVENTS
    total = count_expected_experiments(events)
    completed = len(metrics)
    pct = (100.0 * completed / total) if total > 0 else 0

    if metrics:
        avg_f1 = statistics.mean(m["test_macro_f1"] for m in metrics)
        avg_err = statistics.mean(m["test_error_rate"] for m in metrics)
    else:
        avg_f1 = None
        avg_err = None

    ece_vals = [m["test_ece"] for m in metrics if "test_ece" in m]
    avg_ece = statistics.mean(ece_vals) if ece_vals else None

    events_with_results = len(set(m["event"] for m in metrics))

    return {
        "completed": completed,
        "total": total,
        "pct": pct,
        "avg_f1": avg_f1,
        "avg_err": avg_err,
        "avg_ece": avg_ece,
        "disasters_done": events_with_results,
        "disasters_total": len(events),
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f6fa; color: #2d3436; }
header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
         color: #e8e8e8; padding: 28px 40px; }
header h1 { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
header p { font-size: 13px; opacity: 0.7; }
/* Generic tab-bar (used by single-result generate_html) */
.tab-bar { display: flex; gap: 0; padding: 0 40px; background: #1a1a2e;
           border-bottom: 2px solid #2d3adf; overflow-x: auto; }
.tab-bar .tab { padding: 12px 24px; cursor: pointer; font-size: 14px;
                font-weight: 600; background: transparent; border: none;
                color: #8888aa; transition: all 0.15s; white-space: nowrap; }
.tab-bar .tab.active { color: #fff; border-bottom: 3px solid #2d3adf;
                       background: rgba(45,58,223,0.1); }
.tab-bar .tab:hover:not(.active):not(.disabled) { color: #bbb; }
.tab-bar .tab.disabled { color: #555; cursor: default; opacity: 0.5; }
.tab-content { display: none; }
.tab-content.active { display: block; }
/* Level-1 tab bar (model selector — dark header) */
.tab-bar.level-1 { background: #1a1a2e; border-bottom: 2px solid #2d3adf; }
.tab-bar.level-1 .tab { color: #8888aa; }
.tab-bar.level-1 .tab.active { color: #fff; border-bottom: 3px solid #2d3adf;
                                background: rgba(45,58,223,0.1); }
/* Level-2 tab bar (experiment type — medium-dark) */
.tab-bar.level-2 { background: #252545; border-bottom: 2px solid #4a4aaf; }
.tab-bar.level-2 .tab { padding: 10px 20px; font-size: 13px; color: #9999bb; }
.tab-bar.level-2 .tab.active { color: #e8e8e8; border-bottom: 3px solid #4a4aaf;
                                background: rgba(74,74,175,0.15); }
.tab-bar.level-2 .tab:hover:not(.active):not(.disabled) { color: #ccc; }
/* Level-3 tab bar (experiment name — light) */
.tab-bar.level-3 { background: #f0f0f5; border-bottom: 2px solid #dee2e6; }
.tab-bar.level-3 .tab { padding: 8px 18px; font-size: 13px; font-weight: 500;
                         color: #636e72; }
.tab-bar.level-3 .tab.active { color: #2d3436; border-bottom: 3px solid #2d3adf;
                                background: rgba(45,58,223,0.05); font-weight: 600; }
.tab-bar.level-3 .tab:hover:not(.active) { color: #2d3436; background: #e8e8f0; }
/* Nested content pane visibility */
.l1-content { display: none; }
.l1-content.active { display: block; }
.l2-content { display: none; }
.l2-content.active { display: block; }
.l3-content { display: none; }
.l3-content.active { display: block; }
.cards { display: flex; gap: 20px; padding: 24px 40px; flex-wrap: wrap; }
.card { background: #fff; border: 1px solid #dee2e6; border-radius: 10px;
        padding: 20px 28px; min-width: 180px; flex: 1; }
.card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
               color: #636e72; margin-bottom: 8px; }
.card .value { font-size: 32px; font-weight: 700; color: #2d3436; }
.card .sub { font-size: 12px; color: #636e72; margin-top: 4px; }
.controls { padding: 12px 40px; display: flex; align-items: center; gap: 16px;
            flex-wrap: wrap; }
.toggle-group { display: flex; border: 1px solid #dee2e6; border-radius: 6px;
                overflow: hidden; }
.toggle-btn { padding: 8px 20px; cursor: pointer; font-size: 13px; font-weight: 600;
              background: #fff; border: none; color: #636e72;
              transition: all 0.15s; }
.toggle-btn.active { background: #2d3adf; color: #fff; }
.toggle-btn:hover:not(.active) { background: #f0f0f0; }
.legend { display: flex; gap: 16px; margin-left: auto; font-size: 12px;
          color: #636e72; align-items: center; }
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-dot { width: 12px; height: 12px; border-radius: 3px; }
.content { padding: 0 40px 40px; }
.table-section { margin-top: 28px; }
.table-section h2 { font-size: 16px; font-weight: 600; margin-bottom: 4px; }
.table-section .hint { font-size: 12px; color: #636e72; margin-bottom: 12px; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
thead th { background: #1a1a2e; color: #e8e8e8; font-size: 12px;
           font-weight: 600; padding: 10px 14px; text-align: center;
           white-space: nowrap; }
thead th:first-child { text-align: left; }
tbody td { padding: 10px 14px; font-size: 13px; text-align: center;
           border-bottom: 1px solid #f0f0f0; }
tbody td:first-child { text-align: left; font-weight: 500; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #f8f9ff; }
tr.mean-row { font-weight: 700; border-top: 2px solid #2d3adf; }
tr.mean-row td { background: #f8f9ff; }
.cell-high   { background: #d4edda; color: #155724; }
.cell-mid    { background: #e8f5e9; color: #2e7d32; }
.cell-low    { background: #fff3cd; color: #856404; }
.cell-vlow   { background: #f8d7da; color: #721c24; }
.cell-pending { background: #e9ecef; color: #6c757d; }
.std { font-size: 11px; opacity: 0.65; }
.sortable { cursor: pointer; user-select: none; }
.sortable:hover { background: #2a2a4e; }
.sortable::after { content: ' \\2195'; opacity: 0.4; }
footer { text-align: center; padding: 20px; font-size: 12px; color: #b2bec3; }
"""

_JS = """\
/* --- Single-result (generate_html) backward-compat --- */
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(function(c) {
        c.classList.remove('active');
    });
    document.querySelectorAll('.tab-bar .tab').forEach(function(t) {
        t.classList.remove('active');
    });
    var el = document.getElementById('tab-' + tabId);
    if (el) { el.classList.add('active'); }
    var btn = document.querySelector('.tab-bar .tab[data-tab="' + tabId + '"]');
    if (btn) { btn.classList.add('active'); }
    showView(tabId, 'pivot');
}

/* --- Multi-level tab switching (generate_html_multi) --- */
function showL1Tab(tabId) {
    document.querySelectorAll('.l1-content').forEach(function(c) {
        c.classList.remove('active');
    });
    document.querySelectorAll('.tab-bar.level-1 .tab').forEach(function(t) {
        t.classList.remove('active');
    });
    var el = document.getElementById('l1-' + tabId);
    if (el) el.classList.add('active');
    var btn = document.querySelector('.tab-bar.level-1 .tab[data-tab="' + tabId + '"]');
    if (btn) btn.classList.add('active');
}

function showL2Tab(model, typeId) {
    var container = document.getElementById('l1-' + model);
    if (!container) return;
    container.querySelectorAll('.l2-content').forEach(function(c) {
        c.classList.remove('active');
    });
    container.querySelectorAll('.tab-bar.level-2 .tab').forEach(function(t) {
        t.classList.remove('active');
    });
    var l2Id = model + '--' + typeId;
    var el = document.getElementById('l2-' + l2Id);
    if (el) el.classList.add('active');
    var btn = container.querySelector('.tab-bar.level-2 .tab[data-tab="' + l2Id + '"]');
    if (btn) btn.classList.add('active');
}

function showL3Tab(model, typeId, expId) {
    var l2Id = model + '--' + typeId;
    var container = document.getElementById('l2-' + l2Id);
    if (!container) return;
    container.querySelectorAll('.l3-content').forEach(function(c) {
        c.classList.remove('active');
    });
    container.querySelectorAll('.tab-bar.level-3 .tab').forEach(function(t) {
        t.classList.remove('active');
    });
    var l3Id = model + '--' + typeId + '--' + expId;
    var el = document.getElementById('l3-' + l3Id);
    if (el) el.classList.add('active');
    var btn = container.querySelector('.tab-bar.level-3 .tab[data-tab="' + l3Id + '"]');
    if (btn) btn.classList.add('active');
    showView(l3Id, 'pivot');
}

function showView(tabId, v) {
    var tab = document.getElementById('tab-' + tabId) || document.getElementById('l3-' + tabId);
    if (!tab) return;
    var pivotEl = tab.querySelector('.pivot-view');
    var allEl = tab.querySelector('.all-view');
    if (pivotEl) pivotEl.style.display = v === 'pivot' ? 'block' : 'none';
    if (allEl) allEl.style.display = v === 'all' ? 'block' : 'none';
    tab.querySelectorAll('.toggle-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.view === v);
    });
}

var sortState = {};
function sortAllTable(tabId, col) {
    var key = tabId + '-' + col;
    var tbody = document.getElementById('all-tbody-' + tabId);
    if (!tbody) return;
    var rows = Array.from(tbody.querySelectorAll('tr'));
    if (sortState[key]) { sortState[key] = !sortState[key]; }
    else { sortState[key] = true; }
    var asc = sortState[key];
    rows.sort(function(a, b) {
        var va = a.cells[col].getAttribute('data-val') || a.cells[col].textContent;
        var vb = b.cells[col].getAttribute('data-val') || b.cells[col].textContent;
        var na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) { return asc ? na - nb : nb - na; }
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
}
"""


def _f1_class(val):
    """Return CSS class for an F1 value."""
    if val is None:
        return "cell-pending"
    if val >= 0.70:
        return "cell-high"
    if val >= 0.50:
        return "cell-mid"
    if val >= 0.30:
        return "cell-low"
    return "cell-vlow"


def _err_class(val):
    """Return CSS class for an error-rate value (lower is better)."""
    if val is None:
        return "cell-pending"
    if val < 20:
        return "cell-high"
    if val < 35:
        return "cell-mid"
    if val < 50:
        return "cell-low"
    return "cell-vlow"


def _ece_class(val):
    """Return CSS class for an ECE value (lower is better)."""
    if val is None:
        return "cell-pending"
    if val < 0.10:
        return "cell-high"
    if val < 0.20:
        return "cell-mid"
    if val < 0.35:
        return "cell-low"
    return "cell-vlow"


def _count_cell_style(count, max_count):
    """Return inline style string for heat-map coloring of a class-count cell."""
    if max_count == 0 or count == 0:
        return 'style="background:#f8f9fa; color:#adb5bd;"'
    ratio = count / max_count
    if ratio >= 0.60:
        return 'style="background:#cce5ff; color:#004085;"'
    if ratio >= 0.30:
        return 'style="background:#d4edda; color:#155724;"'
    if ratio >= 0.10:
        return 'style="background:#fff3cd; color:#856404;"'
    return 'style="background:#f8d7da; color:#721c24;"'


def _fmt_cell(mean, std, fmt, color_fn=None):
    """Format a pivot cell with optional +/-std."""
    if mean is None:
        return '<td class="cell-pending">---</td>'
    if color_fn is None:
        color_fn = _f1_class if fmt == ".3f" else _err_class
    css = color_fn(mean)
    val = f"{mean:{fmt}}"
    if std is not None:
        return f'<td class="{css}">{val} <span class="std">&plusmn;{std:{fmt}}</span></td>'
    return f'<td class="{css}">{val}</td>'


def _render_f1_table(pivot, overall, event_classes, events):
    """Render the Macro-F1 pivot table HTML."""
    rows = []
    for event in events:
        name = format_event_name(event)
        cls_count = event_classes.get(event, len(CLASS_LABELS))
        cells = f'<td>{name}</td><td>{cls_count}</td>'
        row_means = []
        for budget in BUDGETS:
            e = pivot[event][budget]
            cells += _fmt_cell(e["f1_mean"], e["f1_std"], ".3f")
            if e["f1_mean"] is not None:
                row_means.append(e["f1_mean"])
        # Row mean
        if row_means:
            rm = statistics.mean(row_means)
            cells += f'<td class="{_f1_class(rm)}"><b>{rm:.3f}</b></td>'
        else:
            cells += '<td class="cell-pending">---</td>'
        rows.append(f"<tr>{cells}</tr>")

    # Mean row
    mean_cells = '<td><b>Mean (all disasters)</b></td><td></td>'
    all_budget_means = []
    for budget in BUDGETS:
        o = overall[budget]
        if o["f1_mean"] is not None:
            mean_cells += f'<td class="{_f1_class(o["f1_mean"])}"><b>{o["f1_mean"]:.3f}</b></td>'
            all_budget_means.append(o["f1_mean"])
        else:
            mean_cells += '<td class="cell-pending">---</td>'
    if all_budget_means:
        gm = statistics.mean(all_budget_means)
        mean_cells += f'<td class="{_f1_class(gm)}"><b>{gm:.3f}</b></td>'
    else:
        mean_cells += '<td class="cell-pending">---</td>'
    rows.append(f'<tr class="mean-row">{mean_cells}</tr>')

    budget_hdrs = "".join(f"<th>{b} / class</th>" for b in BUDGETS)
    return f"""<div class="table-section">
<h2>Macro-F1 by Disaster &times; Label Size</h2>
<p class="hint">Values = mean &plusmn; std over 3 seed sets</p>
<table>
<thead><tr><th>Disaster</th><th>Classes</th>{budget_hdrs}<th>Mean</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>"""


def _render_err_table(pivot, overall, event_classes, events):
    """Render the Error Rate pivot table HTML."""
    rows = []
    for event in events:
        name = format_event_name(event)
        cls_count = event_classes.get(event, len(CLASS_LABELS))
        cells = f'<td>{name}</td><td>{cls_count}</td>'
        row_means = []
        for budget in BUDGETS:
            e = pivot[event][budget]
            cells += _fmt_cell(e["err_mean"], e["err_std"], ".2f")
            if e["err_mean"] is not None:
                row_means.append(e["err_mean"])
        if row_means:
            rm = statistics.mean(row_means)
            cells += f'<td class="{_err_class(rm)}"><b>{rm:.2f}</b></td>'
        else:
            cells += '<td class="cell-pending">---</td>'
        rows.append(f"<tr>{cells}</tr>")

    mean_cells = '<td><b>Mean (all disasters)</b></td><td></td>'
    all_budget_means = []
    for budget in BUDGETS:
        o = overall[budget]
        if o["err_mean"] is not None:
            mean_cells += f'<td class="{_err_class(o["err_mean"])}"><b>{o["err_mean"]:.2f}</b></td>'
            all_budget_means.append(o["err_mean"])
        else:
            mean_cells += '<td class="cell-pending">---</td>'
    if all_budget_means:
        gm = statistics.mean(all_budget_means)
        mean_cells += f'<td class="{_err_class(gm)}"><b>{gm:.2f}</b></td>'
    else:
        mean_cells += '<td class="cell-pending">---</td>'
    rows.append(f'<tr class="mean-row">{mean_cells}</tr>')

    budget_hdrs = "".join(f"<th>{b} / class</th>" for b in BUDGETS)
    return f"""<div class="table-section">
<h2>Error Rate (%) by Disaster &times; Label Size</h2>
<p class="hint">Values = mean &plusmn; std over 3 seed sets &middot; lower is better</p>
<table>
<thead><tr><th>Disaster</th><th>Classes</th>{budget_hdrs}<th>Mean</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>"""


def _render_ece_table(pivot, overall, event_classes, events):
    """Render the ECE pivot table HTML."""
    rows = []
    for event in events:
        name = format_event_name(event)
        cls_count = event_classes.get(event, len(CLASS_LABELS))
        cells = f'<td>{name}</td><td>{cls_count}</td>'
        row_means = []
        for budget in BUDGETS:
            e = pivot[event][budget]
            cells += _fmt_cell(e.get("ece_mean"), e.get("ece_std"), ".3f", _ece_class)
            if e.get("ece_mean") is not None:
                row_means.append(e["ece_mean"])
        if row_means:
            rm = statistics.mean(row_means)
            cells += f'<td class="{_ece_class(rm)}"><b>{rm:.3f}</b></td>'
        else:
            cells += '<td class="cell-pending">---</td>'
        rows.append(f"<tr>{cells}</tr>")

    mean_cells = '<td><b>Mean (all disasters)</b></td><td></td>'
    all_budget_means = []
    for budget in BUDGETS:
        o = overall[budget]
        if o.get("ece_mean") is not None:
            mean_cells += f'<td class="{_ece_class(o["ece_mean"])}"><b>{o["ece_mean"]:.3f}</b></td>'
            all_budget_means.append(o["ece_mean"])
        else:
            mean_cells += '<td class="cell-pending">---</td>'
    if all_budget_means:
        gm = statistics.mean(all_budget_means)
        mean_cells += f'<td class="{_ece_class(gm)}"><b>{gm:.3f}</b></td>'
    else:
        mean_cells += '<td class="cell-pending">---</td>'
    rows.append(f'<tr class="mean-row">{mean_cells}</tr>')

    budget_hdrs = "".join(f"<th>{b} / class</th>" for b in BUDGETS)
    return f"""<div class="table-section">
<h2>ECE by Disaster &times; Label Size</h2>
<p class="hint">Expected Calibration Error &middot; lower is better &middot;
mean &plusmn; std over seed sets</p>
<table>
<thead><tr><th>Disaster</th><th>Classes</th>{budget_hdrs}<th>Mean</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>"""


def _render_lambda_table(lambda_pivot, event_classes, events):
    """Render the Lambda Weights pivot table HTML."""
    rows = []
    for event in events:
        name = format_event_name(event)
        cls_count = event_classes.get(event, len(CLASS_LABELS))
        cells = f'<td>{name}</td><td>{cls_count}</td>'
        for budget in BUDGETS:
            e = lambda_pivot[event][budget]
            if e["l1_mean"] is not None:
                cells += (
                    f'<td>{e["l1_mean"]:.3f}</td>'
                    f'<td>{e["l2_mean"]:.3f}</td>'
                )
            else:
                cells += '<td class="cell-pending">---</td><td class="cell-pending">---</td>'
        rows.append(f"<tr>{cells}</tr>")

    budget_hdrs = "".join(
        f'<th>{b} &lambda;<sub>1</sub></th><th>{b} &lambda;<sub>2</sub></th>'
        for b in BUDGETS
    )
    return f"""<div class="table-section">
<h2>Lambda Weights by Disaster &times; Label Size</h2>
<p class="hint">&lambda;<sub>1</sub> = optimistic (confidence + variability) &middot;
&lambda;<sub>2</sub> = conservative (confidence &minus; variability) &middot;
mean across seed sets</p>
<table>
<thead><tr><th>Disaster</th><th>Classes</th>{budget_hdrs}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>"""


def _render_all_results(metrics, events, tab_id="default"):
    """Render the All Results flat table HTML."""
    # Build lookup for quick access
    lookup = {}
    for m in metrics:
        lookup[(m["event"], m["budget"], m["seed_set"])] = m

    rows = []
    idx = 0
    for event in events:
        for budget in BUDGETS:
            for seed_set in SEED_SETS:
                idx += 1
                m = lookup.get((event, budget, seed_set))
                name = format_event_name(event)
                if m is not None:
                    f1 = m["test_macro_f1"]
                    err = m["test_error_rate"]
                    f1_cls = _f1_class(f1)
                    err_cls = _err_class(err)
                    ece = m.get("test_ece")
                    if ece is not None:
                        ece_td = f'<td class="{_ece_class(ece)}" data-val="{ece:.6f}">{ece:.3f}</td>'
                    else:
                        ece_td = '<td class="cell-pending">---</td>'
                    rows.append(
                        f'<tr>'
                        f'<td>{idx}</td>'
                        f'<td>{name}</td>'
                        f'<td data-val="{budget}">{budget}</td>'
                        f'<td data-val="{seed_set}">{seed_set}</td>'
                        f'<td class="{f1_cls}" data-val="{f1:.6f}">{f1:.4f}</td>'
                        f'<td class="{err_cls}" data-val="{err:.4f}">{err:.2f}</td>'
                        f'{ece_td}'
                        f'<td data-val="{m["dev_macro_f1"]:.6f}">{m["dev_macro_f1"]:.4f}</td>'
                        f'<td data-val="{m["dev_error_rate"]:.4f}">{m["dev_error_rate"]:.2f}</td>'
                        f'<td data-val="{m["lambda1_mean"]:.6f}">{m["lambda1_mean"]:.3f}</td>'
                        f'<td data-val="{m["lambda2_mean"]:.6f}">{m["lambda2_mean"]:.3f}</td>'
                        f'</tr>'
                    )
                else:
                    rows.append(
                        f'<tr>'
                        f'<td>{idx}</td>'
                        f'<td>{name}</td>'
                        f'<td>{budget}</td>'
                        f'<td>{seed_set}</td>'
                        f'<td class="cell-pending">---</td>'
                        f'<td class="cell-pending">---</td>'
                        f'<td class="cell-pending">---</td>'
                        f'<td class="cell-pending">---</td>'
                        f'<td class="cell-pending">---</td>'
                        f'<td class="cell-pending">---</td>'
                        f'<td class="cell-pending">---</td>'
                        f'</tr>'
                    )

    total = count_expected_experiments(events)
    esc_tab = tab_id.replace('"', '&quot;')
    return f"""<div class="table-section">
<h2>All Experiment Results</h2>
<p class="hint">Showing {len(metrics)} of {total} experiments &middot;
click column headers to sort</p>
<table>
<thead><tr>
<th class="sortable" onclick="sortAllTable('{esc_tab}',0)">#</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',1)">Event</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',2)">Budget</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',3)">Seed</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',4)">Test Macro-F1</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',5)">Test Error %</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',6)">Test ECE</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',7)">Dev Macro-F1</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',8)">Dev Error %</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',9)">&lambda;<sub>1</sub> mean</th>
<th class="sortable" onclick="sortAllTable('{esc_tab}',10)">&lambda;<sub>2</sub> mean</th>
</tr></thead>
<tbody id="all-tbody-{esc_tab}">{"".join(rows)}</tbody>
</table></div>"""


def _render_data_guide():
    """Return the static Interpretation Guide HTML shown below the event tables."""

    # ── colour legend swatches ─────────────────────────────────────────────────
    def swatch(bg, fg, label):
        return (
            f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'padding:1px 8px;border-radius:3px;font-size:12px;margin-right:4px;">'
            f'<strong>{label}</strong></span>'
        )

    swatches = (
        swatch("#cce5ff", "#004085", "Blue — high (≥ 60% of max train count)")
        + swatch("#d4edda", "#155724", "Green — medium (30–59%)")
        + swatch("#fff3cd", "#856404", "Yellow — low (10–29%)")
        + swatch("#f8d7da", "#721c24", "Red — very low (< 10%)")
        + swatch("#f8f9fa", "#adb5bd", "Grey — 0 (absent)")
    )

    # ── shared sub-heading style ───────────────────────────────────────────────
    SH = 'style="font-size:15px;font-weight:700;margin:28px 0 8px 0;"'
    P  = 'style="font-size:13px;line-height:1.7;margin:0 0 10px 0;"'
    UL = 'style="font-size:13px;line-height:1.8;margin:0 0 12px 0;padding-left:24px;"'

    # ── callout box helper ─────────────────────────────────────────────────────
    def callout(bg, border, content):
        return (
            f'<div style="background:{bg};border-left:4px solid {border};'
            f'padding:12px 16px;border-radius:4px;font-size:13px;'
            f'line-height:1.7;margin:12px 0 4px 0;">{content}</div>'
        )

    # ── scenario heading helper ────────────────────────────────────────────────
    def scenario_heading(border_col, text):
        return (
            f'<p {SH} style="font-size:15px;font-weight:700;margin:28px 0 8px 0;'
            f'padding-left:12px;border-left:4px solid {border_col};">{text}</p>'
        )

    # ==========================================================================
    # Section 1 — How to read the table
    # ==========================================================================
    s1 = (
        f'<p {SH}>How to Read This Table</p>'
        f'<p {P}><strong>Column groups</strong></p>'
        f'<ul {UL}>'
        f'<li><strong>Train&nbsp;/&nbsp;Dev&nbsp;/&nbsp;Test</strong> — the full dataset splits '
        f'for this event (all available samples across all seed sets).</li>'
        f'<li><strong>L5, L10, L25, L50</strong> — the labeled training subset at the given '
        f'budget (target: that many samples <em>per class</em>), using seed&nbsp;1 as a '
        f'representative. This is the set the two BERT models are supervised-trained on in '
        f'Phase&nbsp;1 (weight generation) and Phase&nbsp;3 (fine-tuning).</li>'
        f'<li><strong>U5, U10, U25, U50</strong> — the unlabeled complement at each budget '
        f'(seed&nbsp;1). These are the tweets <em>excluded</em> from the labeled set. They are '
        f'paired with GPT-4o pseudo-labels to form D<sub>LG</sub>, the sole training set for '
        f'Phase&nbsp;2 co-training. A larger labeled budget means a smaller unlabeled '
        f'complement.</li>'
        f'</ul>'
        f'<p {P}><strong>Heat-map colouring</strong> — each cell is coloured relative to the '
        f'largest class count in the Train column for that event:</p>'
        f'<p style="margin:0 0 16px 0;">{swatches}</p>'
    )

    # ==========================================================================
    # Section 2 — Warning signs
    # ==========================================================================
    s2 = (
        f'<p {SH}>Warning Signs to Look For</p>'
        f'<ul {UL}>'
        f'<li><strong>L# &lt; budget for a class</strong> — the class does not have enough '
        f'samples to fill the requested budget. All available samples are used, but the labeled '
        f'set becomes imbalanced. <em>Example: budget&nbsp;=&nbsp;25 but the class only has 14 '
        f'training samples &rarr; L25 shows 14, not 25.</em></li>'
        f'<li><strong>L# stays the same across multiple budget levels</strong> — the class has '
        f'hit its natural ceiling; all available samples are already included. Increasing the '
        f'budget no longer adds real training data for that class. <em>Example: if both L25 and '
        f'L50 show 14, the class has exactly 14 samples in the training set.</em></li>'
        f'<li><strong>U# = 0 for a class</strong> — all samples of that class were consumed by '
        f'the labeled set; none remain for the unlabeled complement D<sub>LG</sub>. '
        f'Co-training in Phase&nbsp;2 receives no genuine examples of this class, only noise '
        f'from GPT-4o misclassifications.</li>'
        f'<li><strong>L# = Train count for a class</strong> — the entire training set for that '
        f'class is labeled. Combined with U#&nbsp;=&nbsp;0, all available data has been '
        f'exhausted; the algorithm has no headroom for semi-supervised learning on that '
        f'class.</li>'
        f'</ul>'
    )

    # ==========================================================================
    # Section 3 — Scenario 1: Unbalanced Labeled Set
    # ==========================================================================
    s3 = (
        scenario_heading("#004085", "Scenario&nbsp;1 — Unbalanced Labeled Set "
                         "(some L# &lt; budget)")
        + f'<p {P}><strong>All three training phases are degraded for the '
          f'underrepresented class.</strong> The core problem is that standard cross-entropy '
          f'loss treats every sample equally — majority classes dominate the gradient signal '
          f'because they appear more often per epoch.</p>'
        + f'<ul {UL}>'
        + f'<li><strong>Phase&nbsp;1 (Weight Generation) — unreliable lambda weights.</strong> '
          f'Model&nbsp;1 and Model&nbsp;2 are each trained on half the labeled set (D<sub>l1</sub> '
          f'and D<sub>l2</sub>). For a class with only 14 total samples each model sees roughly '
          f'7 examples — compared to hundreds for majority classes. With so few examples the '
          f'model\'s softmax probability for that class stays low and fluctuates unpredictably '
          f'across epochs. The <em>WeightTracker</em> records these noisy probabilities: '
          f'<em>confidence</em> (mean probability) is low and <em>variability</em> (std) is '
          f'inflated. The resulting lambda weights &mdash; which determine how much each '
          f'D<sub>LG</sub> sample contributes to Phase&nbsp;2 training &mdash; are either '
          f'near-zero (the sample is ignored) or erratic (the sample receives inconsistent '
          f'weight). Neither outcome is useful.</li>'
        + f'<li><strong>Phase&nbsp;2 (Co-Training) — a self-reinforcing feedback loop.</strong> '
          f'Lambda weights scale each sample\'s contribution to the co-training loss. Samples '
          f'pseudo-labeled as the rare class receive systematically low weights, so they '
          f'contribute little gradient, so the model does not improve on that class, so the '
          f'next epoch\'s weights remain low — a vicious cycle with no internal break. '
          f'For example, if <em>rescue_volunteering_or_donation_effort</em> has 653 training '
          f'samples and <em>requests_or_urgent_needs</em> has only 14, Phase&nbsp;2 learns an '
          f'excellent boundary for the former and a weak, uncertain one for the latter, '
          f'regardless of how many pseudo-labeled examples of the rare class exist in '
          f'D<sub>LG</sub>.</li>'
        + f'<li><strong>Phase&nbsp;3 (Fine-Tuning) — too little data to correct Phase&nbsp;2 '
          f'bias.</strong> Fine-tuning revisits only D<sub>l1</sub> and D<sub>l2</sub> — the '
          f'same small labeled set split in half again. Seven genuine examples cannot overcome '
          f'a poorly calibrated decision boundary built over many co-training epochs. Early '
          f'stopping compounds the problem: the overall dev macro-F1 may look acceptable '
          f'because all majority classes improved, causing early stopping to fire before the '
          f'minority class is properly learned.</li>'
        + f'</ul>'
        + callout(
            "#e8f4fd", "#3498db",
            f'<strong>Example — Canada Wildfires 2016, '
            f'<em>requests_or_urgent_needs</em>:</strong> '
            f'This class has only 14 samples in Train. At any budget &ge;&nbsp;25, all 14 are '
            f'consumed by the labeled set. Each model receives only ~7 examples, compared to '
            f'~326 for <em>rescue_volunteering_or_donation_effort</em> at the same budget. '
            f'The macro-F1 contribution from this class is consistently much lower than from '
            f'majority classes, anchoring the event\'s overall score below what a balanced '
            f'dataset would achieve.'
        )
    )

    # ==========================================================================
    # Section 4 — Scenario 2: Unbalanced Unlabeled Set
    # ==========================================================================
    s4 = (
        scenario_heading("#856404", "Scenario&nbsp;2 — Unbalanced Unlabeled Set "
                         "(some U# is very low or zero)")
        + f'<p {P}>D<sub>LG</sub> is the <em>exclusive</em> training data for Phase&nbsp;2. '
          f'Its class composition is therefore critical. Two distinct sub-cases arise.</p>'
        + f'<ul {UL}>'
        + f'<li><strong>Sub-case A — U# is low but &gt;&nbsp;0: weak but genuine '
          f'signal.</strong> Phase&nbsp;2 still receives real examples of the class with '
          f'(hopefully) correct pseudo-labels. The lambda weighting partially compensates '
          f'by amplifying high-confidence samples, but the proportionally small class '
          f'representation means the model underfits that class relative to majority classes. '
          f'Performance will be below ideal, but the learning direction is at least correct.</li>'
        + f'<li><strong>Sub-case B — U# = 0: co-training trains on noise, actively corrupting '
          f'the decision boundary.</strong> When no real samples of class&nbsp;C exist in '
          f'D<sub>LG</sub>, the only way class&nbsp;C appears there is through GPT-4o '
          f'misclassification errors — tweets from other classes that GPT-4o incorrectly '
          f'tagged as class&nbsp;C. Phase&nbsp;2 then uses these <em>false</em> pseudo-labels '
          f'as genuine training signal:'
          f'<ul style="margin-top:6px;margin-bottom:6px;">'
          f'<li>The cross-entropy loss pushes the model to classify those tweets as '
          f'class&nbsp;C.</li>'
          f'<li>Those tweets actually belong to other classes, so the model is learning the '
          f'wrong feature associations for class&nbsp;C.</li>'
          f'<li>The decision boundary for class&nbsp;C shifts toward the feature distributions '
          f'of whatever classes GPT-4o confused with it.</li>'
          f'</ul>'
          f'This is actively harmful — worse than simply ignoring the class. Phase&nbsp;3 '
          f'fine-tuning must both relearn the correct boundary <em>and</em> fight the '
          f'corrupted one from Phase&nbsp;2, armed with only a handful of genuine samples.</li>'
        + f'</ul>'
        + callout(
            "#fff8e1", "#f0a500",
            f'<strong>Example — Canada Wildfires 2016, '
            f'<em>requests_or_urgent_needs</em> at budget&nbsp;25/50:</strong> '
            f'U25&nbsp;=&nbsp;0 and U50&nbsp;=&nbsp;0. No genuine tweets of this class are '
            f'available for co-training. Any pseudo-labels tagged as '
            f'<em>requests_or_urgent_needs</em> in D<sub>LG</sub> come from other classes '
            f'that GPT-4o mislabelled — for instance, a '
            f'<em>rescue_volunteering_or_donation_effort</em> tweet containing "urgent" might '
            f'be mislabelled. The co-training model then learns to associate "urgent '
            f'volunteering appeals" with <em>requests_or_urgent_needs</em>, corrupting the '
            f'representation of both classes simultaneously.'
        )
    )

    # ==========================================================================
    # Section 5 — Scenario 3: Both Unbalanced (worst case)
    # ==========================================================================
    budget_table = (
        f'<div style="overflow-x:auto;margin:8px 0 12px 0;">'
        f'<table style="width:auto;border-collapse:collapse;font-size:13px;">'
        f'<thead><tr style="background:#2d3adf;color:#fff;">'
        f'<th style="padding:8px 16px;text-align:left;">Budget</th>'
        f'<th style="padding:8px 16px;text-align:center;">Labeled<br>samples</th>'
        f'<th style="padding:8px 16px;text-align:center;">Real samples<br>in D<sub>LG</sub></th>'
        f'<th style="padding:8px 16px;text-align:left;">Co-training signal</th>'
        f'</tr></thead><tbody>'
        f'<tr><td style="padding:7px 16px;border-bottom:1px solid #dee2e6;">L5</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;text-align:center;">5</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;text-align:center;">9</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;color:#155724;">'
        f'&#10003; 9 genuine examples available</td></tr>'
        f'<tr><td style="padding:7px 16px;border-bottom:1px solid #dee2e6;">L10</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;text-align:center;">10</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;text-align:center;">4</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;color:#856404;">'
        f'&#126; 4 genuine examples (signal shrinking)</td></tr>'
        f'<tr style="background:#fff8f8;">'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;">L25</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;text-align:center;">'
        f'14 (capped)</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;text-align:center;">0</td>'
        f'<td style="padding:7px 16px;border-bottom:1px solid #dee2e6;color:#721c24;">'
        f'&#10007; noise only &mdash; can be <em>worse</em> than L10</td></tr>'
        f'<tr style="background:#fff8f8;">'
        f'<td style="padding:7px 16px;">L50</td>'
        f'<td style="padding:7px 16px;text-align:center;">14 (capped)</td>'
        f'<td style="padding:7px 16px;text-align:center;">0</td>'
        f'<td style="padding:7px 16px;color:#721c24;">'
        f'&#10007; noise only &mdash; identical situation to L25</td></tr>'
        f'</tbody></table></div>'
    )

    s5 = (
        scenario_heading("#721c24", "Scenario&nbsp;3 — Both Labeled and Unlabeled Are "
                         "Unbalanced (the Worst Case)")
        + f'<p {P}><strong>All three phases reinforce each other\'s weaknesses. Recovery is '
          f'impossible for the affected class.</strong> This occurs when a class has too few '
          f'labeled samples (Scenario&nbsp;1) <em>and</em> no real samples in D<sub>LG</sub> '
          f'(Scenario&nbsp;2, sub-case B) simultaneously.</p>'
        + f'<ul {UL}>'
        + f'<li><strong>Phase&nbsp;1:</strong> Too few labeled samples &rarr; low, noisy '
          f'probabilities &rarr; unreliable lambda weights.</li>'
        + f'<li><strong>Phase&nbsp;2:</strong> Zero real samples in D<sub>LG</sub> &rarr; '
          f'trains entirely on GPT-4o misclassification noise &rarr; corrupted decision '
          f'boundary.</li>'
        + f'<li><strong>Phase&nbsp;3:</strong> Too few labeled samples to correct the '
          f'corruption accumulated in Phase&nbsp;2.</li>'
        + f'</ul>'
        + f'<p {P}><strong>The budget paradox — more data can produce lower '
          f'performance.</strong> As the budget grows, the labeled set expands but the '
          f'unlabeled complement shrinks. For a rare class this creates a non-monotonic '
          f'performance curve where macro-F1 at budget&nbsp;25 can be <em>lower</em> than at '
          f'budget&nbsp;10 for the same event. The table below uses '
          f'<em>requests_or_urgent_needs</em> in Canada Wildfires 2016 (14 total train '
          f'samples) as a concrete illustration:</p>'
        + budget_table
        + f'<p {P}>More labeled data does not always mean better performance when the '
          f'unlabeled complement is simultaneously depleted by that increase.</p>'
        + callout(
            "#fdf0f0", "#c0392b",
            f'<strong>Root cause — a violated semi-supervised learning assumption.</strong> '
            f'LG-CoTrain assumes the unlabeled data distribution reflects the true class '
            f'distribution. When D<sub>LG</sub> is constructed by excluding the labeled set '
            f'and a class is rare enough that the budget ceiling exhausts all of its '
            f'available samples, this assumption breaks completely for that class. The '
            f'algorithm cannot distinguish "this class is genuinely rare in the wild" from '
            f'"this class was artificially removed from the unlabeled pool by the '
            f'experimental design." The result: the pipeline can actively harm performance '
            f'on rare classes at higher budgets — a failure mode invisible from the results '
            f'tables alone, but clearly visible in the data distribution tables above.'
        )
    )

    # ==========================================================================
    # Assemble
    # ==========================================================================
    return (
        '<div class="table-section" style="margin-top:52px;padding-top:36px;'
        'border-top:3px solid #dee2e6;">'
        '<h2 style="font-size:18px;margin-bottom:4px;">Interpretation Guide</h2>'
        '<p class="hint">How to read the tables above and understand their impact '
        'on the LG-CoTrain pipeline</p>'
        + s1 + s2 + s3 + s4 + s5
        + '</div>'
    )


def _render_data_tab(data_stats):
    """Render the Data Analysis tab HTML: one class-count table per disaster event.

    Columns: Train | Dev | Test | L5 | L10 | L25 | L50 | U5 | U10 | U25 | U50
    Rows: only class labels present in that event's data (alphabetically sorted).
    Cells are heat-map coloured via _count_cell_style().
    """
    if not data_stats:
        return (
            '<div class="content">'
            '<p style="padding:40px 24px; color:#636e72;">'
            "No data found. Ensure the <code>data/original/</code> directory exists "
            "and is populated, or pass <code>--data-root</code> explicitly."
            "</p></div>"
        )

    COL_KEYS = [
        ("train",              "Train"),
        ("dev",                "Dev"),
        ("test",               "Test"),
        ("labeled_5_set1",     "L5"),
        ("labeled_10_set1",    "L10"),
        ("labeled_25_set1",    "L25"),
        ("labeled_50_set1",    "L50"),
        ("unlabeled_5_set1",   "U5"),
        ("unlabeled_10_set1",  "U10"),
        ("unlabeled_25_set1",  "U25"),
        ("unlabeled_50_set1",  "U50"),
    ]

    sections = []
    for event, file_stats in data_stats.items():
        event_name = format_event_name(event)

        # Collect all class labels present across any file for this event
        all_classes = set()
        for counts in file_stats.values():
            all_classes.update(counts.keys())
        sorted_classes = sorted(all_classes)

        if not sorted_classes:
            sections.append(
                f'<div class="table-section"><h2>{event_name}</h2>'
                f'<p class="hint">No data files found for this event.</p></div>'
            )
            continue

        # Heat-map reference: max count of any class in train; fallback to global max
        train_counts = file_stats.get("train", {})
        max_count = max(train_counts.values(), default=0) if train_counts else 0
        if max_count == 0:
            all_vals = [c for fc in file_stats.values() for c in fc.values()]
            max_count = max(all_vals, default=1)

        # Header row
        header_cells = "<th>Class Label</th>" + "".join(
            f"<th>{col_label}</th>" for _, col_label in COL_KEYS
        )

        # Data rows (one per class) + running totals
        totals = {key: 0 for key, _ in COL_KEYS}
        rows = []
        for cls in sorted_classes:
            cells = f"<td>{cls}</td>"
            for col_key, _ in COL_KEYS:
                count = file_stats.get(col_key, {}).get(cls, 0)
                totals[col_key] += count
                style = _count_cell_style(count, max_count)
                cells += f'<td {style} data-val="{count}">{count}</td>'
            rows.append(f"<tr>{cells}</tr>")

        # Total row
        total_cells = "<td><b>Total</b></td>" + "".join(
            f"<td><b>{totals[key]}</b></td>" for key, _ in COL_KEYS
        )
        rows.append(f'<tr class="mean-row">{total_cells}</tr>')

        hint = (
            "L# = Labeled set, seed&nbsp;1, budget&nbsp;#&emsp;"
            "U# = Unlabeled complement, seed&nbsp;1, budget&nbsp;#"
        )
        sections.append(
            f'<div class="table-section"><h2>{event_name}</h2>'
            f'<p class="hint">{hint}</p>'
            f"<div style='overflow-x:auto'>"
            f"<table>"
            f"<thead><tr>{header_cells}</tr></thead>"
            f'<tbody>{"".join(rows)}</tbody>'
            f"</table></div></div>"
        )

    return f'<div class="content">{"".join(sections)}{_render_data_guide()}</div>'


def _render_tab_content(metrics, results_root, tab_id="default"):
    """Render the full dashboard body for a single result set / tab."""
    events = discover_events(metrics) or DEFAULT_EVENTS
    pivot = build_pivot_data(metrics, events)
    overall = build_overall_means(pivot, events)
    lambda_pivot = build_lambda_pivot(metrics, events)
    summary = compute_summary_cards(metrics, events)
    event_classes = get_event_class_count(metrics, events)

    f1_table = _render_f1_table(pivot, overall, event_classes, events)
    err_table = _render_err_table(pivot, overall, event_classes, events)
    ece_table = _render_ece_table(pivot, overall, event_classes, events)
    lambda_table = _render_lambda_table(lambda_pivot, event_classes, events)
    all_table = _render_all_results(metrics, events, tab_id)

    avg_f1_str = f"{summary['avg_f1']:.3f}" if summary["avg_f1"] is not None else "N/A"
    avg_err_str = f"{summary['avg_err']:.2f}" if summary["avg_err"] is not None else "N/A"
    avg_ece_str = f"{summary['avg_ece']:.3f}" if summary.get("avg_ece") is not None else "N/A"

    esc_tab = tab_id.replace('"', '&quot;')

    return f"""
<div class="cards">
  <div class="card">
    <div class="label">Experiments</div>
    <div class="value">{summary['completed']} / {summary['total']}</div>
    <div class="sub">{summary['pct']:.0f}% complete</div>
  </div>
  <div class="card">
    <div class="label">Avg Macro-F1</div>
    <div class="value">{avg_f1_str}</div>
    <div class="sub">across {summary['completed']} completed runs</div>
  </div>
  <div class="card">
    <div class="label">Avg Error Rate</div>
    <div class="value">{avg_err_str}</div>
    <div class="sub">lower is better</div>
  </div>
  <div class="card">
    <div class="label">Avg ECE</div>
    <div class="value">{avg_ece_str}</div>
    <div class="sub">lower is better</div>
  </div>
  <div class="card">
    <div class="label">Disasters</div>
    <div class="value">{summary['disasters_done']} / {summary['disasters_total']}</div>
    <div class="sub">&times; {len(BUDGETS)} sizes &times; {len(SEED_SETS)} sets</div>
  </div>
</div>

<div class="controls">
  <div class="toggle-group">
    <button class="toggle-btn active" data-view="pivot" onclick="showView('{esc_tab}','pivot')">Pivot Summary</button>
    <button class="toggle-btn" data-view="all" onclick="showView('{esc_tab}','all')">All Results</button>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#28a745"></div> High F1 / Low Err</div>
    <div class="legend-item"><div class="legend-dot" style="background:#dc3545"></div> Low F1 / High Err</div>
    <div class="legend-item"><div class="legend-dot" style="background:#adb5bd"></div> Pending</div>
  </div>
</div>

<div class="content">
  <div class="pivot-view">
    {f1_table}
    {err_table}
    {ece_table}
    {lambda_table}
  </div>
  <div class="all-view" style="display:none">
    {all_table}
  </div>
</div>"""


def generate_html(metrics, results_root, data_root=None):
    """Generate complete self-contained HTML dashboard string (single result set).

    The dashboard has two tabs: Data Analysis (first, active) and Results.
    """
    if data_root is None:
        data_root = str(_find_repo_root() / "data")

    events = discover_events(metrics) or DEFAULT_EVENTS
    summary = compute_summary_cards(metrics, events)
    tab_content = _render_tab_content(metrics, results_root, "default")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_stats = collect_data_stats(data_root)
    data_tab_body = _render_data_tab(data_stats)

    tab_bar = (
        '<nav class="tab-bar">'
        '<button class="tab active" data-tab="data-analysis" '
        "onclick=\"showTab('data-analysis')\">Data Analysis</button>"
        '<button class="tab" data-tab="default" '
        "onclick=\"showTab('default')\">Results</button>"
        '</nav>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LG-CoTrain Results Dashboard</title>
<style>{_CSS}</style>
</head>
<body>

<header>
<h1>LG-CoTrain &mdash; Results Dashboard</h1>
<p>Semi-supervised co-training &middot; BERT &middot; HumAID dataset &middot;
{summary['disasters_total']} disasters &times; {len(BUDGETS) * len(SEED_SETS)} splits</p>
</header>

{tab_bar}

<div id="tab-data-analysis" class="tab-content active">
{data_tab_body}
</div>

<div id="tab-default" class="tab-content">
{tab_content}
</div>

<footer>Generated at {timestamp}</footer>

<script>{_JS}</script>
</body>
</html>"""


def _esc(name):
    """Escape a name for safe use in HTML attributes and JS strings."""
    return name.replace("&", "&amp;").replace('"', "&quot;").replace("'", "\\'")


def generate_html_multi(result_sets, data_root=None):
    """Generate multi-tab HTML dashboard with 3-level nested tabs.

    Args:
        result_sets: nested dict from :func:`discover_result_sets`::

            {model: {exp_type: [(exp_name, path), ...]}}

        data_root: path to the data/ directory. Auto-detected from repo root
            if *None*.
    """
    if data_root is None:
        data_root = str(_find_repo_root() / "data")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Data Analysis tab is always first and active
    data_stats = collect_data_stats(data_root)
    l1_buttons = [
        '<button class="tab active" data-tab="data-analysis" '
        "onclick=\"showL1Tab('data-analysis')\">Data Analysis</button>"
    ]
    l1_divs = [
        f'<div id="l1-data-analysis" class="l1-content active">\n'
        f'{_render_data_tab(data_stats)}\n</div>'
    ]

    total_metrics = 0

    for model, types in result_sets.items():
        esc_model = _esc(model)
        l1_buttons.append(
            f'<button class="tab" data-tab="{esc_model}" '
            f"onclick=\"showL1Tab('{esc_model}')\">{model}</button>"
        )

        # --- Level 2 tab bar (inside this model) ---
        l2_buttons = []
        l2_divs = []
        first_type = True
        for exp_type, experiments in types.items():
            esc_type = _esc(exp_type)
            l2_id = f"{esc_model}--{esc_type}"
            active_cls = " active" if first_type and experiments else ""
            disabled = " disabled" if not experiments else ""

            l2_buttons.append(
                f'<button class="tab{active_cls}{disabled}" data-tab="{l2_id}" '
                f"onclick=\"showL2Tab('{esc_model}','{esc_type}')\">"
                f'{exp_type}{"" if experiments else " (empty)"}</button>'
            )

            if not experiments:
                l2_divs.append(
                    f'<div id="l2-{l2_id}" class="l2-content">'
                    '<div class="content"><p style="padding:40px;color:#636e72;">'
                    'No experiments yet.</p></div></div>'
                )
                continue

            # --- Level 3 tab bar (inside this type) ---
            l3_buttons = []
            l3_divs = []
            first_exp = True
            for exp_name, exp_path in experiments:
                esc_exp = _esc(exp_name)
                l3_id = f"{esc_model}--{esc_type}--{esc_exp}"
                l3_active = " active" if first_exp else ""

                l3_buttons.append(
                    f'<button class="tab{l3_active}" data-tab="{l3_id}" '
                    f"onclick=\"showL3Tab('{esc_model}','{esc_type}','{esc_exp}')\">"
                    f'{exp_name}</button>'
                )

                metrics = collect_all_metrics(exp_path)
                total_metrics += len(metrics)
                content = _render_tab_content(metrics, exp_path, l3_id)
                l3_divs.append(
                    f'<div id="l3-{l3_id}" class="l3-content{l3_active}">\n'
                    f'{content}\n</div>'
                )
                first_exp = False

            l3_bar = f'<nav class="tab-bar level-3">{"".join(l3_buttons)}</nav>'
            l2_divs.append(
                f'<div id="l2-{l2_id}" class="l2-content{active_cls}">\n'
                f'{l3_bar}\n{"".join(l3_divs)}\n</div>'
            )
            if first_type and experiments:
                first_type = False

        l2_bar = f'<nav class="tab-bar level-2">{"".join(l2_buttons)}</nav>'
        l1_divs.append(
            f'<div id="l1-{esc_model}" class="l1-content">\n'
            f'{l2_bar}\n{"".join(l2_divs)}\n</div>'
        )

    l1_bar = f'<nav class="tab-bar level-1">{"".join(l1_buttons)}</nav>'

    n_models = len(result_sets)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LG-CoTrain Results Dashboard</title>
<style>{_CSS}</style>
</head>
<body>

<header>
<h1>LG-CoTrain &mdash; Results Dashboard</h1>
<p>Semi-supervised co-training &middot; BERT &middot; HumAID dataset &middot;
{n_models} model(s) &middot; {total_metrics} total experiments</p>
</header>

{l1_bar}

{"".join(l1_divs)}

<footer>Generated at {timestamp}</footer>

<script>{_JS}</script>
</body>
</html>"""


def _find_repo_root():
    """Find the repository root by looking for the lg_cotrain package directory."""
    for candidate in [Path.cwd()] + list(Path.cwd().parents):
        if (candidate / "lg_cotrain").is_dir():
            return candidate
    return Path.cwd()


def main():
    default_results = str(_find_repo_root() / "results")

    parser = argparse.ArgumentParser(
        description="Generate HTML results dashboard from experiment metrics"
    )
    parser.add_argument(
        "--results-root", type=str, default=default_results,
        help="Root directory containing event result subdirectories",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML path (default: {results_root}/dashboard.html)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help=(
            "Root data directory containing original/ subfolder "
            "(default: auto-detected from repo root)"
        ),
    )
    args = parser.parse_args()

    output = args.output or str(Path(args.results_root) / "dashboard.html")
    result_sets = discover_result_sets(args.results_root)

    if not result_sets:
        # No nested hierarchy — treat root as a single flat result set
        metrics = collect_all_metrics(args.results_root)
        html = generate_html(metrics, args.results_root, data_root=args.data_root)
        total = len(metrics)
        n_models = 0
    else:
        # Hierarchical result sets: 3-level nested tabs
        html = generate_html_multi(result_sets, data_root=args.data_root)
        total = sum(
            len(collect_all_metrics(path))
            for model_types in result_sets.values()
            for experiments in model_types.values()
            for _, path in experiments
        )
        n_models = len(result_sets)

    Path(output).write_text(html)
    print(f"Dashboard written to {output}")
    print(f"  {n_models} model(s), {total} total experiments")


if __name__ == "__main__":
    main()
