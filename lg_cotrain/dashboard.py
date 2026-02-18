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

    Returns list of ``(name, path)`` tuples. If the root itself contains
    event directories with ``metrics.json`` (legacy flat layout), it is
    included as the "default" result set.
    """
    root = Path(results_root)
    result_sets = []

    # Check if root itself has the legacy flat layout
    if _has_metrics(root):
        result_sets.append(("default", str(root)))

    # Check sub-folders
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name != "__pycache__":
                if _has_metrics(child):
                    result_sets.append((child.name, str(child)))

    return result_sets if result_sets else [("default", str(root))]


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
.tab-bar { display: flex; gap: 0; padding: 0 40px; background: #1a1a2e;
           border-bottom: 2px solid #2d3adf; overflow-x: auto; }
.tab-bar .tab { padding: 12px 24px; cursor: pointer; font-size: 14px;
                font-weight: 600; background: transparent; border: none;
                color: #8888aa; transition: all 0.15s; white-space: nowrap; }
.tab-bar .tab.active { color: #fff; border-bottom: 3px solid #2d3adf;
                       background: rgba(45,58,223,0.1); }
.tab-bar .tab:hover:not(.active) { color: #bbb; }
.tab-content { display: none; }
.tab-content.active { display: block; }
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

function showView(tabId, v) {
    var tab = document.getElementById('tab-' + tabId);
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


def generate_html(metrics, results_root):
    """Generate complete self-contained HTML dashboard string (single result set)."""
    events = discover_events(metrics) or DEFAULT_EVENTS
    summary = compute_summary_cards(metrics, events)
    tab_content = _render_tab_content(metrics, results_root, "default")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

<div id="tab-default" class="tab-content active">
{tab_content}
</div>

<footer>Generated at {timestamp}</footer>

<script>{_JS}</script>
</body>
</html>"""


def generate_html_multi(result_sets):
    """Generate multi-tab HTML dashboard from multiple result sets.

    Args:
        result_sets: list of (name, path) tuples from discover_result_sets.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build tab bar
    tab_buttons = []
    tab_divs = []
    for i, (name, path) in enumerate(result_sets):
        active = " active" if i == 0 else ""
        esc_name = name.replace('"', '&quot;')
        tab_buttons.append(
            f'<button class="tab{active}" data-tab="{esc_name}" '
            f"onclick=\"showTab('{esc_name}')\">{name}</button>"
        )

        metrics = collect_all_metrics(path)
        content = _render_tab_content(metrics, path, name)
        tab_divs.append(
            f'<div id="tab-{esc_name}" class="tab-content{active}">\n{content}\n</div>'
        )

    tab_bar_html = f'<nav class="tab-bar">{"".join(tab_buttons)}</nav>'

    # Compute a global summary for the header subtitle
    total_metrics = sum(
        len(collect_all_metrics(path)) for _, path in result_sets
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
{len(result_sets)} result sets &middot; {total_metrics} total experiments</p>
</header>

{tab_bar_html}

{"".join(tab_divs)}

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
    args = parser.parse_args()

    output = args.output or str(Path(args.results_root) / "dashboard.html")
    result_sets = discover_result_sets(args.results_root)

    if len(result_sets) == 1:
        # Single result set: use single-page generation
        name, path = result_sets[0]
        metrics = collect_all_metrics(path)
        html = generate_html(metrics, path)
        total = len(metrics)
    else:
        # Multiple result sets: use multi-tab generation
        html = generate_html_multi(result_sets)
        total = sum(len(collect_all_metrics(p)) for _, p in result_sets)

    Path(output).write_text(html)
    print(f"Dashboard written to {output}")
    print(f"  {len(result_sets)} result set(s), {total} total experiments")


if __name__ == "__main__":
    main()
