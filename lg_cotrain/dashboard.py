"""Generate an interactive HTML results dashboard from experiment metrics."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from .data_loading import CLASS_LABELS
from .run_all import BUDGETS, SEED_SETS

EVENTS = [
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


def format_event_name(event):
    """Convert 'california_wildfires_2018' to 'California Wildfires 2018'."""
    return event.replace("_", " ").title()


def collect_all_metrics(results_root):
    """Scan results_root for all metrics.json files and return list of dicts.

    Silently skips missing or malformed files.
    """
    metrics = []
    root = Path(results_root)
    for event in EVENTS:
        for budget in BUDGETS:
            for seed_set in SEED_SETS:
                path = root / event / f"{budget}_set{seed_set}" / "metrics.json"
                if not path.exists():
                    continue
                try:
                    with open(path) as f:
                        data = json.load(f)
                    metrics.append(data)
                except (json.JSONDecodeError, OSError):
                    continue
    return metrics


def count_expected_experiments():
    """Return total expected: len(EVENTS) * len(BUDGETS) * len(SEED_SETS)."""
    return len(EVENTS) * len(BUDGETS) * len(SEED_SETS)


def get_event_class_count(metrics):
    """Derive class count per event from len(test_per_class_f1).

    Returns dict {event: int}. Defaults to len(CLASS_LABELS) for events
    with no results.
    """
    counts = {}
    for m in metrics:
        event = m["event"]
        if event not in counts:
            per_class = m.get("test_per_class_f1", [])
            counts[event] = len(per_class) if per_class else len(CLASS_LABELS)
    # Fill in missing events
    for event in EVENTS:
        if event not in counts:
            counts[event] = len(CLASS_LABELS)
    return counts


def build_pivot_data(metrics):
    """Build pivot: {event: {budget: {f1_mean, f1_std, err_mean, err_std, count}}}.

    Groups by (event, budget), computes mean/std across seed sets.
    """
    # Group
    groups = {}
    for m in metrics:
        key = (m["event"], m["budget"])
        groups.setdefault(key, []).append(m)

    pivot = {}
    for event in EVENTS:
        pivot[event] = {}
        for budget in BUDGETS:
            results = groups.get((event, budget), [])
            f1s = [r["test_macro_f1"] for r in results]
            errs = [r["test_error_rate"] for r in results]
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
            pivot[event][budget] = entry
    return pivot


def build_lambda_pivot(metrics):
    """Build lambda weight pivot: {event: {budget: {l1_mean, l2_mean, count}}}."""
    groups = {}
    for m in metrics:
        key = (m["event"], m["budget"])
        groups.setdefault(key, []).append(m)

    pivot = {}
    for event in EVENTS:
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


def build_overall_means(pivot):
    """Compute 'Mean (all disasters)' row: average across events per budget."""
    overall = {}
    for budget in BUDGETS:
        f1_means = []
        err_means = []
        for event in EVENTS:
            entry = pivot[event][budget]
            if entry["f1_mean"] is not None:
                f1_means.append(entry["f1_mean"])
                err_means.append(entry["err_mean"])
        if f1_means:
            overall[budget] = {
                "f1_mean": statistics.mean(f1_means),
                "err_mean": statistics.mean(err_means),
            }
        else:
            overall[budget] = {"f1_mean": None, "err_mean": None}
    return overall


def compute_summary_cards(metrics):
    """Compute the 4 summary card values."""
    total = count_expected_experiments()
    completed = len(metrics)
    pct = (100.0 * completed / total) if total > 0 else 0

    if metrics:
        avg_f1 = statistics.mean(m["test_macro_f1"] for m in metrics)
        avg_err = statistics.mean(m["test_error_rate"] for m in metrics)
    else:
        avg_f1 = None
        avg_err = None

    events_with_results = len(set(m["event"] for m in metrics))

    return {
        "completed": completed,
        "total": total,
        "pct": pct,
        "avg_f1": avg_f1,
        "avg_err": avg_err,
        "disasters_done": events_with_results,
        "disasters_total": len(EVENTS),
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
#all-view { display: none; }
"""

_JS = """\
function showView(v) {
    document.getElementById('pivot-view').style.display = v === 'pivot' ? 'block' : 'none';
    document.getElementById('all-view').style.display = v === 'all' ? 'block' : 'none';
    document.querySelectorAll('.toggle-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.view === v);
    });
}

var sortCol = null, sortAsc = true;
function sortAllTable(col) {
    var tbody = document.getElementById('all-tbody');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    if (sortCol === col) { sortAsc = !sortAsc; } else { sortCol = col; sortAsc = true; }
    rows.sort(function(a, b) {
        var va = a.cells[col].getAttribute('data-val') || a.cells[col].textContent;
        var vb = b.cells[col].getAttribute('data-val') || b.cells[col].textContent;
        var na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) { return sortAsc ? na - nb : nb - na; }
        return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
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


def _fmt_cell(mean, std, fmt):
    """Format a pivot cell with optional ±std."""
    if mean is None:
        return '<td class="cell-pending">---</td>'
    css = _f1_class(mean) if fmt == ".3f" else _err_class(mean)
    val = f"{mean:{fmt}}"
    if std is not None:
        return f'<td class="{css}">{val} <span class="std">&plusmn;{std:{fmt}}</span></td>'
    return f'<td class="{css}">{val}</td>'


def _render_f1_table(pivot, overall, event_classes):
    """Render the Macro-F1 pivot table HTML."""
    rows = []
    for event in EVENTS:
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


def _render_err_table(pivot, overall, event_classes):
    """Render the Error Rate pivot table HTML."""
    rows = []
    for event in EVENTS:
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


def _render_lambda_table(lambda_pivot, event_classes):
    """Render the Lambda Weights pivot table HTML."""
    rows = []
    for event in EVENTS:
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


def _render_all_results(metrics):
    """Render the All Results flat table HTML."""
    # Build lookup for quick access
    lookup = {}
    for m in metrics:
        lookup[(m["event"], m["budget"], m["seed_set"])] = m

    rows = []
    idx = 0
    for event in EVENTS:
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
                    rows.append(
                        f'<tr>'
                        f'<td>{idx}</td>'
                        f'<td>{name}</td>'
                        f'<td data-val="{budget}">{budget}</td>'
                        f'<td data-val="{seed_set}">{seed_set}</td>'
                        f'<td class="{f1_cls}" data-val="{f1:.6f}">{f1:.4f}</td>'
                        f'<td class="{err_cls}" data-val="{err:.4f}">{err:.2f}</td>'
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
                        f'</tr>'
                    )

    return f"""<div class="table-section">
<h2>All Experiment Results</h2>
<p class="hint">Showing {len(metrics)} of {count_expected_experiments()} experiments &middot;
click column headers to sort</p>
<table>
<thead><tr>
<th class="sortable" onclick="sortAllTable(0)">#</th>
<th class="sortable" onclick="sortAllTable(1)">Event</th>
<th class="sortable" onclick="sortAllTable(2)">Budget</th>
<th class="sortable" onclick="sortAllTable(3)">Seed</th>
<th class="sortable" onclick="sortAllTable(4)">Test Macro-F1</th>
<th class="sortable" onclick="sortAllTable(5)">Test Error %</th>
<th class="sortable" onclick="sortAllTable(6)">Dev Macro-F1</th>
<th class="sortable" onclick="sortAllTable(7)">Dev Error %</th>
<th class="sortable" onclick="sortAllTable(8)">&lambda;<sub>1</sub> mean</th>
<th class="sortable" onclick="sortAllTable(9)">&lambda;<sub>2</sub> mean</th>
</tr></thead>
<tbody id="all-tbody">{"".join(rows)}</tbody>
</table></div>"""


def generate_html(metrics, results_root):
    """Generate complete self-contained HTML dashboard string."""
    pivot = build_pivot_data(metrics)
    overall = build_overall_means(pivot)
    lambda_pivot = build_lambda_pivot(metrics)
    summary = compute_summary_cards(metrics)
    event_classes = get_event_class_count(metrics)

    f1_table = _render_f1_table(pivot, overall, event_classes)
    err_table = _render_err_table(pivot, overall, event_classes)
    lambda_table = _render_lambda_table(lambda_pivot, event_classes)
    all_table = _render_all_results(metrics)

    avg_f1_str = f"{summary['avg_f1']:.3f}" if summary["avg_f1"] is not None else "N/A"
    avg_err_str = f"{summary['avg_err']:.2f}" if summary["avg_err"] is not None else "N/A"
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
    <div class="label">Disasters</div>
    <div class="value">{summary['disasters_done']} / {summary['disasters_total']}</div>
    <div class="sub">&times; {len(BUDGETS)} sizes &times; {len(SEED_SETS)} sets</div>
  </div>
</div>

<div class="controls">
  <div class="toggle-group">
    <button class="toggle-btn active" data-view="pivot" onclick="showView('pivot')">Pivot Summary</button>
    <button class="toggle-btn" data-view="all" onclick="showView('all')">All Results</button>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#28a745"></div> High F1 / Low Err</div>
    <div class="legend-item"><div class="legend-dot" style="background:#dc3545"></div> Low F1 / High Err</div>
    <div class="legend-item"><div class="legend-dot" style="background:#adb5bd"></div> Pending</div>
  </div>
</div>

<div class="content">
  <div id="pivot-view">
    {f1_table}
    {err_table}
    {lambda_table}
  </div>
  <div id="all-view">
    {all_table}
  </div>
</div>

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
    metrics = collect_all_metrics(args.results_root)
    html = generate_html(metrics, args.results_root)

    Path(output).write_text(html)
    print(f"Dashboard written to {output}")
    print(f"  {len(metrics)}/{count_expected_experiments()} experiments found")


if __name__ == "__main__":
    main()
