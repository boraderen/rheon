"""Generate three sudden-drift paper logs and one plot per perspective.

Run: uv run --with matplotlib python example/generate_paper_examples.py
Use --plots-only to rebuild plots and tables from saved logs without resampling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rheon

PAPER = ROOT / "paper"
BASE = dict(num_traces=1600, num_activities=5, num_resources=4, num_regions=3,
            tree_weights={"sequence": 1.0, "choice": 0.0, "parallel": 0.0, "loop": 0.0})
SCENARIOS = {
    "control_flow": {"seed": 107, "drift": {"type": "control_flow", "mode": "sudden", "drift_point": 0.5,
                    "num_activities": 5, "tree_weights": {"sequence": 0.6, "choice": 0.4, "parallel": 0.0, "loop": 0.0}}},
    "resource_reassignment": {"seed": 202, "params": {"num_resources": 3}, "drift": {"type": "reassignment", "mode": "sudden", "drift_point": 0.5}},
    "arrival_rate": {"seed": 303, "params": {"num_traces": 2400},
                     "drift": {"type": "arrival_rate", "mode": "sudden", "drift_point": 0.5, "factor": 20.0}},
}
CASE, ACT, START, END, DUR, RES = ("case:concept:name", "concept:name", "start_timestamp",
                                 "time:timestamp", "event:duration_min", "org:resource")
START_DATE, END_DATE = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")


def read_log(name):
    df = pd.read_csv(PAPER / "logs" / f"{name}.csv")
    for col in [START, END]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values([CASE, START], kind="stable").reset_index(drop=True)


def read_metadata(name):
    """Use the saved sidecar's drift point and resource mappings, as the notebooks do."""
    text = (PAPER / "logs" / f"{name}_meta.md").read_text().split("## Drifts", 1)[1]
    date = pd.Timestamp(re.search(r"- Drift point: [0-9.]+ \(([^)]+)\)", text)[1])
    maps = re.findall(r"\| ([a-z]+) \| (res_\d+) → (res_\d+) \|", text)
    return {"date": date, "dominant_before": {a: b for a, b, c in maps},
            "dominant_after": {a: c for a, b, c in maps}}


def cases(df):
    # Same case construction and chronological order as notebooks/intra.ipynb and inter.ipynb.
    return df.groupby(CASE).agg(start=(START, "min"), end=(END, "max"),
                               variant=(ACT, " ".join)).reset_index().sort_values("start").reset_index(drop=True)


def split_log(df, t, by="event"):
    """Partition the entire log: complete cases by arrival, events by processing start."""
    if by == "case":
        arrivals = df.groupby(CASE)[START].transform("min")
        return df[arrivals < t], df[arrivals >= t]
    return df[df[START] < t], df[df[START] >= t]


def dfg_counts(df, acts):
    nxt = df.groupby(CASE)[ACT].shift(-1)
    return pd.crosstab(df[ACT], nxt).reindex(index=acts, columns=acts, fill_value=0)


def save(fig, name):
    fig.savefig(PAPER / "figures" / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def count_comparison(before, after, name, xlabel, ylabel):
    """Compare absolute frequencies using the same cell order and color scale."""
    limit = max(1, int(max(before.to_numpy().max(), after.to_numpy().max())))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, matrix, label in zip(axes, [before, after], ["Before", "After"]):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=limit, aspect="auto")
        for i, j in np.ndindex(matrix.shape):
            val = int(matrix.iloc[i, j])
            ax.text(j, i, str(val), ha="center", va="center", fontsize=16,
                    color="white" if val > .55 * limit else "black")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set(title=label, xlabel=xlabel, ylabel=ylabel if ax is axes[0] else "")
    fig.colorbar(im, ax=axes, label="Count", shrink=.9)
    save(fig, name)


def control_flow(df, meta):
    t = meta["date"]
    acts = sorted(df[ACT].unique())
    pre, post = split_log(df, t, by="case")
    before, after = dfg_counts(pre, acts), dfg_counts(post, acts)
    count_comparison(before, after, "control_flow_counts", "To", "From")
    all_cases = cases(df)
    variants = [all_cases.loc[mask, "variant"].value_counts().to_dict()
                for mask in [all_cases.start < t, all_cases.start >= t]]
    if set(variants[0]) != {"a b c d e"} or set(variants[1]) != {"a b c", "a d e"}:
        raise ValueError("Generated control-flow models do not match the intended simple demonstration")
    return {"variants_before": variants[0], "variants_after": variants[1],
            "cases_before": pre[CASE].nunique(), "cases_after": post[CASE].nunique(),
            "df_counts_before": before.to_dict(), "df_counts_after": after.to_dict()}


def resource_reassignment(df, meta):
    pre, post = split_log(df, meta["date"])
    acts, resources = sorted(df[ACT].unique()), sorted(df[RES].unique())
    if resources != ["res_01", "res_02", "res_03"]:
        raise ValueError("The resource example must have exactly three resources")
    def act_res(sub):
        # Same contingency counts as the notebook's res_act, transposed to activity rows.
        return pd.crosstab(sub[ACT], sub[RES]).reindex(index=acts, columns=resources, fill_value=0)
    before, after = act_res(pre), act_res(post)
    count_comparison(before, after, "resource_reassignment", "Resource", "Activity")
    modes = [sub.groupby(ACT)[RES].agg(lambda s: s.value_counts().index[0]).to_dict()
             for sub in [df[df[START] < meta["date"]], df[df[START] >= meta["date"]]]]
    if modes != [meta["dominant_before"], meta["dominant_after"]]:
        raise ValueError("Observed resource dominants do not match the saved reassignment metadata")
    return {"dominant_before": meta["dominant_before"], "dominant_after": meta["dominant_after"],
            "events_before": len(pre), "events_after": len(post),
            "activity_resource_counts_before": before.to_dict(), "activity_resource_counts_after": after.to_dict()}


def arrival_rate(df, meta):
    c = cases(df)
    t = meta["date"]
    order = pd.Series(np.arange(1, len(c) + 1), index=c[CASE])
    acts = sorted(df[ACT].unique())
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    for a, color in zip(acts, ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]):
        sub = df[df[ACT] == a]
        ax.scatter(sub[END], sub[CASE].map(order), s=8, alpha=.55, color=color, label=a, linewidths=0)
    ax.axvline(t, color="crimson", ls="--", lw=1.5)
    ax.text(t + pd.Timedelta(days=5), .06 * len(c), "Sudden drift", color="crimson", fontsize=14)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set(xlim=(START_DATE, END_DATE), ylim=(0, len(c) + 25), title="Arrival rate: dense arrivals become sparse",
           xlabel="Event completion date (2020, UTC)", ylabel="Case, ordered by arrival")
    ax.legend(title="Activity", ncol=5, loc="upper left", frameon=True, fontsize=12, title_fontsize=12)
    ax.grid(alpha=.2)
    save(fig, "arrival_rate_dotted")
    gaps = c.start.diff().dt.total_seconds() / 3600
    before, after = gaps[c.start < t].dropna(), gaps[c.start >= t].dropna()
    n_before, n_after = int((c.start < t).sum()), int((c.start >= t).sum())
    if n_before < 10 * n_after or n_after < 20:
        raise ValueError("Arrival example is not sufficiently dense before and sparse after the drift")
    return {"cases_before": n_before, "cases_after": n_after,
            "mean_gap_hours_before": float(before.mean()), "mean_gap_hours_after": float(after.mean()),
            "configured_gap_factor": SCENARIOS["arrival_rate"]["drift"]["factor"]}


def tex_escape(value):
    return str(value).replace("_", r"\_")


def tables(logs, analysis):
    descriptions = [("Intra-case", r"control\_flow", "sequence to two short paths"),
                    ("Resource", "reassignment", "3 resources; new activity assignments"),
                    ("Inter-case", r"arrival\_rate", r"gap $\times 20$; rate drops to 5\%")]
    rows = []
    for (name, df), (perspective, drift, setting) in zip(logs.items(), descriptions):
        rows.append(f"{perspective} & \\texttt{{{drift}}} & {setting} & {SCENARIOS[name]['seed']} & {df[CASE].nunique()} & {len(df)} " + r"\\")
    (PAPER / "generated" / "scenario_rows.tex").write_text("\\newcommand{\\ScenarioRows}{%\n" + "\n".join(rows) + "\n}\n")
    rows = []
    for _, row in logs["control_flow"].head(6).iterrows():
        cells = [tex_escape(row["event:id"]), tex_escape(row[CASE]), row[ACT],
                 row[START].strftime("%Y-%m-%d %H:%M:%S"), row[END].strftime("%Y-%m-%d %H:%M:%S"),
                 f"{row[DUR]:.2f}", tex_escape(row[RES]), f"{row['case:amount']:.2f}", tex_escape(row['case:region'])]
        rows.append(" & ".join(cells) + r" \\")
    (PAPER / "generated" / "example_rows.tex").write_text("\\newcommand{\\ExampleRows}{%\n" + "\n".join(rows) + "\n}\n")
    arrival = analysis["arrival_rate"]
    values = {"ArrivalCasesBefore": str(arrival['cases_before']), "ArrivalCasesAfter": str(arrival['cases_after']),
              "ArrivalGapBefore": f"{arrival['mean_gap_hours_before']:.2f}",
              "ArrivalGapAfter": f"{arrival['mean_gap_hours_after']:.2f}"}
    res = analysis["resource_reassignment"]
    values["ResourceABefore"] = tex_escape(res["dominant_before"]["a"])
    values["ResourceAAfter"] = tex_escape(res["dominant_after"]["a"])
    (PAPER / "generated" / "measured_summary.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plots-only", action="store_true")
    mode.add_argument("--regenerate", choices=SCENARIOS, help="Resample only this log; preserve the others")
    args = parser.parse_args()
    for folder in ["logs", "figures", "generated"]:
        (PAPER / folder).mkdir(parents=True, exist_ok=True)
    regenerate = [] if args.plots_only else [args.regenerate] if args.regenerate else list(SCENARIOS)
    for name in regenerate:
        scenario = SCENARIOS[name]
        params = BASE | scenario.get("params", {})
        rheon.generate_log([scenario["drift"]], PAPER / "logs" / f"{name}.csv",
                           format="csv", seed=scenario["seed"], **params)
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14})
    logs = {name: read_log(name) for name in SCENARIOS}
    metadata = {name: read_metadata(name) for name in SCENARIOS}
    analysis = {"control_flow": control_flow(logs["control_flow"], metadata["control_flow"]),
                "resource_reassignment": resource_reassignment(logs["resource_reassignment"], metadata["resource_reassignment"]),
                "arrival_rate": arrival_rate(logs["arrival_rate"], metadata["arrival_rate"])}
    tables(logs, analysis)
    manifest_path = PAPER / "generated" / "generation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"base_parameters": BASE, "start_date": START_DATE.isoformat(), "end_date": END_DATE.isoformat(),
                    "versions": {p: importlib.metadata.version(p) for p in ["rheon", "pm4py", "numpy", "pandas", "matplotlib"]},
                    "scenarios": {}}
    for name, scenario in SCENARIOS.items():
        digest = hashlib.sha256((PAPER / "logs" / f"{name}.csv").read_bytes()).hexdigest()
        if name in regenerate:
            manifest["scenarios"][name] = {"seed": scenario["seed"], "parameters": BASE | scenario.get("params", {}),
                                           "drifts": [scenario["drift"]], "csv_sha256": digest}
        elif digest != manifest["scenarios"][name]["csv_sha256"]:
            raise ValueError(f"Saved log {name} has changed since generation")
    manifest["plot_logic"] = {"control_flow": "notebooks/intra.ipynb: full-log split by case arrival; directly-follows counts before normalization, separate pre/post matrices",
                               "resource_reassignment": "notebooks/resource.ipynb: full-log split by event start; activity-resource contingency counts, separate pre/post matrices",
                               "arrival_rate": "notebooks/inter.ipynb: chronological cases; all event completions plotted as dots"}
    manifest["analysis"] = analysis
    manifest_path.write_text(json.dumps(manifest, indent=2, default=lambda x: int(x)) + "\n")
    for name, df in logs.items():
        print(f"{name}: {df[CASE].nunique()} cases, {len(df)} events")
    print(json.dumps({n: {k: v for k, v in a.items() if not isinstance(v, dict)} for n, a in analysis.items()}, indent=2))


if __name__ == "__main__":
    main()
