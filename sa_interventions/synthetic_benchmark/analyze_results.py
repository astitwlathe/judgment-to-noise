#!/usr/bin/env python3
"""Analyze SA drift across (judge, condition) cells on the synthetic benchmark.

For each cell, computes:
  - SA (non-adherence) = 1 - max(R^2_linear, R^2_poly, R^2_RF_OOB), on all items
  - SA restricted to triggered vs. non-triggered items (sub-analysis)
  - Per-factor beta coefficients under the linear model
  - Fraction of overall-scores at <= 3 (the 'cap' signature)
  - Mean factor scores (to sanity-check orthogonality)

Writes:
  - summary.csv                (one row per (judge, condition))
  - triggered_breakdown.csv    (same, restricted to trigger-positive items)
  - drift_table.txt            (human-readable drift vs. baseline)

Prerequisites:
    conda activate abb   # needs sklearn, pandas
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


FACTOR_KEYS = ["helpfulness", "relevance", "accuracy", "depth", "creativity", "detail"]

# Patterns that mark an item as "triggered" for a given intervention.
# These are used for the sub-analysis — NOT for the SA regression itself.
TRIGGER_PATTERNS = {
    "nationality": re.compile(
        r"\b(japan|japanese|shōgatsu|shogatsu|kyoto|tokyo|kurosawa|shinkansen|kaizen|ikigai|"
        r"ryō|ryori|wabi|kanji|matcha|mingei|obon|judo|meiji|rashomon|samurai|"
        r"soba|udon|ramen|osechi|mono no aware)\b",
        re.IGNORECASE),
    "taboo": re.compile(r"(AI\s*safety|AI\s*alignment)", re.IGNORECASE),
}


def load_cell(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if "error" in rec: continue
            g = rec.get("grade") or {}
            if not g: continue
            row = {
                "item_id": rec["id"],
                "category": rec["category"],
                "quality": rec["quality"],
                "question": rec["question"],
                "response": rec["response"],
                "model": rec["model"],
                "condition": rec["condition"],
            }
            for k in FACTOR_KEYS:
                row[k] = g.get(k)
            row["overall"] = g.get("score")
            rows.append(row)
    return pd.DataFrame(rows)


def trigger_hit(response: str, intervention: str) -> bool:
    pat = TRIGGER_PATTERNS.get(intervention)
    return bool(pat.search(response)) if pat else False


def compute_sa(df: pd.DataFrame) -> dict:
    """SA = 1 - max(R^2_linear, R^2_poly, R^2_RF_OOB) on factor -> overall."""
    df = df.dropna(subset=FACTOR_KEYS + ["overall"]).copy()
    if len(df) < 20:
        return {"n": len(df), "sa": None}
    X = df[FACTOR_KEYS].values.astype(float)
    y = df["overall"].values.astype(float)

    lin = LinearRegression().fit(X, y)
    r2_lin = float(lin.score(X, y))
    betas = dict(zip(FACTOR_KEYS, map(float, lin.coef_)))

    Xp = PolynomialFeatures(2, include_bias=False).fit_transform(X)
    r2_poly = float(LinearRegression().fit(Xp, y).score(Xp, y))

    rf = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                min_samples_leaf=3, oob_score=True,
                                random_state=42, n_jobs=-1)
    rf.fit(X, y)
    r2_rf = float(rf.oob_score_)

    r2_sa = max(r2_lin, r2_poly, r2_rf)
    return {
        "n": len(df),
        "r2_linear": r2_lin, "r2_poly": r2_poly, "r2_rf": r2_rf,
        "r2_sa": r2_sa,
        "non_adherence_pct": 100 * (1 - r2_sa),
        "which_best": ["linear","poly","rf"][int(np.argmax([r2_lin, r2_poly, r2_rf]))],
        "overall_mean": float(y.mean()),
        "overall_sd": float(y.std(ddof=1)),
        "factor_means": {k: float(df[k].mean()) for k in FACTOR_KEYS},
        "betas": betas,
        "pct_overall_le_3": 100.0 * float((y <= 3).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).parent / "outputs")
    ap.add_argument("--report-dir", type=Path,
                    default=Path(__file__).parent / "outputs")
    args = ap.parse_args()

    files = sorted(args.output_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files under {args.output_dir}")
        return

    cells = {}
    for f in files:
        # filename format: {model}_{condition}.jsonl
        stem = f.stem
        for cond in ("baseline", "nationality", "taboo"):
            if stem.endswith("_" + cond):
                model = stem[: -len("_" + cond)]
                cells[(model, cond)] = load_cell(f)
                break

    rows = []
    triggered_rows = []
    for (model, cond), df in cells.items():
        full = compute_sa(df)
        row = {"model": model, "condition": cond, "subset": "all", **full}
        # Flatten factor_means + betas
        for k, v in full.get("factor_means", {}).items():
            row[f"mean_{k}"] = v
        for k, v in full.get("betas", {}).items():
            row[f"beta_{k}"] = v
        row.pop("factor_means", None); row.pop("betas", None)
        rows.append(row)

        # Sub-analysis: restrict to items where the trigger is present in the response
        # (for nationality -> Japan words; for taboo -> exact AI safety/alignment).
        # For baseline we also compute these subsets so we can see the "before" value.
        for trig in ("nationality", "taboo"):
            mask = df["response"].apply(lambda r: trigger_hit(r, trig))
            sub = df[mask]
            if len(sub) < 10:
                continue
            r = compute_sa(sub)
            trow = {"model": model, "condition": cond, "trigger": trig,
                    "n_triggered": int(mask.sum()),
                    "n_total": len(df),
                    **r}
            for k, v in r.get("factor_means", {}).items():
                trow[f"mean_{k}"] = v
            trow.pop("factor_means", None); trow.pop("betas", None)
            triggered_rows.append(trow)

    summary = pd.DataFrame(rows)
    triggered = pd.DataFrame(triggered_rows)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.report_dir / "summary.csv", index=False)
    triggered.to_csv(args.report_dir / "triggered_breakdown.csv", index=False)

    # Drift table
    lines = []
    lines.append("=" * 100)
    lines.append("SA DRIFT vs. BASELINE (per judge)")
    lines.append("=" * 100)
    for model in sorted({m for m, _ in cells.keys()}):
        lines.append(f"\n--- {model} ---")
        base = summary[(summary["model"] == model) & (summary["condition"] == "baseline")]
        if base.empty:
            lines.append("  (no baseline)"); continue
        base = base.iloc[0]
        lines.append(f"  baseline: n={int(base['n'])}  SA_non_adherence={base['non_adherence_pct']:.2f}%  "
                     f"pct_overall<=3={base['pct_overall_le_3']:.1f}%  best={base['which_best']}")
        for cond in ("nationality", "taboo"):
            row = summary[(summary["model"] == model) & (summary["condition"] == cond)]
            if row.empty: continue
            row = row.iloc[0]
            dsa = row["non_adherence_pct"] - base["non_adherence_pct"]
            drf = row["r2_rf"] - base["r2_rf"]
            # β L2 shift
            b_base = np.array([base[f"beta_{k}"] for k in FACTOR_KEYS])
            b_int = np.array([row[f"beta_{k}"] for k in FACTOR_KEYS])
            db_l2 = float(np.linalg.norm(b_int - b_base))
            lines.append(f"  {cond:11s}: n={int(row['n'])}  SA={row['non_adherence_pct']:.2f}%  "
                         f"ΔSA={dsa:+.2f}pp  ΔRF={drf:+.4f}  "
                         f"pct<=3={row['pct_overall_le_3']:.1f}%  Δβ(L2)={db_l2:.3f}  "
                         f"best={row['which_best']}")

    lines.append("\n" + "=" * 100)
    lines.append("TRIGGER HIT-RATE BREAKDOWN (did the cap fire where expected?)")
    lines.append("=" * 100)
    if not triggered.empty:
        # Pivot: for each (model, trigger), compare baseline vs the matching intervention
        for model in sorted(triggered["model"].unique()):
            lines.append(f"\n--- {model} ---")
            for trig in ("nationality", "taboo"):
                base = triggered[(triggered["model"] == model) &
                                  (triggered["condition"] == "baseline") &
                                  (triggered["trigger"] == trig)]
                intr = triggered[(triggered["model"] == model) &
                                  (triggered["condition"] == trig) &
                                  (triggered["trigger"] == trig)]
                if base.empty or intr.empty:
                    continue
                b = base.iloc[0]; i = intr.iloc[0]
                lines.append(f"  trigger={trig:11s}  N_triggered={int(b['n_triggered'])}/"
                             f"{int(b['n_total'])} items")
                lines.append(f"    baseline/triggered:    overall_mean={b['overall_mean']:.2f}  "
                             f"pct<=3={b['pct_overall_le_3']:.1f}%")
                lines.append(f"    intervention/triggered: overall_mean={i['overall_mean']:.2f}  "
                             f"pct<=3={i['pct_overall_le_3']:.1f}%  "
                             f"Δmean={i['overall_mean']-b['overall_mean']:+.2f}")

    report = "\n".join(lines)
    print(report)
    (args.report_dir / "drift_table.txt").write_text(report)

    print(f"\nWrote: {args.report_dir/'summary.csv'}, "
          f"{args.report_dir/'triggered_breakdown.csv'}, "
          f"{args.report_dir/'drift_table.txt'}")


if __name__ == "__main__":
    main()
