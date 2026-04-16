#!/usr/bin/env python3
"""Analyze SA drift under the 4 prompt-level interventions.

For each condition (baseline + 4 interventions), compute on GPT-4o-mini x MT-Bench:
  - N extracted rows, extraction rate
  - R^2 under linear, poly, RF (OOB)
  - SA = max of the three regressors; non-adherence = 1 - SA
  - Per-factor beta coefficients from the linear model (drift signature)
  - Score distribution moments (mean, sd) — sanity check that intervention
    actually changed the judge's output distribution

Writes a summary CSV + a JSON report with everything needed to replot.

Prerequisites:
    conda activate abb
    The baseline + 4 intervention ajudge jobs must be completed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, "/Users/benjaminfeuer/Documents/abb/bias-bounded-evaluation")
from differential_debiasing.benchmarks import MTBenchLoader  # noqa: E402


MT_BENCH_JOBS_DIR = Path("/Users/benjaminfeuer/Documents/abb/data/mt_bench/jobs")

CONDITIONS: dict[str, str | None] = {
    # label -> job-dir glob pattern (None = auto-detect from name)
    "baseline":       "mt-bench-factored-gpt4o-mini-context_dataset-*",
    "phrase":         "mt-bench-intervention-phrase-fixation-gpt4o-mini-context_dataset-*",
    "voice":          "mt-bench-intervention-voice-bias-gpt4o-mini-context_dataset-*",
    "lexicographic":  "mt-bench-intervention-lexicographic-gpt4o-mini-context_dataset-*",
    "combined":       "mt-bench-intervention-combined-gpt4o-mini-context_dataset-*",
}

SCORE_RANGE = (1.0, 10.0)


def _factor_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns
                  if c.endswith("_score") and c != "overall_score")


def resolve_job(pattern: str) -> Path:
    """Find the single job dir matching the pattern. Errors if 0 or >1."""
    matches = sorted(MT_BENCH_JOBS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No job dir found matching {pattern}")
    if len(matches) > 1:
        # Take the most recently modified — common pattern for repeat runs
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  [warn] {len(matches)} matches for {pattern}; using most recent: {matches[0].name}")
    return matches[0]


def compute_cell(df: pd.DataFrame, factor_cols: list[str]) -> dict:
    """Return SA, per-regressor R², extraction rate, β coefficients, distributions."""
    lo, hi = SCORE_RANGE
    # Clip and count
    df = df[df["overall_score"].between(lo, hi)].copy()
    for c in factor_cols:
        df.loc[~df[c].between(lo, hi), c] = np.nan

    n_total = len(df)
    dfB = df.dropna(subset=factor_cols + ["overall_score"])
    n_drop = len(dfB)
    extraction = n_drop / n_total if n_total else 0.0

    if n_drop < 20:
        return {
            "n_total": n_total, "n_extracted": n_drop,
            "extraction_rate": extraction, "error": "insufficient data",
        }

    X = dfB[factor_cols].values
    y = dfB["overall_score"].values

    # Linear
    lin = LinearRegression().fit(X, y)
    r2_lin = float(lin.score(X, y))
    betas = {fc: float(b) for fc, b in zip(factor_cols, lin.coef_)}
    betas["_intercept"] = float(lin.intercept_)

    # Polynomial (degree 2, with interactions)
    poly_X = PolynomialFeatures(2, include_bias=False).fit_transform(X)
    poly_mod = LinearRegression().fit(poly_X, y)
    r2_poly = float(poly_mod.score(poly_X, y))

    # Random Forest, out-of-bag score
    rf = RandomForestRegressor(
        n_estimators=200, max_features="sqrt",
        min_samples_leaf=5, oob_score=True,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    r2_rf = float(rf.oob_score_)

    r2_sa = max(r2_lin, r2_poly, r2_rf)
    sa_non_adherence_pct = 100.0 * (1.0 - r2_sa)

    return {
        "n_total": int(n_total),
        "n_extracted": int(n_drop),
        "extraction_rate": extraction,
        "r2_linear": r2_lin,
        "r2_poly": r2_poly,
        "r2_rf_oob": r2_rf,
        "r2_sa": r2_sa,
        "sa_non_adherence_pct": sa_non_adherence_pct,
        "which_best": (["linear", "poly", "rf"])[
            int(np.argmax([r2_lin, r2_poly, r2_rf]))
        ],
        "betas": betas,
        "overall_mean": float(y.mean()),
        "overall_sd": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "factor_means": {fc: float(X[:, i].mean()) for i, fc in enumerate(factor_cols)},
        "factor_sds": {fc: float(X[:, i].std(ddof=1)) for i, fc in enumerate(factor_cols)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "intervention_results.json")
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "intervention_results.csv")
    args = parser.parse_args()

    loader = MTBenchLoader()
    results: dict[str, dict] = {}

    for label, pattern in CONDITIONS.items():
        print(f"\n=== {label} ===")
        try:
            job_dir = resolve_job(pattern)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            results[label] = {"error": str(e)}
            continue
        print(f"  job: {job_dir.name}")
        df = loader.load_scores(job_dir)
        fcs = _factor_cols(df)
        cell = compute_cell(df, fcs)
        cell["job_dir"] = str(job_dir)
        cell["factor_cols"] = fcs
        results[label] = cell
        if "error" in cell:
            print(f"  ERROR: {cell['error']}")
            continue
        print(f"  N={cell['n_total']}  extracted={cell['n_extracted']}  extr={100*cell['extraction_rate']:.1f}%")
        print(f"  R²_lin={cell['r2_linear']:.4f}  R²_poly={cell['r2_poly']:.4f}  R²_RF={cell['r2_rf_oob']:.4f}")
        print(f"  SA_non_adherence = {cell['sa_non_adherence_pct']:.2f}%  (best: {cell['which_best']})")
        print(f"  overall mean={cell['overall_mean']:.2f}  sd={cell['overall_sd']:.2f}")

    # Drift signature: delta from baseline
    if "baseline" in results and "error" not in results["baseline"]:
        base = results["baseline"]
        print("\n=== Drift from baseline ===")
        print(f"{'Condition':16s} {'ΔSA':>8s} {'ΔRF':>8s} {'RF-poly':>9s} | {'Δβ (L2)':>8s} | {'Δmean':>8s}")
        for label, r in results.items():
            if label == "baseline" or "error" in r: continue
            dsa = r["sa_non_adherence_pct"] - base["sa_non_adherence_pct"]
            drf = r["r2_rf_oob"] - base["r2_rf_oob"]
            rf_minus_poly = r["r2_rf_oob"] - r["r2_poly"]
            # β L2 shift
            b_base = np.array([base["betas"][fc] for fc in r["factor_cols"]])
            b_int  = np.array([r["betas"][fc] for fc in r["factor_cols"]])
            db_l2 = float(np.linalg.norm(b_int - b_base))
            dmean = r["overall_mean"] - base["overall_mean"]
            print(f"{label:16s} {dsa:>+7.2f}% {drf:>+8.4f} {rf_minus_poly:>+9.4f} | "
                  f"{db_l2:>8.3f} | {dmean:>+7.2f}")

    args.out.parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nSaved JSON: {args.out}")

    # Flat CSV for quick viewing
    rows = []
    for label, r in results.items():
        if "error" in r:
            rows.append({"condition": label, "error": r["error"]}); continue
        rows.append({
            "condition": label,
            "n_total": r["n_total"],
            "n_extracted": r["n_extracted"],
            "extraction_rate_pct": 100 * r["extraction_rate"],
            "R2_linear": r["r2_linear"],
            "R2_poly": r["r2_poly"],
            "R2_RF_oob": r["r2_rf_oob"],
            "R2_SA": r["r2_sa"],
            "non_adherence_pct": r["sa_non_adherence_pct"],
            "best_regressor": r["which_best"],
            "overall_mean": r["overall_mean"],
            "overall_sd": r["overall_sd"],
        })
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"Saved CSV:  {args.csv}")


if __name__ == "__main__":
    main()
