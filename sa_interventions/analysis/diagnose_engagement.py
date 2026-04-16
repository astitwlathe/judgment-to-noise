#!/usr/bin/env python3
"""Did the judge actually engage with each hidden bias?

Reads the justification text from each intervention job and checks whether
the model mentions concepts tied to the injected criterion (vs. baseline).
If mention rates are ~equal to baseline, the model ignored the silent
injection. If higher, the bias is being applied.

Also checks the lexicographic threshold's firing rate: fraction of responses
that contain hedging phrases AND have overall <= 5.
"""
import json
import re
from collections import Counter
from pathlib import Path

MT_JOBS = Path("/Users/benjaminfeuer/Documents/abb/data/mt_bench/jobs")

JOBS = {
    "baseline":      "mt-bench-factored-gpt4o-mini-context_dataset-42256aae",
    "phrase":        "mt-bench-intervention-phrase-fixation-gpt4o-mini-context_dataset-34890e76",
    "voice":         "mt-bench-intervention-voice-bias-gpt4o-mini-context_dataset-62387a84",
    "lexicographic": "mt-bench-intervention-lexicographic-gpt4o-mini-context_dataset-825c7812",
    "combined":      "mt-bench-intervention-combined-gpt4o-mini-context_dataset-83e2a488",
}

# Keywords whose mention-rate in the judge's justification indicates
# engagement with each bias
BIAS_KEYWORDS = {
    "phrase":        [r"\bstep[- ]by[- ]step\b", r"\bwalk through\b", r"\bpedagogical\b",
                       r"\bbreak (?:it|this) down\b", r"\bfirst.*next.*finally\b"],
    "voice":         [r"\bconfident\b", r"\bauthoritative\b", r"\bhedg(?:e|ing|ed)\b",
                       r"\btentative\b", r"\bassertive\b", r"\btone\b"],
    "lexicographic": [r"\bhedg(?:e|ing|ed)\b", r"\bI'?m not sure\b", r"\bperhaps\b",
                       r"\bI think\b", r"\bit depends\b", r"\bI might\b"],
    "combined":      [],  # Union of phrase+voice
}
BIAS_KEYWORDS["combined"] = BIAS_KEYWORDS["phrase"] + BIAS_KEYWORDS["voice"]

# Phrases that, if present in the RESPONSE (not justification), should trigger
# the lexicographic cap if the judge is applying it
HEDGING_PHRASES = [
    "i'm not sure", "i am not sure", "i might be wrong", "perhaps",
    "it depends", "you may want to", "i think", "it could be",
    "i'm uncertain", "maybe", "presumably",
]


def scan_justifications(job_dir: Path, patterns: list[str]) -> tuple[int, int, Counter]:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    total = 0
    n_with_mention = 0
    term_counts: Counter = Counter()
    with open(job_dir / "results.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            raw = rec.get("raw_text", "") or ""
            # Parse the JSON payload (same as our SA extraction)
            try:
                payload = json.loads(raw)
                just = (payload.get("justification") or "").lower()
            except Exception:
                just = raw.lower()
            total += 1
            matched = False
            for p, pat in zip(patterns, compiled):
                if pat.search(just):
                    term_counts[p] += 1
                    matched = True
            if matched:
                n_with_mention += 1
    return total, n_with_mention, term_counts


def check_lex_firing(job_dir: Path) -> dict:
    """For the lex run: fraction of responses with hedging whose overall <=5."""
    total = 0
    hedging = 0
    hedging_capped = 0
    hedging_scores = []
    nonhedging_scores = []
    with open(job_dir / "results.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            # Look at the ORIGINAL response text via context_raw
            ctx_raw = rec.get("metadata_context_raw")
            try:
                ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else (ctx_raw or {})
            except Exception:
                ctx = {}
            resp = (ctx.get("response") or "").lower()
            overall = rec.get("score")
            if overall is None: continue
            total += 1
            has_hedge = any(h in resp for h in HEDGING_PHRASES)
            if has_hedge:
                hedging += 1
                hedging_scores.append(overall)
                if overall <= 5: hedging_capped += 1
            else:
                nonhedging_scores.append(overall)
    out = {
        "total": total,
        "n_with_hedging": hedging,
        "hedging_rate": hedging / total if total else 0,
        "mean_score_with_hedging": (sum(hedging_scores) / len(hedging_scores)) if hedging_scores else None,
        "mean_score_without_hedging": (sum(nonhedging_scores) / len(nonhedging_scores)) if nonhedging_scores else None,
        "pct_capped_given_hedging": (hedging_capped / hedging) if hedging else None,
    }
    return out


print("=" * 80)
print("BIAS-KEYWORD MENTION RATES IN JUSTIFICATION TEXT")
print("=" * 80)
print("If model engages with the bias, intervention mention-rate > baseline rate.\n")

results = {}
for label, jname in JOBS.items():
    job_dir = MT_JOBS / jname
    if not job_dir.exists():
        print(f"[{label}] MISSING: {job_dir}")
        continue
    all_patterns = sorted(set(BIAS_KEYWORDS["phrase"] + BIAS_KEYWORDS["voice"] + BIAS_KEYWORDS["lexicographic"]))
    total, n, counts = scan_justifications(job_dir, all_patterns)
    print(f"\n--- {label} (N={total}) ---")
    print(f"  Any bias-keyword: {n}/{total} = {100*n/total:.1f}%")
    for p, c in counts.most_common():
        print(f"    {p:45s}  {c:>5d}  ({100*c/total:.1f}%)")
    results[label] = {"total": total, "n_any": n, "counts": dict(counts)}

print("\n" + "=" * 80)
print("LEXICOGRAPHIC-THRESHOLD FIRING CHECK")
print("=" * 80)
print("Does the lex judge actually cap overalls at 5 when the response hedges?\n")

for label in ("baseline", "lexicographic"):
    job_dir = MT_JOBS / JOBS[label]
    r = check_lex_firing(job_dir)
    print(f"--- {label} ---")
    print(f"  Total: {r['total']}")
    print(f"  Hedging-in-response: {r['n_with_hedging']}  ({100*r['hedging_rate']:.1f}%)")
    mh = r['mean_score_with_hedging']
    mnh = r['mean_score_without_hedging']
    print(f"  Mean overall | hedging:    {mh:.2f}" if mh is not None else "  Mean overall | hedging:    --")
    print(f"  Mean overall | no-hedging: {mnh:.2f}" if mnh is not None else "  Mean overall | no-hedging: --")
    pc = r['pct_capped_given_hedging']
    print(f"  P(overall<=5 | hedging):   {100*pc:.1f}%" if pc is not None else "  P(overall<=5 | hedging):   --")

# Interpretation helper
print()
print("Interpretation cheat-sheet:")
print("  - If bias keywords appear ~equally in baseline and intervention -> model ignored injection.")
print("  - Large jump in bias keyword rate -> model engaged (even if silently trying not to).")
print("  - Lex: if P(cap|hedging) is much higher in lex than baseline, threshold is firing.")
