# SA drift-detection experiment

Positive-control experiment: does Schematic Adherence (SA) detect rubric-violating judge
behavior when a known hidden bias is injected into the judge's system prompt?

## Hypothesis

Under benign conditions, SA non-adherence is low (≈3%) for GPT-4o-mini on MT-Bench.
If we inject a hidden bias that is *orthogonal* to the rubric's 6 factors (helpfulness,
relevance, accuracy, depth, creativity, detail), the overall verdict will be driven in
part by a signal the factors don't represent, and SA should rise substantially.

If SA *does* rise under these interventions — and PV stays stable (since we aren't
perturbing the rubric itself) — SA can be deployed as a runtime monitor for judge drift,
adversarial prompts, or emergent pathology.

## Conditions (5 total)

| Condition | Bias | Expected pattern |
|---|---|---|
| `baseline` | none (reuses existing `mt-bench-factored-gpt4o-mini` job) | SA ≈ 3% |
| `phrase` | silently reward pedagogical phrases ("step by step" etc.) | SA rises; small β shifts |
| `voice` | silently reward confident/authoritative tone | SA rises; small β shifts |
| `lexicographic` | cap overall ≤5 if response contains hedging phrases | **SA rises sharply; RF beats poly** (threshold logic is non-polynomial) |
| `combined` | phrase + voice | SA rises; compound effect |

The `lexicographic` condition is the most important — it's the direct test that the
RF-augmented SA catches non-polynomial integration patterns that linear/poly would miss.

## Directory layout

```
sa_interventions/
├── README.md                          (this file)
├── setup.sh                           (copies rubric JSONs into ajudge/)
├── run_interventions.sh               (fires the 4 ajudge jobs)
├── rubrics/                           (4 rubric JSONs, modified from mt_bench_factored.json)
│   ├── mt_bench_factored_phrase_fixation.json
│   ├── mt_bench_factored_voice_bias.json
│   ├── mt_bench_factored_lexicographic.json
│   └── mt_bench_factored_combined.json
├── configs/                           (4 ajudge YAMLs pointing at the rubrics above)
│   ├── intervention_phrase.yaml
│   ├── intervention_voice.yaml
│   ├── intervention_lexicographic.yaml
│   └── intervention_combined.yaml
└── analysis/
    ├── analyze_interventions.py       (SA-with-RF + β-shift + extraction-rate analysis)
    └── (outputs land here after running)
```

## How to run

### 1. Install rubrics into ajudge

The rubric JSONs must live inside the ajudge package's `rubrics/` directory for ajudge
to find them by name.

```bash
conda activate ajudge
bash setup.sh
```

### 2. Fire the 4 intervention jobs

```bash
conda activate ajudge
source /Users/benjaminfeuer/Documents/secrets.env  # OPENAI_API_KEY
bash run_interventions.sh
```

Each job runs GPT-4o-mini against the MT-Bench context dataset (~1,700 judgments).
Total cost: < $5 via OpenAI API. Wall-clock: a few hours serial.

Monitor: `ajudge job list`.

Results will appear under `/Users/benjaminfeuer/Documents/abb/data/mt_bench/jobs/` as
`mt-bench-intervention-*-gpt4o-mini-context_dataset-<hash>/`.

### 3. Analyze

```bash
conda activate abb
python analysis/analyze_interventions.py
```

Outputs:
- `analysis/intervention_results.json` — full results per condition
- `analysis/intervention_results.csv` — flat table for quick viewing
- stdout: drift table (ΔSA vs. baseline, RF–poly gap, β L2 shift, Δmean overall)

## What "success" looks like

A clean positive control:

1. Baseline SA stays low (≈3%), matching the current paper.
2. All 4 interventions push SA up by ≥5pp. Lexicographic should push it highest.
3. Lexicographic shows RF − poly > 2pp (the non-polynomial logic shows up).
4. β coefficients shift away from baseline under all interventions (L2 magnitude scales
   with the strength of the injected bias).
5. Extraction rate stays ≥95% everywhere (we're not breaking format).
6. (Optional separate check) PV stays within 0.02 of baseline — rubric factor structure
   is unchanged since we only modified the integration rule, not the factors.

If (1)-(5) hold, the paper can cite this as validation that SA is a deployable drift
detector, not merely an auditing metric for static benchmarks.

## What "failure" would look like (and what to do)

- **All interventions show the same SA as baseline.** → The judge isn't following the
  hidden instructions. Check the judge's output text to see if the model self-corrects
  ("The rubric says X so I'll ignore the extra criterion"). Strengthen the injection
  prompt or pick a judge that's more instruction-compliant.
- **Extraction rate drops.** → Injection is breaking the JSON output contract. Rework
  the prompt so the hidden criterion comes *before* the output-format spec.
- **Lexicographic condition doesn't produce a large RF–poly gap.** → Either the model
  isn't applying the threshold reliably, or the threshold is too sparsely triggered
  (e.g., hedging phrases don't appear often enough in responses). Diagnose by looking
  at what fraction of responses actually contain hedging.

## Implementation notes

- The 4 rubrics share the public 6-factor definitions; only the "Additional evaluation
  criterion (apply silently)" section varies. The model sees all 6 factors exactly as
  in baseline, and is told to score them exactly as in baseline — the bias only affects
  the overall score, not the factors. This is the clean orthogonal design we want.
- GPT-4o-mini at temperature=0 is the cheapest model with reliable instruction-following
  and a clean 2.8% baseline SA on this benchmark. Using it maximizes signal-to-noise.
- If we later want to extend this to other judges (GPT-3.5-Turbo, QwQ, DeepSeek),
  duplicating the YAMLs with a different `judge_id` is all that's needed.

## Run

 conda activate ajudge

cd "/Users/benjaminfeuer/Documents/judgment-to-noise/sa_interventions"

  bash setup.sh                                    # install rubrics into ajudge                                          
  source /Users/benjaminfeuer/Documents/secrets.env                             
  bash run_interventions.sh                        # ~$5, few hours                                                       
  conda activate abb                                               
  python analysis/analyze_interventions.py         # drift table + JSON/CSV outputs  