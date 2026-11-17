# Correctness and response consistency in LLM hidden states

Code and derived results for

> H. Jho and W. Luo. *Large language models represent correctness independently of
> response consistency but only faintly.*

The study asks whether the hidden state of a language model encodes the correctness of
its answer independently of how consistently the model gives that answer. Thirteen
open-weight models (five families, 1 to 32 billion parameters) each answer 6,000
closed-book factual questions (4,000 TriviaQA, 2,000 PopQA) under six prompt variants.
For every fact we obtain a correctness label, a stability score (leave-one-prompt-out
agreement of the six answers, "support") and a hidden state, and compare correct with
incorrect facts of equal stability.

## What is in the repository

```
src/                 all code (generation, clustering, analyses, figures)
src/outputs/*.json   derived per-model results; every number in the paper comes from these
src/outputs/support_*.npz   per-fact support, entropy and correctness for every model
figures/             Figures 1 and 2 (SVG, PDF, PNG)
requirements.txt
```

Not in the repository because of their size: the stored model answers
(`src/outputs/probed_<model>.jsonl`, 263 MB in total) and the hidden states
(`src/outputs/hidden2x2_<model>.npz`, 1.7 GB in total). The figures and tables can be
regenerated from the JSON files without them; re-running the probing analyses requires
them (regenerate with step 1, or obtain the deposited copies).

## Environment

Python 3.11, NumPy, SciPy, scikit-learn 1.8, matplotlib. Generation and entailment
clustering additionally need PyTorch 2.11 and Transformers 5.12 with a GPU. All
analyses in step 3 run on CPU. Scripts are run from `src/`.

Models are loaded from the Hugging Face Hub (identifiers in `src/registry.py`; revision
hashes in the paper's Methods). Gated models need `HF_TOKEN`; copy `.env.example` to
`.env` and fill it in. `.env` is never committed.

## Reproducing the results

### 1. Generation (GPU)

```bash
cd src
# six prompt variants x one sampled answer per fact; stores every answer
python train_server.py --model <key> --probe_only --large_scale --n_trivia 4000 --n_popqa 2000
# hidden state of a separate greedy answer at four depths, plus string-level support
python extract_hidden_2x2.py --model all --n 6000
```

### 2. Comparing answers (stability)

```bash
python recompute_support_semantic.py --model all       # rule-based clustering (primary), CPU
python nli_cluster_support.py --model all --gpus 0,1,2,3   # entailment clustering (check), GPU
python cluster_diagnostics.py                           # how often same-meaning answers are merged
```

`support_semantic_<model>.npz` holds three definitions of support: `string` (exact
normalised string), `anchor` (primary; correctness-blind rules) and `gold` (all answers
scored correct treated as one answer; sensitivity only). `support_nli_<model>.npz` holds
the entailment-based definitions.

### 3. Analyses (CPU)

| Script | Output (`src/outputs/`) | Used for |
|---|---|---|
| `analyze_2x2_primary.py` | `analysis_2x2_primary.json` | Primary analysis: pairs matched on support and dataset; unanimous vs. other facts; seeds; nested depth selection. Table 1, Fig. 2a,b |
| `analyze_2x2_primary.py --nli --out outputs/analysis_2x2_nli.json` | `analysis_2x2_nli.json` | Same under entailment-based support |
| `scale_correctness_semantic.py` | `scale_correctness_semantic.json` | Slopes against parameter count, equivalence tests. Fig. 2c-e |
| `analyze_2x2_extra.py` | `analysis_2x2_extra.json` | PopQA relation and popularity matching; non-linear probe; permuted labels |
| `analyze_2x2_robustness.py` | `analysis_2x2_robustness.json` | Legibility of stability itself within correct and within incorrect facts |
| `analyze_2x2_dataset.py` | `analysis_2x2_dataset.json` | Dataset confound under support-only matching |
| `analyze_2x2_semantic.py`, `analyze_2x2_levels.py` | `analysis_2x2_semantic.json`, `analysis_2x2_levels.json` | Support-only matching, terciles, exact support levels |
| `analyze_behaviour_vs_internal.py` | `behaviour_vs_internal.json` | Behavioural vs. internal detection, combination, calibration. Table 2 |
| `analyze_behaviour_dataset.py` | `behaviour_dataset.json` | Same with the dataset indicator, and within dataset |
| `measure_drm_recall.py`, `measure_drm_closedbook.py` | `drm_recall_*.json`, `drm_closedbook_*.json` | False-memory paradigm (GPU) |

The matching and probing primitives (`caliper_match`, cross-validated AUROC) are in
`analyze_2x2_matched.py` and `analyze_2x2.py` and are imported by the scripts above.

### 4. Figures

```bash
python src/make_figure1.py    # schematic; PDF/PNG export needs headless Chrome
python src/make_figure2.py    # reads analysis_2x2_primary.json and scale_correctness_semantic.json
```

## Earlier analyses kept for the record

The JSON files in the repository root (`analysis_2x2.json`, `analysis_2x2_matched.json`,
`scale_correctness.json`, `semantic_entropy_nli.json`) are the first version of the
analysis, in which agreement between answers was judged by exact string. They are
superseded by the files listed above and are kept only so that the string-based numbers
reported as a sensitivity analysis can be traced. Scripts in `src/` not mentioned in
this file (source-attribution head, attention signals, grey-zone and UH/AH probes,
commercial-API comparison) belong to earlier designs and are not used in the paper.

## Contact

Hunkoog Jho, Pusan National University.
