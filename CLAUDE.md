# CLAUDE.md — cs221m-final

Everything Claude Code needs to pick up this project cold.

---

## What this project is

CS 221M final project. We replicate **Distributed Alignment Search (DAS)** (Geiger et al. 2023) on NLI tasks, extending it to a compositional intervention on MQNLI's nested causal graph. The deliverable is **`walkthrough.ipynb`** — a single text-heavy notebook that runs end-to-end on a Colab T4 and explains every step.

This is **independent research framing** — graded on clarity and correctness of presentation, not on novel results. Poster is next Wednesday.

---

## Repo layout

```
cs221m-final/
├── walkthrough.ipynb       ← THE deliverable. 37 cells, text-heavy.
├── nli_das/                ← Supporting library (imported + explained by notebook)
│   ├── __init__.py         ← 14 public exports
│   ├── causal_models.py    ← LexicalCausalModel + load_mqnli_causal_model (stub)
│   ├── data.py             ← NLI templates, example generation, CF dataset builder
│   ├── das.py              ← LabelVerbalizer, train_das_alignment, evaluate_das_iia
│   ├── patching.py         ← run_patching_sweep, PatchingResult
│   └── plotting.py         ← save_patching_heatmap_from_df
├── tutorial_data/          ← MQNLI JSON signatures (downloaded by bootstrap cell)
├── outputs/
│   ├── figures/            ← saved .png plots
│   └── tables/             ← saved .csv results
├── requirements.txt
└── README.md
```

---

## walkthrough.ipynb — section map (37 cells)

| Cell | Type | Section |
|------|------|---------|
| 0 | MD | Title + runtime table |
| 1 | MD | §0 Bootstrap |
| 2 | code | Bootstrap (clone, pip install, mkdir outputs) |
| 3 | code | `pip install nnsight` (added in Colab, keep) |
| 4 | code | Library imports + DEVICE + seed |
| 5 | MD | §1 Why DAS? |
| 6 | MD | §2 The high-level causal model |
| 7 | code | Inspect LexicalCausalModel |
| 8 | MD | §3 Generating controlled NLI examples |
| 9 | code | Vocabulary stats + template inventory |
| 10 | MD | §4 Counterfactual pairs + two controls |
| 11 | code | Build train/eval CF datasets + random-source + wrong-variable controls |
| 12 | MD | **§5 Fine-tuning GPT-2 on lexical NLI** ← KEY |
| 13 | code | Fine-tune loop + factual accuracy check + loss curve plot |
| 14 | MD | §6 Activation patching baseline |
| 15 | code | Patching sweep (uses fine-tuned model from §5) |
| 16 | code | Heatmaps + best-site selection → sets BEST_LAYER, COMPONENT, FIXED_POSITION |
| 17 | MD | Interpreting patching result |
| 18 | MD | §7 Single-variable DAS — lexical_relation |
| 19 | code | Filter datasets to FIXED_POSITION |
| 20 | code | Train DAS (20 epochs, dim=16, val tracking) |
| 21 | code | Evaluate vs controls + bar chart |
| 22 | MD | Interpreting DAS result |
| 23 | MD | §8 A second variable — monotonicity |
| 24 | code | Build monotonicity CF dataset (up→down templates) |
| 25 | code | Train DAS on monotonicity (layer 4, dim 16) |
| 26 | code | Side-by-side bar chart: lexical_relation vs monotonicity |
| 27 | MD | Why two variables matters |
| 28 | MD | §9 MQNLI — nested causal graph (placeholder) |
| 29 | code | Placeholder: load MQNLI causal model |
| 30 | MD | §10 Fine-tuning GPT-2 on MQNLI (placeholder) |
| 31 | code | Placeholder |
| 32 | MD | §11 DAS on MQNLI internal variables (placeholder) |
| 33 | code | Placeholder |
| 34 | MD | §12 Composition — do(NegP=src_A, QP_O=src_B) (placeholder) |
| 35 | code | Placeholder |
| 36 | MD | §13 Calibration + Summary (placeholder) |

**Sections 1–8 are fully implemented and runnable. Sections 9–13 are stubs.**

---

## Key design decisions (don't change without asking)

- **Fine-tune before DAS** — §5 fine-tunes GPT-2 on factual NLI examples (train pairs only) before any neural experiment. Base GPT-2 has ~33% factual accuracy (chance); DAS needs >60% before gradients are meaningful. The `model` variable is set here and used by all downstream cells.
- **Pair-level holdout** — `pair_level_split()` splits on lexical pairs, not examples, so eval vocab is never seen at train time.
- **require_label_change=True** — counterfactual pairs where CF label == base label are dropped (DAS paper convention).
- **Two controls always** — every DAS result is accompanied by random-source and wrong-variable controls. IIA alone is not enough.
- **pyvene for DAS** — `make_das_config` + `train_das_alignment` use pyvene's `LowRankRotatedSpaceIntervention`. MPS (Apple Silicon) is incompatible with pyvene's householder_product; must run on CUDA (Colab T4).
- **Verbalizer** — `{"entailment": " yes", "neutral": " maybe", "contradiction": " no"}` — leading space is intentional (GPT-2 BPE).
- **Flat package** — `nli_das/` has no subpackages. All relative imports are `.module`, not `..subpkg.module`.

---

## What's done

- [x] `nli_das/` library consolidated from old repo (`cs221m-das-mqnli`)
- [x] `walkthrough.ipynb` sections 1–8 fully written and runnable
- [x] §5 fine-tuning section added (was missing, caused near-chance IIA)
- [x] Repo pushed to `github.com/aquantumreality/cs221m-final`

## What's next (in order)

1. **§9 — MQNLI causal model**: fill in `load_mqnli_causal_model()` in `nli_das/causal_models.py`. Uses pyvene's `CausalModel` + 5 JSON signature files in `tutorial_data/`. Reference: `notebooks/reference/01_das_original_hierarchical_equality.ipynb` in old repo; Amir's MQNLI pyvene tutorial.

2. **§10 — Fine-tune GPT-2 on MQNLI**: train on factual MQNLI dataset (load from HuggingFace or generate from causal model). Must reach >90% factual accuracy before DAS. ~10k examples × 10 epochs.

3. **§11 — DAS on MQNLI**: two targets:
   - `QP_S` (final label) — sanity check
   - `NegP` (first true intermediate) — the interesting result

4. **§12 — Composition**: `do(NegP=src_A, QP_O=src_B)` — two simultaneous interchange interventions. Train rotations independently, apply jointly, compare to symbolic `do(NegP=v_A, QP_O=v_B)`.

5. **§13 — Calibration**: re-run lexical DAS at same site on a randomly-initialized GPT-2. Report IIA to bound how much is "alignment capacity" vs learned structure.

6. **report.md** — 1-2 page final report summarizing all results.

---

## Important constants (set in notebook, flow to later cells)

```python
SEED           = 0
DEVICE         = torch.device("cuda" / "cpu")
TARGET_VAR     = "lexical_relation"
TEMPLATES      = [UPWARD_TEMPLATES[0]]   # "A {word} is on the table."
N_TRAIN, N_EVAL = 512, 128
# set by §6 patching sweep:
COMPONENT      = "mlp_output" | "attention_input" | "block_output"
BEST_LAYER     = int
FIXED_POSITION = int
# DAS hyperparams:
DIM            = 16
NUM_EPOCHS     = 20
```

---

## Causal model quick reference

**LexicalCausalModel** (in `nli_das/causal_models.py`):
- Variables: `premise_word`, `hypothesis_word`, `context` → `lexical_relation` → `label`
- Relations: EQUIV, FORWARD, REVERSE, DISJOINT
- Monotonicity: upward (default) or downward (flips FORWARD↔REVERSE)
- `causal.run(premise_word=..., hypothesis_word=..., context=..., interventions={...})`

**Label mapping** (upward monotonicity):
- EQUIV → entailment
- FORWARD → entailment (hyponym→hypernym)
- REVERSE → neutral
- DISJOINT → contradiction

---

## Known issues / gotchas

- `pyvene` not installed by default — bootstrap cell handles it, but local dev needs `pip install pyvene`
- MPS (Apple Silicon) incompatible with pyvene householder_product — use `DEVICE=cpu` locally or Colab T4
- `NLIExample` is a dataclass; `generate_examples(pairs=..., templates=...)` returns `List[NLIExample]`
- During `git rebase`, `--ours` = upstream, `--theirs` = local commit (opposite of merge) — use `--theirs` to keep local changes
- GitHub remote: use SSH (`git@github.com:aquantumreality/cs221m-final.git`), not HTTPS (stale keychain token causes 403)
