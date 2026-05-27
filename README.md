# CS 221M Final Project — Compositional Distributed Alignment Search for Nested NLI Causal Graphs

A walkthrough notebook replicating and extending **Distributed Alignment Search (DAS)** (Geiger et al. 2023) for nested NLI causal graphs. We show that:

1. DAS recovers an interpretable single high-level variable (`lexical_relation`) in GPT-2 on a controlled NLI task.
2. The same method extends to a second variable (`monotonicity`) at a different intervention site.
3. On MQNLI's nested causal graph, two independently-trained DAS subspaces **compose** — joint interchange interventions produce the symbolic counterfactual.
4. A random-init GPT-2 control gives nontrivial IIA, illustrating the calibration concern raised by Makelov et al. and Sutter et al.

## What's here

- **`walkthrough.ipynb`** — the deliverable. Text-heavy walkthrough, ~30 cells. Runs end-to-end on a Colab T4 in ~30 min for the lexical NLI sections, ~60 min including MQNLI.
- **`nli_das/`** — supporting library (5 modules, all imported and explained by the notebook).
- **`tutorial_data/`** — MQNLI signature JSON files (downloaded once by the notebook bootstrap).
- **`outputs/`** — generated tables and figures.
- **`report.md`** — 1-2 page final report.

## How to run

1. Open the notebook in Colab (T4 GPU runtime).
2. Run the bootstrap cell — clones this repo, installs `pyvene` + deps, downloads MQNLI signature files.
3. Run all cells. Outputs are saved to `outputs/`.

## Paper being engaged with

- **Primary**: Geiger, Wu, Potts, Icard, Goodman (2023). *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*.
- **Extension**: Wu, Geiger, Icard, Potts, Goodman (2023). *Interpretability at Scale: Identifying Causal Mechanisms in Alpaca* (Boundless DAS).
- **Critique cited**: Makelov, Lange, Nanda (2023). *Is This the Subspace You Are Looking for?* and Sutter, Minder, Hofmann, Pimentel (2025). *Non-Linear Representation Dilemma*.

This is **independent research framing** per CS221M final-project guidelines: graded on clarity of presentation, not on producing novel results.
