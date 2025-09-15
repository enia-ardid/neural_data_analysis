# Neural Population Decoding in an 8‑Arm Task (PFC/HPC): TCA/SliceTCA, Random Forest, and QNN

This repository contains a **reproducible pipeline** to analyze rodent neural population activity from **prefrontal cortex (PFC)** and **hippocampus (HPC)** during an **8‑arm decision task**, with three complementary aims:

- **Aim 1 — Structure discovery:** Tensor Component Analysis (**TCA**) and **SliceTCA** to reveal neuronal ensembles (A), temporal/positional motifs (B), and event weights (C).
- **Aim 2 — Supervised decoding:** Robust **Random Forest (RF)** decoders that predict the arm_id **during-event** and **pre‑event (decision)**, with strong anti‑leakage safeguards and uncertainty estimates.
- **Aim 3 — Quantum baseline:** A capacity‑matched **Quantum Neural Network (QNN)** using **5 qubits** (angle encoding) compared against classical baselines under identical preprocessing.

> **Why this repo?** It emphasizes **fair, capacity‑matched comparisons**, **time‑aware evaluation**, and **interpretable controls** (sanity checks, confidence intervals, permutation tests).

---

## 🔁 End‑to‑end pipeline (notebook order)

> **Run in this exact order**; each notebook saves artifacts used by the next steps.

1. **`import_preprocessing.ipynb`** — Load raw data; basic cleaning; **speed filter (optional)**; metadata checks.
2. **`construct_tensors.ipynb`** — Build event‑level matrices and tensors in **(neurons × time × events)** convention; **duration outlier filtering (MAD)**; time normalization targets.
3. **`position_modulation.ipynb`** — Compute linearized position / slices; explore rate vs. position; prepare **slice indices** for SliceTCA.
4. **`RF_decoder.ipynb`** — **During‑event** decoding with **Random Forest**. Chronological holdouts with **GAP**; **blocked CV**; **TRAIN‑only** preprocessing (drop low‑spike neurons, set T, build X, scale, optional PCA). Reports **Accuracy (Wilson CI)**, **Macro‑F1 (bootstrap CI)**, **perm‑test p‑value**, confusions, and importances.
5. **`pre_event_decoding.ipynb`** — **Pre‑event** decoding (decision). Align fixed windows **before event onset**; guarantee **no overlap** with during‑event windows to avoid leakage; same evaluation protocol as (4).
6. **`TCA_decomp.ipynb`** — CP/TCA on (neurons × time × events); visualize **A/B/C factors**; use **C** (or PCA of C) for 3‑D clustering and simple classifiers.
7. **`slicetca_decomp.ipynb`** — SliceTCA with **time/position slices** and **smoothness** on B; cleaner motifs and often clearer class separation in C. Export C for downstream decoding/visualization.
8. **`qnn.ipynb`** — **Aim 3**: Reduce each event to **5 PCs**; 1‑to‑1 **angle encoding** onto 5 qubits; **3‑layer ring‑entangled** ansatz; 5 ⟨Z⟩ → linear **5→3** head. Train with cross‑entropy and **parameter‑shift**. Compare to logistic regression, linear/RBF‑SVM, and RF on the same 5‑D inputs using **GroupKFold(trial_id)**.

---

## 🧪 Data & expected layout

```
data/
├── raw/               # raw recordings and annotations (not versioned)
├── interim/           # intermediate artifacts (npz/csv) saved by notebooks
└── processed/         # tensors, factors, trained models, and figures
```

- The code assumes **event‑level matrices** (neurons × time) can be constructed per trial.
- Use consistent sampling/binning across sessions; see `construct_tensors.ipynb` for time normalization targets:  
  **`median_pad`** (truncate long, pad short with zeros) or **`median_minus_std`** (truncate to median−1 SD).

---

## 🔒 Reproducibility & leakage guardrails

- **Chronological splits + GAP** between TRAIN+VAL and TEST to break temporal autocorrelation.
- **Blocked CV** in TRAIN+VAL; **TRAIN‑only** fitting for neuron filtering, time‑length **T**, scaler, and optional PCA.
- **Sanity checks**: label **shuffles** and **temporal rotations** (chance‑level); **permutation tests** on TEST (p‑values).
- **GroupKFold(trial_id)** for the QNN & classical baselines (Aim 3) so that **events from the same trial** never appear in both train and test.

---

## 🧠 What is “during‑event” vs “pre‑event (decision)” decoding?

- **During‑event**: features come from the **event window itself** (aligned to entry/behavior). Tests if population activity **represents** the chosen arm while it unfolds.
- **Pre‑event (decision)**: features come from a **fixed window before event onset** (non‑overlapping). Tests if population activity **predicts** the upcoming arm **before** observable behavior.  
  *Both use the same RF protocol; windows are disjoint to avoid leakage.*

---

## 🚀 Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name eightarm-venv
# Open Jupyter and run notebooks 1→8 in order
```

- Python **3.10+** recommended. If available, set `PL_LIGHTNING_GPU=1` to enable PennyLane Lightning GPU backend (optional).

---

## 📦 Requirements (pip)

See `requirements.txt`. Key deps:
- **Core**: numpy, scipy, pandas, matplotlib, scikit‑learn, tqdm
- **Tensor methods**: tensorly (TCA); (optional) slicetca if available
- **Quantum**: pennylane, pennylane‑lightning (optional), torch
- **Viz/nb**: jupyter, ipykernel, plotly (optional)

---

## 📊 Outputs & figures

- `processed/factors/` — TCA/SliceTCA factors (A,B,C).  
- `processed/models/` — RF/QNN trained objects and CV summaries.  
- `processed/metrics/` — Accuracy, Macro‑F1 (with CIs), permutation tests, confusion matrices.  
- `processed/figures/` — Event‑space 3‑D plots (C), importances, heatmaps.

---

## 📝 How to cite / license

- **License:** MIT (feel free to adapt; add `LICENSE` file).  
- If you use this code in academic work, please cite the repository and relevant TCA/SliceTCA/QML references.

---

## ❓FAQ (short)

- **Why Random Forest?** Strong baseline for noisy, nonlinear neural data; minimal tuning; feature importances map back to neurons (no PCA case).
- **Why 5 PCs for the QNN?** Matches **5 qubits** for **angle encoding** one‑to‑one; keeps capacity matched to classical baselines.
- **Why GroupKFold by trial?** Prevents events from the same trial leaking across splits; fair evaluation.
- **Why SliceTCA?** Slice alignment + smoothness yields cleaner motifs and often clearer class separation than plain TCA.
