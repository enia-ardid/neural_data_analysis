# Neural Data Analysis — 8-Arm Task

This repository contains seven Jupyter notebooks that take you from raw rodent recordings to interpretable analyses tailored to an **8-arm radial maze** task. The core theme is to **relate population activity to behavior**—both **during** arm visits and **before** they happen—while also uncovering **low-dimensional structure** in the population.

**Notebooks (in run order):**

1. `import_preprocess.ipynb`
2. `construct_tensors.ipynb`
3. `position_modulation.ipynb`
4. `RF_decoder.ipynb`
5. `pre_event_decoding.ipynb`
6. `TCA_decomp.ipynb`
7. `slicetca_decomp.ipynb`

> There is no QNN notebook in this repo. The documentation below covers exactly these seven notebooks.

---

## Why this pipeline for an 8-arm task?

In an 8-arm maze, the animal’s behavior naturally decomposes into **events**—departing the center, entering an arm, possibly obtaining a reward, and exiting. Neural activity is expected to reflect:

* **Where** the animal is (arm identity and position along the arm),
* **What** it is about to do (upcoming arm choice),
* **How** population patterns organize across **time** (temporal motifs) and **trials** (event structure).

This repo therefore builds **event-aligned matrices/tensors**, performs **behavioral decoding** (during-event and pre-event), and uses **tensor decompositions** to identify **ensembles**, **temporal components**, and **event-space structure**.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name neural-analysis
```

Python 3.10+ is typically suitable.

---

## Notebook overview — what and why

### 1) `import_preprocess.ipynb` — Load & clean

**What:** Load spikes/clusters and behavioral trajectories (e.g., `.whl/.lwhl`) and perform basic sanity checks (time coverage, speed distributions, session/day metadata).
**Why:** Downstream analyses require **time-aligned**, **artifact-aware** inputs. Even small mistakes (dropped samples, clock drift, or speed thresholds) can leak into decoding and factorization, giving inflated or misleading results.

**Typical considerations explained in code/comments:**

* **Speed filters** to exclude immobility or grooming that do not reflect navigational coding.
* **Alignment checks** to ensure spikes and position timestamps match one another.
* **Cell-type masks** (if available) to restrict to neurons of interest for a given analysis.

---

### 2) `construct_tensors.ipynb` — Event-level matrices/tensors

**What:** Segment the continuous session into **events** (arm visits/trials) and extract matrices in the convention **(neurons × time bins)** per event. Optionally normalize durations (e.g., truncation/padding) so events are comparable.
**Why:** Event alignment lets us ask **within-event** questions (temporal motifs as the animal traverses an arm) and **across-event** questions (consistency of coding across arms/trials). It also prepares the data for **decoding** and **tensor decompositions** that assume a consistent (neurons × time × events) shape.

**Design choices discussed:**

* **Binning:** fixed-width time bins vs. time-warped bins to equalize event duration. Fixed bins preserve real time; warping eases cross-event comparability.
* **Inclusion criteria:** which events count (rewarded arms only vs. all arms), and how to treat partial/noisy events.

---

### 3) `position_modulation.ipynb` — Linearized position & basic modulation

**What:** Compute **linearized position** along each arm from XY trajectories; optionally divide into **slices/segments**. Relate firing (rates or binned spikes) to position along the arm.
**Why:** In a radial maze, coding often respects the **geometry** of arms. Linearization collapses 2D motion onto a 1D metric along each arm, enabling clear **position-tuning** summaries and comparisons across events and arms.

**Analytical notes:**

* Linearization depends on center coordinates and arm orientation/length assumptions.
* Position slicing (e.g., equal-length segments) supports models that need discrete states or spatial bins for interpretability.

---

### 4) `RF_decoder.ipynb` — During-event decoding (supervised)

**What:** Train a **Random Forest** classifier on features derived from the **event window itself** (e.g., binned spikes/rates) to decode **arm identity** or related labels.
**Why:** During-event decoding answers: *Given the neural activity while the animal is in an arm, how well can we recover where it is or what it’s doing?* RF is a robust baseline, handles nonlinearities, and yields interpretable **feature importance**.

**Methodological cautions surfaced in the notebook:**

* **Leakage avoidance:** keep train/test events temporally separated (chronological or blocked splits).
* **Label balance & chance controls:** simple shuffles provide sanity checks for overfitting.
* **Reporting:** accuracy with confusion matrices gives **which-arm** error structure (e.g., confusions among neighboring arms).

---

### 5) `pre_event_decoding.ipynb` — Pre-event (decision) decoding

**What:** Use the **pre-event window** (a fixed interval before event onset) with the **same RF protocol** to test **predictive** information, i.e., can we decode the **upcoming arm** *before* the animal enters it?
**Why:** This probes **planning/decision-related** signals. Comparing pre-event vs. during-event performance separates **prospective** from **current-state** coding.

**Key points emphasized:**

* **No overlap** between pre-event and during-event windows.
* Same leakage-aware splits as in the during-event notebook for fair comparison.
* Interpretation: significant pre-event accuracy suggests anticipatory coding beyond mere position.

---

### 6) `TCA_decomp.ipynb` — Tensor Component Analysis (CP)

**What:** Apply CP/TCA to the **(neurons × time × events)** tensor to extract factors: **A** (neurons/ensembles), **B** (temporal motifs), **C** (event weights/structure).
**Why:** TCA seeks **low-rank structure** jointly across neurons, time, and events. This can reveal:

* **Ensembles** that co-activate,
* **Motifs** aligned to event progression (e.g., entry/approach/exit),
* **Event clustering** (e.g., different patterns for rewarded vs. non-rewarded arms).

**Modeling notes:**

* **Rank selection** (number of components) trades reconstruction vs. interpretability.
* **Stability checks** (initialization seeds / split-half comparisons) help validate motifs.

---

### 7) `slicetca_decomp.ipynb` — SliceTCA (position/time-aware)

**What:** Decompose data with **slice-aware** constraints (e.g., position or time slices) to obtain smoother, often more interpretable factors than plain CP.
**Why:** In navigational tasks, signals change smoothly with position/time. SliceTCA leverages this prior to **denoise** factors and sharpen **event-space** separations, making motifs easier to read and relate to behavior.

**Interpretation hints:**

* Compare SliceTCA factors to CP/TCA: clearer motifs or tighter event clustering indicate the value of the slice prior.
* Link event-weights (C) to behavioral variables (arm identity, reward) to ground factors in task structure.

---

## Data & analysis assumptions (conceptual)

* **Event definition:** A “visit” typically spans center-out → in-arm → (optional) reward → arm-out. Exact onset/offset are defined in preprocessing/segmentation code.
* **Units:** Analyses assume consistent time units (e.g., seconds) across position and spike timestamps.
* **Binning:** Fixed-width bins preserve physical time; time-warping fosters across-event comparability. Choice depends on the research question.
* **Cell selection:** If cell types/quality metrics exist, you may restrict analyses to specific neuron classes for clearer effects.

---

## Good practice & caveats

* **Temporal leakage:** Always split by time/blocks; avoid mixing near-by events across train/test.
* **Controls:** Add label shuffles or temporal rotations as a baseline.
* **Parameter logging:** Record bin width, pre-event window length, velocity/quality thresholds, and event definitions. They strongly impact decoding and factorization.
* **Interpretability:** For TCA/SliceTCA, cross-validate ranks and check factor stability; don’t over-interpret a single run.

---

## License

MIT — see `LICENSE`.

---
