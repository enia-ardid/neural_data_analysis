# Technical guide

## Saved figures

The notebooks retain selected static PNG figures from the original analysis. The README trajectory comes from `build_event_position_tensor` in `construct_tensors.ipynb`; the CP figure is the first sorted component of `tensorNEP_rewarded100hp` in `TCA_decomp.ipynb`. Both images are unchanged copies of saved notebook outputs.

The recordings are unavailable here, so the figures have not been regenerated with the current code. Saved decoding estimates predate corrections to event alignment and window extraction. They remain exploratory historical results; quantitative conclusions require a rerun. Figures with identified plotting errors are omitted.

## Recording inputs

The pipeline starts with annotated, spike-sorted recordings. It does not acquire data, sort spikes, detect maze visits from video or compute the linearized position coordinate. The original recordings cannot be redistributed; no public download or external preprocessing implementation is included.

Configure `data_folder` (or the `NEURAL_DATA_DIR` environment variable), `rat`, `day` and `sessions` in `import_preprocess.ipynb`. For a single session, the loader expects:

```text
data_folder/
└── {rat}-{day}/
    ├── {rat}-{day}.des
    ├── {rat}-{day}_{session}.res
    ├── {rat}-{day}_{session}.clu
    ├── {rat}-{day}_{session}.whl
    ├── {rat}-{day}_{session}.reward_arms
    ├── {rat}-{day}_{session}.all_arms
    ├── {rat}_{day}_{session}.speed
    ├── {rat}_{day}_{session}.trials
    ├── {rat}_{day}_{session}.lwhl_raw
    ├── {rat}_{day}_{session}.lwhl
    └── {rat}_{day}_{session}.lwhl_speed
```

The filename patterns and formats below describe what the loader expects, including the distinction between hyphens and underscores.

- `.res`: one integer spike timestamp per line, in samples at **20,000 Hz**. `.clu`: a first-line cluster-count header, followed by one cluster ID per spike.
- `.des`: one cell-type label per row, starting at cluster ID 2; the loader prepends `noise` and `multiunit`. The notebooks use `pp` for the population called PFC and `p1` for the population called hippocampus. Those anatomical assignments must be checked against the recording metadata.
- `.whl`: a whitespace-delimited numeric trajectory without a header. `.speed`: one numeric speed per tracking sample, without a header. NaNs in speed become zero and signed speeds become absolute values.
- `.reward_arms`: integer arm IDs for the session. `.all_arms`: one row of visited arm IDs per trial, in event order. A “rewarded” visit means a visit to a listed arm, not a verified reward-delivery outcome.
- `.trials`: one row per trial, containing a start, repeated `(entry, reward1, reward2, exit)` timestamps, then a trial end. These are tracking sample indices at **39.0625 Hz**. Each entry–exit pair becomes an event in milliseconds. Incomplete groups are skipped and the total event/arm count is asserted.
- `.lwhl_raw`, `.lwhl` and `.lwhl_speed`: whitespace-delimited tables with headers. Downstream code uses `.lwhl` columns `x1`, `y1`, `linearized_pos_long` and `trial_id`. The other two tables are loaded but are not used in the downstream computations.

Tracking rows are assigned a zero-based sample index. Time zero must match the spike clock; the code does not estimate clock drift or session offsets. Position is treated as centimeters along a 140 cm coordinate. Speed threshold values assume the units of the supplied speed file; their physical calibration is not verified here.

**Use one session per run.** Although the loader accepts a list, it concatenates spike and trial timestamps without offsetting them. It does not establish a common clock or handle changing rewarded-arm sets across sessions. Process such recordings separately unless their alignment is established outside this pipeline.

`raw_data.pkl` stores spikes, labels, tracking, speed, event intervals and trial IDs. Trial IDs are assigned from the modal valid `trial_id` within each event (minimum fraction 0.5); ambiguity is coded as `-1`. The decoder does not currently group its splits by those trial IDs.

## Tensor construction

`construct_tensors.ipynb` selects cell types, bins spike counts at **100 or 500 ms**, and applies a Gaussian filter with **sigma = 3 bins** along the full session. Sigma is in bins, so its physical width differs between the two bin sizes. These matrices contain smoothed counts per bin, not rates in Hz. An additional 200-spike binning example is diagnostic and is not exported as an analysis input.

The notebook demonstrates a compressed speed-filtered matrix, but constructs event tensors from matrices that retain the original session timeline. Removing session-wide columns before using timestamps would invalidate the alignment.

The two tensor conventions serve different purposes:

| Representation | Shape and construction | Consequence |
| --- | --- | --- |
| `tensorTE_*` | `(neurons, retained time bins, events)`. Slice visits with floor/ceil bin boundaries, retain bins with mean speed ≥ 5, then truncate or zero-pad to the median retained event length. Keep neurons with at least 100 counts summed across the speed-filtered event slices. | No time warping is applied, but removing slow bins compresses time. The second axis is not continuous elapsed time from entry. |
| `tensorNEP_*` | `(neurons, events, 140 positions)`. Use neural-bin centers and the tracking sample at or immediately after each center; round/clip position to 0–139, apply speed ≥ 5 at that sample, and average counts for each event/position. Neurons first pass a session-wide 100-count filter. | Unoccupied positions are zero. The representation does not distinguish no observation from observed silence. |

`all` and `rewarded` suffixes indicate the arm selection; `100`/`500` indicate bin width in milliseconds, and `pp`/`hp` identify the two selected cell populations. Cell selection can differ between representations, so a row index is not a stable neuron identifier across all tensors. Original cluster-ID mappings and position occupancy counts are not saved in `tensors.pkl`.

`tensors.pkl` includes both tensor families, the 100 ms session matrices, event/arm labels, tracking, speed and rewarded-arm IDs. `qnn_data.pkl` includes the 100 ms time tensors and trial/arm labels for an external project. Intermediate files are local artifacts, excluded from Git, and are overwritten on reruns.

## Position modulation

`position_modulation.ipynb` interpolates the supplied position and optional speed to neural-bin centers. It converts 100 ms counts to Hz and compares **14 position slices** across the 140 cm coordinate. The examples use a minimum of 100 counts per neuron, at least 10 bins per slice, 1,000 circular-shift permutations of an ANOVA F-statistic, and Benjamini–Hochberg correction at 0.01. The speed-filtered example uses **speed > 5**, whereas tensor construction uses **≥ 5**.

Outputs include occupancy, tuning means and variability, raw and corrected p-values, and even/odd-bin tuning correlations. The default permutation shifts are applied after speed filtering, on the retained sequence. The alternate Kruskal–Wallis path is defined but not used in the example cells.

## Decoding

`RF_decoder.ipynb` first compares logistic regression, Random Forest, SVM and MLP on contiguous event blocks. Its main Random Forest routine then uses chronological test fractions of 15%, 20% and 25%, each separated from training/validation by five events. These are overlapping holdouts of the same session, not independent replications.

Within training/validation, five contiguous blocks tune 16 Random Forest configurations: 200/400 trees, depth unrestricted/20, minimum leaf size 1/2 and `max_features` of `sqrt`/0.5. Each block is evaluated against the remaining blocks, which can contain both earlier and later events; there is no inner temporal gap. Neuron selection, target event length, scaling and optional PCA are fit on each training fold. Variable-duration events are truncated or zero-padded and flattened. The examples use the median training length.

The main routine reports accuracy, balanced accuracy, macro/weighted F1, per-class metrics, Wilson accuracy intervals and bootstrap macro-F1 intervals. Controls include shuffled training labels and rotated labels. Its test p-value permutes test labels against fixed model predictions; it does not refit models under each permuted training set. Classes absent from the training portion are excluded from event-level evaluation.

Behavioral comparisons use radial distance and sine/cosine of XY angle, optionally with per-sample differences. `center=None` uses the input coordinate origin. Those features are a comparison based on trajectories, not a statistical adjustment removing behavioral information from neural decoding.

`pre_event_decoding.ipynb` has two analyses:

- **Entry-aligned event classification:** 2 s windows beginning at −6, −5, −4, −3, −2, −1 and 0 s relative to arm entry; the final two overlap or follow entry. It uses the event-level RF routine with a 20% holdout and five-event gap.
- **Previous-exit bin classification:** 2 s windows from −4 to +6 s relative to the preceding visit's exit, predicting the next arm. Neural bins are individual examples, with all bins from one event kept together in each split. The example uses four inner blocks, a one-event gap, a 20% test holdout and 3,000 test-label permutations. Behavioral features are interpolated to the same number of time bins as the neural window.

Window extractors use floor-based session-bin boundaries; returned relative times are nominal bin centers, with sub-bin alignment error possible. Incomplete windows can be dropped or padded. Transition construction pairs consecutive visits, including visits that straddle trial boundaries, and does not constrain windows to end before the next visit.

## Tensor decompositions

`TCA_decomp.ipynb` applies TensorLy's unconstrained CP/PARAFAC at **rank 3**, random initialization and tolerance `1e-6`. It reorders inputs to `(events, neurons, position/time)` and plots one factor for each mode. Random initialization is not seeded in the notebook. Event sorting is for display and preserves the original factors.

`SliceTCA` represents each component with a vector along one mode and a matrix spanning the other two. It is not a smoothness constraint on positions. See the [library's definition and examples](https://github.com/arthur-pe/slicetca).

`slicetca_decomp.ipynb` uses `(events, neurons, position/time)`, min–max normalizes each event, then divides by the tensor's global standard deviation. This removes between-event amplitude scale. Examples request **(3, 3, 3)** components, learning rate `5e-3`, up to 10,000 iterations, `min_std=1e-5` and seed 0 on CPU. Both positive and unconstrained examples exist. Invariance optimization is applied only when `positive=True`, unlike the library's unconstrained example. Returned projections use the same component representation as the plotted model.

Neither notebook performs rank search, held-out reconstruction assessment or repeated-fit stability analysis. Factor scales/signs and 3D distances should not be treated as unique biological quantities. SliceTCA's generic sorted/unsorted image filenames are overwritten by later calls; save individual fits separately when retaining a full analysis record.

## Interpretation limits

The following limits affect interpretation:

- Symmetric Gaussian smoothing uses activity on both sides of a time point. A neural window ending before entry can therefore contain contributions from later spikes. Windows from neighboring events can also overlap.
- Session-wide neuron filtering, duration-outlier removal and behavioral median-length normalization happen before splitting. Blocked evaluation reduces some temporal mixing but does not make the entire pipeline independent of test data.
- The plotted RF holdout is selected by its test macro-F1. Learning curves and importance comparisons reuse tuned parameters. These are exploratory displays, not additional unbiased performance estimates.
- Bin-level confidence intervals, bootstrap resampling and permutations do not account for dependence among bins of the same event. Even/odd position reliability also compares temporally close, smoothed observations. Their uncertainty should not be read as independent-trial uncertainty.
- Clipped tracking positions, zero padding, incomplete trial annotations and the heuristic for mixed 0/1-based arm IDs need checking against real recording metadata. No cross-animal/session generalization or separation of planning from position, movement and recent history has been established.

The synthetic example and regression tests check specific software behavior. They do not validate these assumptions or reproduce the original recordings.
