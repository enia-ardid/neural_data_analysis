"""Exercise spike binning, event tensors and arm decoding using simulated data only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.notebook_code import load_definitions


def simulate_session(seed=7):
    """Generate 64 visits with deliberately arm-dependent Poisson spike counts.

    These inputs do not approximate a particular recording. Both the preferred
    arms and the trajectories are prescribed, so decoding success is expected.
    """
    rng = np.random.default_rng(seed)
    arms = np.tile(np.arange(1, 9), 8)
    durations = rng.choice([16, 20, 24], size=len(arms))
    starts = np.r_[10, 10 + np.cumsum(durations[:-1] + 10)]
    stops = starts + durations
    n_bins = int(stops[-1] + 10)
    rates = np.full((24, n_bins), 0.4)
    preferred_arms = np.tile(np.arange(1, 9), 3)
    for arm, start, stop in zip(arms, starts, stops):
        rates[preferred_arms == arm, start:stop] += 4.0
    counts = rng.poisson(rates)
    neuron, time_bin = np.nonzero(counts)
    multiplicity = counts[neuron, time_bin]
    clusters = np.repeat(neuron + 2, multiplicity)
    spike_times = np.repeat(time_bin * 2000, multiplicity)
    spike_times += rng.integers(0, 2000, size=len(spike_times))
    order = np.argsort(spike_times)

    tracking_ms = np.arange(int(np.ceil(n_bins * 100 * 39.0625 / 1000))) * 25.6
    speed = np.full(len(tracking_ms), 2.0)
    x, y, position = (np.zeros(len(tracking_ms)) for _ in range(3))
    trial_ids = np.zeros(len(tracking_ms), dtype=int)
    for i, (arm, start, stop) in enumerate(zip(arms, starts * 100, stops * 100)):
        mask = (tracking_ms >= start) & (tracking_ms < stop)
        distance = np.linspace(0, 139, mask.sum())
        angle = (arm - 1) * np.pi / 4
        x[mask], y[mask] = distance * np.cos(angle), distance * np.sin(angle)
        position[mask] = distance
        speed[mask] = 12
        trial_ids[mask] = i // 8

    return {
        "res": spike_times[order], "clu": clusters[order],
        "putative_type": pd.DataFrame({"type": ["noise", "multiunit"] + ["pp"] * 16 + ["p1"] * 8}),
        "event_intervals_ms": np.column_stack([starts, stops]) * 100,
        "event_arm_ids": arms, "speed": speed,
        "lwhl": pd.DataFrame({"x1": x, "y1": y, "linearized_pos_long": position, "trial_id": trial_ids}),
    }


def run_demo(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = simulate_session()
    tensor_code = load_definitions("construct_tensors.ipynb")
    decoder = load_definitions("RF_decoder.ipynb")

    activity = tensor_code.binning_time(
        session["res"], session["clu"], session["putative_type"],
        bin_size_ms=100, sampling_rate_neurons=20000, sigma=3, cell_types=["pp"])
    tensor, tensor_labels = tensor_code.build_tensor_time_binning(
        activity, session["event_intervals_ms"], 100, session["event_arm_ids"],
        speed=session["speed"], speed_threshold=5, min_spikes=100)
    events, labels, kept_indices = decoder.build_event_matrices(
        activity, session["event_intervals_ms"], 100, session["event_arm_ids"],
        speed=session["speed"], speed_threshold=5)
    np.testing.assert_array_equal(labels, session["event_arm_ids"][kept_indices])
    assert tensor.shape[2] == len(tensor_labels) == 64
    assert np.isfinite(tensor).all()

    # A single small grid point keeps this demonstration quick.
    results = decoder.rf_blockcv_grid_with_holdout_robust(
        events, labels, holdout_fracs=(0.20,), n_blocks_inner=3, gap=5,
        min_spikes=100, param_grid=[{"n_estimators": 32, "max_depth": 8}],
        n_sanity_shuffles=1, n_perm_test=49, random_state=7, verbose=False)
    result = results[0]
    confusion_figure, _, plotted_metrics, _ = decoder.plot_holdout_confusion_from_result(
        events, labels, result, target_len="median_pad", random_state=7)
    assert plotted_metrics["accuracy"] == result.test_metrics["accuracy"]
    confusion_matrix = np.asarray(confusion_figure.axes[0].images[0].get_array())
    plt.close(confusion_figure)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("From spikes to arm decoding", x=0.07, ha="left", y=0.96, fontsize=20)
    fig.text(0.07, 0.885, "SYNTHETIC EXAMPLE  ·  64 simulated visits  ·  No experimental data", fontsize=11)
    heat = axes[0].imshow(tensor[:, :, 0], aspect="auto", origin="lower", cmap="viridis")
    axes[0].set(title="One simulated arm visit", xlabel="Retained 100 ms bin", ylabel="Simulated neuron")
    fig.colorbar(heat, ax=axes[0], shrink=0.8, label="Smoothed spikes / bin")
    axes[1].imshow(confusion_matrix, vmin=0, vmax=1, cmap="Blues")
    axes[1].set(title=f"Chronological holdout ({result.test_ci['n_test_samples']} visits)",
                xlabel="Predicted arm", ylabel="Simulated arm",
                xticks=range(8), xticklabels=range(1, 9), yticks=range(8), yticklabels=range(1, 9))
    for row in range(8):
        for col in range(8):
            value = confusion_matrix[row, col]
            axes[1].text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=9,
                         color="white" if value > .5 else "#333333")
    fig.text(0.07, 0.06, "Arm-dependent firing is built into the simulation. This illustrates the workflow; it is not a biological result.", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.77, bottom=0.20, wspace=0.38)
    fig.savefig(output_dir / "synthetic_workflow.png", dpi=100)
    plt.close(fig)

    summary = {
        "data": "synthetic; arm-dependent firing prescribed by simulate_session", "seed": 7,
        "events": len(labels), "tensor_shape_neurons_bins_events": list(tensor.shape),
        "holdout_fraction": 0.20, "gap_events": 5, "trees": 32,
        "test_metrics": result.test_metrics, "test_intervals": result.test_ci,
        "interpretation": "Software demonstration only; no inference about experimental recordings.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Synthetic demonstration completed: {len(labels)} events, tensor {tensor.shape}.")
    print(f"Figure and metrics: {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    run_demo(parser.parse_args().output_dir)
