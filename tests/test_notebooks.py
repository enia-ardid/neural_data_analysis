"""Regression tests against the functions defined in the analysis notebooks."""

import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.notebook_code import ROOT, load_definitions


def cell_source(notebook, marker):
    cells = json.loads((ROOT / notebook).read_text())["cells"]
    return next("".join(c["source"]) for c in cells
                if c["cell_type"] == "code" and marker in "".join(c["source"]))


class NotebookRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensors = load_definitions("construct_tensors.ipynb")
        cls.pre = load_definitions("pre_event_decoding.ipynb")
        cls.rf = load_definitions("RF_decoder.ipynb")
        cls.tca = load_definitions("TCA_decomp.ipynb")

    def tearDown(self):
        plt.close("all")

    def test_small_spike_batch_has_no_leading_empty_bin(self):
        types = pd.DataFrame({"type": ["noise", "multiunit", "pp"]})
        result = self.tensors.binning_n_spike(
            np.array([10, 20]), np.array([2, 2]), types, 200, ["pp"])
        np.testing.assert_array_equal(result, [[2]])

    def test_per_arm_curve_averages_events_preserving_time_axis(self):
        tensor = np.array([[[1., 3.], [5., 7.], [9., 11.]]])
        with patch.object(plt, "show"):
            self.tensors.plot_avg_spikecount_by_arm(tensor, [8, 8], neuron_indices=[0])
        np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), [2, 6, 10])

    def test_tensor_diagnostics_use_arm_labels_and_retained_neuron(self):
        activity = np.array([[0., 0., 0., 0.], [2., 4., 6., 8.]])
        with patch.object(plt, "show"):
            self.tensors.build_tensor_time_binning(
                activity, [(0, 100), (100, 200), (200, 300), (300, 400)],
                100, [8, 2, 8, 2], min_spikes=1, verify=True,
                verify_opts={"neuron_idx": 0})
        first, second = [plt.figure(i) for i in plt.get_fignums()[:2]]
        np.testing.assert_array_equal(first.axes[0].collections[0].get_offsets()[:, 0], [0, 1, 2, 3])
        np.testing.assert_allclose(first.axes[0].collections[0].get_offsets()[:, 1], [2, 4, 6, 8])
        np.testing.assert_allclose([p.get_height() for p in second.axes[0].patches], [6, 4])

    def test_dropped_event_preserves_labels_and_original_indices(self):
        matrices, labels, indices = self.rf.build_event_matrices(
            np.arange(6.)[None, :], [(0, 200), (200, 400), (400, 600)],
            100, [2, 4, 8], sampling_rate_wheel=10,
            speed=np.array([10, 10, 0, 0, 10, 10]), speed_threshold=5)
        np.testing.assert_array_equal(labels, [2, 8])
        np.testing.assert_array_equal(indices, [0, 2])
        np.testing.assert_array_equal(matrices[1], [[4, 5]])

    def test_outside_event_is_dropped_before_speed_partitioning(self):
        matrices, labels, indices = self.rf.build_event_matrices(
            np.ones((1, 2)), [(500, 700)], 100, [2], sampling_rate_wheel=10,
            speed=np.ones(10) * 10, speed_threshold=5)
        self.assertEqual(len(matrices), 0)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(indices), 0)

    def test_window_boundaries_preserve_time_and_never_wrap_negative_slices(self):
        activity = np.array([[10., 20., 30., 40.]])
        for name in ("extract_window_tensor_binned", "extract_window_relative_to_event_end"):
            extractor = getattr(self.pre, name)
            for anchor, expected in ((-300, [0, 0, 0]), (0, [0, 10, 20]), (300, [30, 40, 0])):
                with self.subTest(extractor=name, anchor=anchor):
                    tensor, kept, times = extractor(
                        activity, [(anchor, anchor)], window_start_ms=-100,
                        window_end_ms=200, bin_size_ms=100, on_short="pad")
                    np.testing.assert_array_equal(tensor[0, :, 0], expected)
                    np.testing.assert_array_equal(times, [-50, 50, 150])
                    self.assertTrue(kept[0])

    def test_outside_window_is_dropped(self):
        for name in ("extract_window_tensor_binned", "extract_window_relative_to_event_end"):
            tensor, kept, _ = getattr(self.pre, name)(
                np.arange(10.)[None, :], [(-300, -300)],
                window_start_ms=-100, window_end_ms=200, on_short="drop")
            self.assertEqual(tensor.shape, (1, 3, 0))
            self.assertFalse(kept[0])

    def test_prewindow_missing_tail_is_padded_at_the_end(self):
        tensor, _, _ = self.pre.extract_prewindow_tensor_binned(
            np.array([[10., 20., 30., 40.]]), [(600, 700)],
            start_offset_ms=300, end_offset_ms=0, on_short="pad")
        np.testing.assert_array_equal(tensor[0, :, 0], [40, 0, 0])

    def test_binwise_modalities_have_same_number_of_time_bins(self):
        intervals = [(200 + i * 500, 400 + i * 500) for i in range(6)]
        trajectory = pd.DataFrame({"x1": np.linspace(0, 10, 150), "y1": np.ones(150)})
        widths = []

        def decoder(matrices, labels, **kwargs):
            widths.append([m.shape[1] for m in matrices])
            return SimpleNamespace(test_metrics={}, test_ci={})

        with patch.object(self.pre, "_rf_binwise_grouped_by_event", side_effect=decoder):
            result = self.pre.run_sliding_window_decoding_bins(
                np.ones((2, 40)), np.ones((3, 40)), trajectory, intervals,
                [2, 4, 2, 4, 2, 4], sliding_windows_ms=[(0, 200)],
                min_events_total=2, min_per_class=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(widths, [[2] * 6] * 3)

    def test_fallback_converter_uses_neuron_time_event_order(self):
        tensor = np.arange(24).reshape(2, 3, 4)
        result = self.pre._tensor_to_event_mats_fallback(tensor)
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(result[1], tensor[:, :, 1])

    def test_tca_sorting_preserves_original_factors(self):
        factors = [np.array([[10.], [20.], [30.]]), np.ones((2, 1)), np.ones((4, 1))]
        original = [f.copy() for f in factors]
        sorted_factors, labels = self.tca.sort_events_and_tcafactors(factors, np.array([8, 2, 4]))
        np.testing.assert_array_equal(sorted_factors[0][:, 0], [20, 30, 10])
        np.testing.assert_array_equal(labels, [2, 4, 8])
        for before, after in zip(original, factors):
            np.testing.assert_array_equal(before, after)

    def test_tca_sorting_accepts_list_labels(self):
        factors = [np.array([[10.], [20.]]), np.ones((2, 1)), np.ones((4, 1))]
        _, labels = self.tca.sort_events_and_tcafactors(factors, [8, 2])
        np.testing.assert_array_equal(labels, [2, 8])

    def test_position_overlay_uses_recorded_speed_threshold(self):
        source = cell_source("position_modulation.ipynb", "best_neuron = next")
        result = {"order_by_p": [0], "p_raw": np.array([.01]), "params": {"speed_threshold": 5}}
        with patch.object(plt, "show"), patch("builtins.print"):
            plotter = unittest.mock.Mock()
            exec(source, {"np": np, "results_no_speed": result, "results_speed": result,
                          "plot_compare_tuning_for_neuron": plotter})
        self.assertIn("5", plotter.call_args.kwargs["label_b"])

    def test_holdout_plot_uses_same_event_representation_as_fitted_result(self):
        source = cell_source("RF_decoder.ipynb", "best_i_rew, results_b_rew =")
        time_events, position_events = [np.ones((2, 3))], [np.ones((2, 140))]
        result = SimpleNamespace(test_metrics={"macro_f1": .5})
        confusion = unittest.mock.Mock(return_value=(None, None, None, None))
        importance = unittest.mock.Mock(return_value=(None, None))
        with patch("builtins.print"):
            exec(source, {"pick_best_result": lambda *a, **k: (0, result),
                          "results_pp_rewarded": [result], "event_matrices_rewarded": time_events,
                          "arm_ids_rewarded": [2], "events_rewarded": position_events,
                          "event_arm_ids_rewarded": [2],
                          "plot_holdout_confusion_from_result": confusion,
                          "plot_top_neuron_importances_from_result": importance})
        self.assertIs(confusion.call_args.args[0], time_events)
        self.assertIs(importance.call_args.args[0], time_events)

    def test_slicetca_pfc_result_does_not_overwrite_hippocampus(self):
        source = cell_source("slicetca_decomp.ipynb", "tensor_nte_100pp_rewarded =")
        namespace = {"np": np, "tensorTE_filt_100pp_rewarded": np.zeros((2, 3, 4)),
                     "event_arm_ids_rewarded": np.array([2, 4, 2, 4]),
                     "components100hp_rewarded": "hippocampus",
                     "apply_slicetca": lambda *a, **k: ("pfc", None, None)}
        exec(source, namespace)
        self.assertEqual(namespace["components100hp_rewarded"], "hippocampus")
        self.assertEqual(namespace["components100pp_rewarded"], "pfc")

    def test_speed_file_preserves_decimal_values(self):
        loader = load_definitions("import_preprocess.ipynb")
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            base = Path(directory) / "toy-date"
            base.mkdir()
            contents = {
                "toy-date_run.res": "10\n20\n", "toy-date_run.clu": "3\n2\n2\n",
                "toy-date.des": "pp\n", "toy-date_run.whl": "0 0\n1 1\n2 2\n",
                "toy_date_run.speed": "1.25\n2.50\n3.75\n",
                "toy-date_run.reward_arms": "2\n", "toy-date_run.all_arms": "2\n",
                "toy_date_run.trials": "0 0 1 1 2 2\n",
                "toy_date_run.lwhl": "x1 y1 linearized_pos_long trial_id\n0 0 0 0\n1 1 1 0\n2 2 2 0\n",
                "toy_date_run.lwhl_raw": "x1 y1\n0 0\n1 1\n2 2\n",
                "toy_date_run.lwhl_speed": "speed\n1.25\n2.50\n3.75\n",
            }
            for name, content in contents.items():
                (base / name).write_text(content)
            with patch("builtins.print"):
                loaded = loader.load_data(directory, "toy", "date", ["run"])
        np.testing.assert_allclose(loaded[4], [1.25, 2.50, 3.75])

    def test_tca_time_cell_can_be_rerun_without_changing_input_axes(self):
        source = cell_source("TCA_decomp.ipynb", "factors_TE_filt_100pp_rewarded =")
        original = np.ones((2, 5, 3))
        seen_shapes = []
        namespace = {"tensorTE_filt_100pp_rewarded": original,
                     "event_arm_ids_rewarded": [2, 4, 8],
                     "apply_tca": lambda tensor, **k: seen_shapes.append(tensor.shape),
                     "plot_tca_components": lambda *a, **k: None}
        exec(source, namespace)
        exec(source, namespace)
        self.assertEqual(seen_shapes, [(2, 3, 5), (2, 3, 5)])
        self.assertIs(namespace["tensorTE_filt_100pp_rewarded"], original)

    def test_slicetca_returns_the_components_of_the_plotted_model(self):
        module = load_definitions("slicetca_decomp.ipynb")
        initial = [[np.zeros((1, 3)), np.zeros((1, 2, 4))]]
        transformed = [[np.ones((1, 3)), np.ones((1, 2, 4))]]
        model = SimpleNamespace(get_components=lambda **k: transformed)
        with patch.object(module.slicetca, "decompose", return_value=(initial, model)), \
             patch.object(module.slicetca, "invariance", return_value=model), \
             patch.object(module.slicetca, "plot"), patch.object(plt, "savefig"), \
             patch.object(plt, "show"), patch("builtins.print"):
            components, plotted_model, _ = module.apply_slicetca(
                np.arange(24.).reshape(3, 2, 4), np.array([2, 4, 8]))
        self.assertIs(plotted_model, model)
        np.testing.assert_array_equal(components[0][0], transformed[0][0])


if __name__ == "__main__":
    unittest.main()
