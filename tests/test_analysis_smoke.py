"""Small synthetic executions; these do not validate experimental conclusions."""

import ast
import base64
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from scripts.notebook_code import ROOT, load_definitions


class AnalysisSmokeTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_notebooks_are_valid_and_store_only_static_figures(self):
        for path in ROOT.glob("*.ipynb"):
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        ast.parse(cell.source)
                        self.assertIsNone(cell.execution_count)
                        for output in cell.outputs:
                            self.assertEqual(output.output_type, "display_data")
                            self.assertEqual(set(output.data), {"image/png"})
                            self.assertEqual(output.metadata, {})
                            image = base64.b64decode("".join(output.data["image/png"]).replace("\n", ""), validate=True)
                            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_position_tensor_uses_tracking_sample_after_bin_center(self):
        module = load_definitions("construct_tensors.ipynb")
        tracking = pd.DataFrame({"linearized_pos_long": [0, 10, 20, 30],
                                 "x1": [0, 1, 2, 3], "y1": [0, 1, 2, 3]})
        with patch.object(plt, "show"):
            tensor, labels, stats = module.build_event_position_tensor(
                np.array([[2., 4., 6.]]), tracking, [(0, 300)], [2],
                sampling_rate_wheel=10)
        self.assertEqual(tensor.shape, (1, 1, 140))
        np.testing.assert_array_equal(tensor[0, 0, [10, 20, 30]], [2, 4, 6])
        self.assertEqual(stats["counts_per_pos"].sum(), 3)
        np.testing.assert_array_equal(labels, [2])

    def test_position_statistics_return_bounded_corrected_pvalues(self):
        module = load_definitions("position_modulation.ipynb")
        activity = np.random.default_rng(3).poisson(2, size=(2, 200)).astype(float)
        tracking = pd.DataFrame({"linearized_pos_long": np.repeat([10., 100., 10., 100.], 50)})
        result = module.identify_position_modulated_neurons(
            activity, 100, tracking, sampling_rate_wheel=10, n_slices=2,
            min_spikes=0, n_perm=19, top_k_plots=0, verbose=False)
        self.assertEqual(result["bins_per_slice"].sum(), 200)
        self.assertTrue(np.all(result["bins_per_slice"] >= 10))
        self.assertTrue(np.isfinite(result["F_obs"]).all())
        self.assertTrue(np.all((result["p_corrected"] >= 0) & (result["p_corrected"] <= 1)))
        self.assertEqual(result["tuning_mean_hz"].shape, (2, 2))

    def test_cp_reconstructs_a_rank_one_tensor(self):
        module = load_definitions("TCA_decomp.ipynb")
        tensor = np.einsum("i,j,k->ijk", [1., 2., 3.], [2., 4., 6., 8.], [1., 3., 2., 4., 5.])
        np.random.seed(7)
        factors = module.apply_tca(tensor, rank=1)
        reconstructed = np.einsum("ir,jr,kr->ijk", *factors)
        np.testing.assert_allclose(reconstructed, tensor.transpose(1, 0, 2), rtol=1e-5)

    def test_binwise_decoder_holds_out_whole_events(self):
        module = load_definitions("pre_event_decoding.ipynb")
        labels = np.tile([2, 4], 12)
        events = [np.array([[float(y), float(y) + .1], [1., 2.]]) for y in labels]
        result = module._rf_binwise_grouped_by_event(
            events, labels, holdout_frac=.25, gap_events=2, n_blocks_inner=2,
            param_grid=[{"n_estimators": 8, "max_depth": 3}], n_perm_test=9)
        self.assertEqual(result.counts, {"n_events_trval": 16, "n_events_test": 6,
                                         "n_bins_trval": 32, "n_bins_test": 12})
        self.assertTrue(np.isfinite(list(result.test_metrics.values())).all())

    def test_slicetca_fit_and_event_projection(self):
        module = load_definitions("slicetca_decomp.ipynb")
        data = np.random.default_rng(7).uniform(size=(6, 4, 8))
        # Exercise the real optimizer and invariance code with a small iteration
        # budget. This checks integration, not fit quality or convergence.
        decompose, invariance = module.slicetca.decompose, module.slicetca.invariance

        def short_fit(*args, **kwargs):
            kwargs.update(max_iter=5, progress_bar=False)
            return decompose(*args, **kwargs)

        def short_invariance(model):
            return invariance(model, max_iter=5, progress_bar=False)

        with patch.object(module.slicetca, "decompose", side_effect=short_fit), \
             patch.object(module.slicetca, "invariance", side_effect=short_invariance), \
             patch.object(plt, "show"), patch.object(plt, "savefig"):
            components, model, normalized = module.apply_slicetca(
                data, np.array([2, 4, 8, 2, 4, 8]), rank=(1, 1, 1), positive=True)
        self.assertEqual(normalized.shape, data.shape)
        self.assertEqual(module.get_event_factors_slicetca(components).shape, (6, 1))
        self.assertTrue(np.isfinite(model.construct().detach().numpy()).all())


if __name__ == "__main__":
    unittest.main()
