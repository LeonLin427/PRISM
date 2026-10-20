"""Regression tests for constructor seeding, training and group isolation."""
import unittest
import numpy as np
import torch
from gnn_model import MLP, RGCN
from train_gnn import seeded_model, train_gnn, iterative_strat


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_constructor_independent_of_prior_rng(self):
        for factory, args in [(MLP, (8, 6)), (RGCN, (8, 4, 6, 6))]:
            a = seeded_model(factory, 1000, *args)
            torch.manual_seed(923)
            torch.rand(1234)
            factory(*args)
            b = seeded_model(factory, 1000, *args)
            for x, y in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.equal(x, y))
            c = seeded_model(factory, 1001, *args)
            self.assertTrue(any(not torch.equal(x, y)
                                for x, y in zip(a.parameters(), c.parameters())))

    def test_training_repeats_after_intervening_model(self):
        x = torch.arange(80, dtype=torch.float32).reshape(10, 8) / 80
        y = np.array([[i % 2, (i+1) % 2]*3 for i in range(10)], dtype=float)
        for factory, args, graph in [(MLP, (8, 6), False), (RGCN, (8, 4, 6, 6), True)]:
            adj = [torch.eye(14) for _ in range(6)]
            def fit(seed):
                model = seeded_model(factory, seed, *args)
                train_gnn(model, x, adj, np.arange(10), y, 4, .001, seed, graph)
                model.eval()
                return (model(x, adj, torch.arange(10)) if graph else model(x)).detach()
            first = fit(1000)
            fit(1002)
            self.assertTrue(torch.equal(first, fit(1000)))

    def test_group_split_independent_of_torch_rng(self):
        labels = np.array([[i % 2, (i//2) % 2]*3 for i in range(20)])
        a = iterative_strat(labels, 5, seed=0)
        torch.manual_seed(700)
        torch.rand(100)
        np.testing.assert_array_equal(a, iterative_strat(labels, 5, seed=0))
        self.assertEqual(set(a), set(range(5)))


if __name__ == '__main__':
    unittest.main()
