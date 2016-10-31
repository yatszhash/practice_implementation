
import unittest

from unittest import TestCase

from nlp_ml.seq_labeling.FBAGraph import FBAGraph


class TestNode(TestCase):

    def setUp(self):
        super().setUp()
        self.graph = FBAGraph()
        self.labels = ["c1", "c2"]
        self.X = ["x1", "x2", "x3", "x4"]
        self.graph.initialize(self.X, self.labels)

        #initialize test data
        self.graph.search_edge("BOS", "c1", 1).path_cost = 1.0
        self.graph.search_edge("BOS", "c2", 1).path_cost = 1.0

        self.graph.search_edge("c1", "c1", 2).path_cost = 0.2
        self.graph.search_edge("c2", "c1", 2).path_cost = 0.3
        self.graph.search_edge("c1", "c2", 2).path_cost = 0.1
        self.graph.search_edge("c2", "c2", 2).path_cost = 0.1
        
        self.graph.search_edge("c1", "c1", 3 ).path_cost = 0.2
        self.graph.search_edge("c2", "c1", 3 ).path_cost = 0.2
        self.graph.search_edge("c1", "c2", 3 ).path_cost = 0.1
        self.graph.search_edge("c2", "c2", 3 ).path_cost = 0.1
        
        self.graph.search_edge("c1", "c1", 4).path_cost = 0.3
        self.graph.search_edge("c2", "c1", 4).path_cost = 0.1
        self.graph.search_edge("c1", "c2", 4).path_cost = 0.2
        self.graph.search_edge("c2", "c2", 4).path_cost = 0.1

        self.graph.search_edge("c1", "EOS", 5).path_cost = 1.0
        self.graph.search_edge("c2", "EOS", 5).path_cost = 1.0

    def test_get_alpha_cost(self):
        #layer0
        self.assertEqual(self.graph.search_node("BOS", 0).get_alpha_cost(), 1)

        #layer1
        self.assertEqual(self.graph.search_node("c1", 1).get_alpha_cost(), 1)
        self.assertEqual(self.graph.search_node("c2", 1).get_alpha_cost(), 1)

        #layer2
        self.assertEqual(self.graph.search_node("c1", 2).get_alpha_cost(), 0.5)
        self.assertEqual(self.graph.search_node("c2", 2).get_alpha_cost(), 0.2)

        # layer3
        self.assertEqual(self.graph.search_node("c1", 3).get_alpha_cost(), 0.14)
        self.assertEqual(self.graph.search_node("c2", 3).get_alpha_cost(), 0.07)

        # layer4
        self.assertEqual(self.graph.search_node("c1", 4).get_alpha_cost(), 0.049)
        self.assertEqual(self.graph.search_node("c2", 4).get_alpha_cost(), 0.035)

        # layer5
        self.assertEqual(self.graph.search_node("EOS", 5).get_alpha_cost(),
                         0.084)

    def test_get_alpha_only_last_cost(self):
        self.assertEqual(
            self.graph.search_node("EOS", len(self.X) + 1).get_alpha_cost(),
            0.084)

    def test_get_beta_cost(self):
        #layer0
        self.assertAlmostEqual(
            self.graph.search_node("BOS", 0).get_beta_cost(), 0.084)

        #layer1
        self.assertAlmostEqual(
            self.graph.search_node("c1", 1).get_beta_cost(), 0.036)
        self.assertAlmostEqual(
            self.graph.search_node("c2", 1).get_beta_cost(), 0.048)

        #layer2
        self.assertAlmostEqual(
            self.graph.search_node("c1", 2).get_beta_cost(), 0.12)
        self.assertAlmostEqual(
            self.graph.search_node("c2", 2).get_beta_cost(), 0.12)

        # layer3
        self.assertAlmostEqual(
            self.graph.search_node("c1", 3).get_beta_cost(), 0.5)
        self.assertAlmostEqual(
            self.graph.search_node("c2", 3).get_beta_cost(), 0.2)

        # layer4
        self.assertEqual(self.graph.search_node("c1", 4).get_beta_cost(), 1)
        self.assertEqual(self.graph.search_node("c2", 4).get_beta_cost(), 1)

        # layer5
        self.assertEqual(self.graph.search_node("EOS", 5).get_beta_cost(),
                         1)

    def test_get_beta_cost_only_first_cost(self):
        self.assertAlmostEqual(
            self.graph.search_node("BOS", 0 ).get_beta_cost(),
            0.084)

    def test_get_z_coef(self):
        actual = self.graph.get_z_coef()
        self.assertAlmostEqual(actual, 0.084)

    def test_get_marginal_probability(self):
        self.assertAlmostEqual(
            self.graph.get_marginal_probability("c1", "c1", 3), 0.595, 3)
        self.assertAlmostEqual(
            self.graph.get_marginal_probability("c2", "c1", 3), 0.238, 3)
        self.assertAlmostEqual(
            self.graph.get_marginal_probability("c1", "c2", 3), 0.119, 3)
        self.assertAlmostEqual(
            self.graph.get_marginal_probability("c2", "c2", 3), 0.048, 3)


if __name__ == "__main__":
    unittest.main()