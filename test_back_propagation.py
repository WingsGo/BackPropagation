import unittest
import random
from back_propagation import NeuralNetwork


class MyTestCase(unittest.TestCase):
    EPSILON = 1.0e-04

    def test_choose_parameters(self):
        training_sets = [
            [[0, 0], [0]],
            [[0, 1], [1]],
            [[1, 0], [1]],
            [[1, 1], [0]]
        ]

        nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
        for i in range(1000000):
            training_inputs, training_outputs = random.choice(training_sets)
            nn.train(training_inputs, training_outputs)
        self.assertTrue(nn.calculate_total_error(training_sets) < self.EPSILON)

    def test_not_choose_parameters(self):
        nn = NeuralNetwork(2, 2, 2)
        for i in range(10000):
            nn.train([0.05, 0.1], [0.01, 0.09])
        self.assertTrue(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.09]]]) < self.EPSILON)
