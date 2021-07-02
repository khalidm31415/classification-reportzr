from collections import OrderedDict
from typing import Iterable, List, Type

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score

pd.set_option('display.max_colwidth', 250)

class Reporterzr:
    def __init__(self, EstimatorClass: Type[ClassifierMixin], param_grid: List[dict]):
        self.EstimatorClass = EstimatorClass
        self.param_grid = param_grid
        self.classification_reports = OrderedDict()
        self.reports: List[pd.DataFrame] 
        
    def run_experiment(self, samples: Iterable, labels: Iterable, test_sizes: List[float] = [round(i * 0.1, 1) for i in range(1,10)], repetition: int = 10):
        assert len(samples) == len(labels), 'Samples and labels must have equal length'
        assert all(type(test_size) == float for test_size in test_sizes), 'test_sizes should be a list of floats'
        assert type(repetition) == int, 'The number of repetition should be an integer'

        self.reports: List[pd.DataFrame] = []

        test_size_column: List[float] = []
        train_accuracies_column: List[np.ndarray] = []
        test_accuracies_column: List[np.ndarray] = []

        hyper_parameters_column = {}
        for key in self.param_grid.keys():
            hyper_parameters_column[key] = []

        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = test_size, stratify = labels)
            for estimator_kwargs in ParameterGrid(self.param_grid):
                train_accuracies = []
                test_accuracies = []
                for _ in range(repetition):
                    estimator = self.EstimatorClass(**estimator_kwargs)
                    estimator.fit(X_train, y_train)
            
                    y_pred_train = estimator.predict(X_train)
                    y_pred_test = estimator.predict(X_test)
            
                    train_accuracy = round(accuracy_score(y_train, y_pred_train), 3)
                    test_accuracy = round(accuracy_score(y_test, y_pred_test), 3)
                    train_accuracies.append(train_accuracy)
                    test_accuracies.append(test_accuracy)
                
                test_size_column.append(test_size)
                train_accuracies_column.append(np.array(train_accuracies))
                test_accuracies_column.append(np.array(test_accuracies))
                for key, value in estimator_kwargs.items():
                    hyper_parameters_column[key].append(value)

        report = pd.DataFrame({
            'Test Size': test_size_column,
            **hyper_parameters_column,
            'Train Accuracies': train_accuracies_column,
            'Max Train': [round(train_accuracies.max(), 3) for train_accuracies in train_accuracies_column],
            'Mean Train': [round(train_accuracies.mean(), 3) for train_accuracies in train_accuracies_column],
            'Stdev Train': [round(train_accuracies.std(), 3) for train_accuracies in train_accuracies_column],
            'Test Accuracies': test_accuracies_column,
            'Max Test': [round(test_accuracies.max(), 3) for test_accuracies in test_accuracies_column],
            'Mean Test': [round(test_accuracies.mean(), 3) for test_accuracies in test_accuracies_column],
            'Stdev Test': [round(test_accuracies.std(), 3) for test_accuracies in test_accuracies_column],
        })
        print(report)
