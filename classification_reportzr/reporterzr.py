from collections import OrderedDict
from typing import Iterable, List, Type
from typing_extensions import Literal

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_colwidth', 250)

class Reporterzr:
    def __init__(self, EstimatorClass: Type[ClassifierMixin], estimator_kwargs: dict):
        self.EstimatorClass = EstimatorClass
        self.estimator_kwargs = estimator_kwargs
        self.classification_reports = OrderedDict()
        self.reports: List[pd.DataFrame] 
        
    def run_experiment(self, samples: Iterable, labels: Iterable, test_sizes: List[float] = [round(i * 0.1, 1) for i in range(1,10)], rep: int = 10):
        assert len(samples) == len(labels), 'Samples and labels must have equal length'

        self.test_sizes = test_sizes
        self.reports: List[pd.DataFrame] = []

        for test_size in test_sizes:

            train_accuracies = []
            test_accuracies = []

            for _ in range(rep):
                X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = test_size, stratify = labels)

                estimator = self.EstimatorClass(**self.estimator_kwargs)
                estimator.fit(X_train, y_train)
        
                y_pred_train = estimator.predict(X_train)
                y_pred_test = estimator.predict(X_test)
        
                train_accuracy = round(accuracy_score(y_train, y_pred_train), 3)
                test_accuracy = round(accuracy_score(y_test, y_pred_test), 3)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)

            train_accuracies = np.array(train_accuracies)
            test_accuracies = np.array(test_accuracies)

            report = pd.DataFrame({
                'mean': [train_accuracies.mean(), test_accuracies.mean()],
                'stdev': [train_accuracies.std(), test_accuracies.std()],
                'accuracies': [train_accuracies, test_accuracies]
            }, index=['train', 'test'])

            self.reports.append(report)
            
    def report_all_classification(self):
        if not self.reports:
            raise Exception('Run experiment first with ".run_experiment()" method')
            
        for report, test_size in zip(self.reports, self.test_sizes):
            print(f'Split: {100 * test_size}% test - {100 * (1 - test_size)}% train')
            print(report, end='\n\n')
