from collections import OrderedDict
from typing import Iterable, List, Type
from typing_extensions import Literal

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class Reporterzr:
    def __init__(self, EstimatorClass: Type[ClassifierMixin], estimator_kwargs: dict, samples: Iterable, labels: Iterable, random_state: int = 0):
        self.EstimatorClass = EstimatorClass
        self.estimator_kwargs = estimator_kwargs
        
        assert len(samples) == len(labels), 'Samples and labels must have equal length'
        self.samples = samples
        self.labels = labels
        
        self.random_state = random_state
        
        self.classification_reports = OrderedDict()
        self.accuracy_report: pd.DataFrame
        
    def run_experiment(self, test_sizes: List[float] = [round(i * 0.1, 1) for i in range(1,10)]):
        self.test_sizes = test_sizes
        train_accuracies = []
        test_accuracies = []
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(self.samples, self.labels, test_size = test_size, stratify = self.labels, random_state = self.random_state)

            estimator = self.EstimatorClass(**self.estimator_kwargs)
            estimator.fit(X_train, y_train)
    
            y_pred_test = estimator.predict(X_test)
            y_pred_train = estimator.predict(X_train)
    
            classification_report_test = classification_report(y_test, y_pred_test, output_dict=False)
            classification_report_train = classification_report(y_train, y_pred_train, output_dict=False)
            self.classification_reports[test_size] = {'on_train_data':classification_report_train, 'on_test_data':classification_report_test}
    
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_accuracy = accuracy_score(y_train, y_pred_train)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        self.accuracy_report = pd.DataFrame({'train_accuracy': train_accuracies, 'test_accuracy': test_accuracies, 'test_size': test_sizes})
            
    def get_classification_report(self, test_size: float, split: Literal['train', 'test']):
        if not self.classification_reports:
            raise Exception('Run experiment first with ".run_experiment()" method')

        if test_size not in self.test_sizes:
            raise ValueError(f'The specified test_size is invalid. The experiment was run with these test_sizes: {self.test_sizes}')

        if split not in ('train', 'test'):
            raise ValueError('Invalid split selection, expected "train" or "test"')

        return self.classification_reports[test_size][f'on_{split}_data']
            
    def present_all_classification_report(self):
        if not self.classification_reports:
            raise Exception('Experiment have to bu run first with ".run_experiment()" method')
            
        for test_size, classification_report in self.classification_reports.items():
            print(f'Test size: {test_size}')
            print('=' * 50)
            print('Classification report on train data')
            print(classification_report['on_train_data'])
            print('=' * 50)
            print('Classification report on test data')
            print(classification_report['on_test_data'])
            print('=' * 50, '\n', '=' * 50, '\n\n\n')
            
    def get_accuracy_report(self):
        if not self.classification_reports:
            raise Exception('Run experiment first with ".run_experiment()" method')
            
        return self.accuracy_report
