# Classification Reportzr

Automate machine learning classification task report for Pak Zuherman

## Install

```bash
pip install -U classification-reportzr
```

## Test

```bash
pytest -v
```

## Usage

### Setting-up the experiment

```python
from sklearn import datasets
from sklearn.svm import SVC

from classification_reportzr.reporterzr import Reporterzr

iris = datasets.load_iris()
samples, labels = iris.data[:-1], iris.target[:-1]

svc_kwargs = {'C':100.0, 'gamma':0.001}
svc_reporter = Reporterzr(SVC, svc_kwargs)
```

### Run The Experiment

```python
# `test_sizes` defaults to [0.1, ..., 0.9]
# `rep` defaults to 10
svc_reporter.run_experiment(samples, labels, test_sizes=[0.1, 0.2, 0.3], rep=7)
```

### Get Accuracy Report

```python
print(svc_reporter.report())
```

prints

```
Split: 10.0% test - 90.0% train
           mean     stdev                                      accuracies
train  0.973429  0.006758  [0.978, 0.985, 0.963, 0.97, 0.978, 0.97, 0.97]
test   0.961714  0.033156     [0.933, 1.0, 1.0, 1.0, 0.933, 0.933, 0.933]

Split: 20.0% test - 80.0% train
           mean     stdev                                        accuracies
train  0.965143  0.010343  [0.958, 0.966, 0.966, 0.983, 0.958, 0.95, 0.975]
test   0.952429  0.039326         [1.0, 0.967, 1.0, 0.9, 0.9, 0.967, 0.933]

Split: 30.0% test - 70.0% train
           mean     stdev                                         accuracies
train  0.976571  0.009897    [0.971, 0.962, 0.99, 0.971, 0.99, 0.971, 0.981]
test   0.942857  0.026368  [0.933, 0.978, 0.911, 0.978, 0.911, 0.933, 0.956]
```
