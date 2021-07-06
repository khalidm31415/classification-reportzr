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

from reporterzr import Reporterzr

iris = datasets.load_iris()
samples, labels = iris.data[:-1], iris.target[:-1]

param_grid = {
    'C': [10,50,100],
    'gamma': [0.005,0.05,0.5],
    'kernel': ['poly', 'rbf', 'linear']
}
svc_reporter = Reporterzr(SVC, param_grid)
```

### Run The Experiment

```python
# `test_sizes` defaults to [0.1, ..., 0.9]
# `repetition` defaults to 10
report = svc_reporter.run_experiment(samples, labels, test_sizes=[0.1, 0.2], repetition=3)
print(report)
```

prints

```
   Test Size   C  gamma  kernel       Train Accuracies  Max Train  Mean Train  \
0        0.1  10  0.005    poly  [0.873, 0.896, 0.881]      0.896       0.883
1        0.1  10  0.005     rbf   [0.978, 0.97, 0.978]      0.978       0.975
2        0.1  10  0.005  linear  [0.985, 0.985, 0.978]      0.985       0.983
3        0.1  10  0.050    poly   [0.985, 0.97, 0.978]      0.985       0.978
4        0.1  10  0.050     rbf  [0.993, 0.993, 0.993]      0.993       0.993

   Stdev Train      Test Accuracies  Max Test  Mean Test  Stdev Test  \
0        0.010    [1.0, 0.8, 0.867]     1.000      0.889       0.083
1        0.004  [0.933, 0.933, 0.8]     0.933      0.889       0.063
2        0.003  [0.933, 1.0, 0.933]     1.000      0.955       0.032
3        0.006  [0.933, 0.933, 1.0]     1.000      0.955       0.032
4        0.000    [1.0, 0.933, 1.0]     1.000      0.978       0.032

        Experiment Times
0  [0.008, 0.001, 0.001]
1  [0.002, 0.001, 0.002]
2  [0.001, 0.001, 0.001]
3  [0.001, 0.001, 0.001]
4  [0.001, 0.001, 0.001]
```
