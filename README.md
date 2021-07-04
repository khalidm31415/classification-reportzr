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
    'kernel': ['poly', 'rbf', 'kernel']
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
    Test Size    C  gamma       Train Accuracies  Max Train  Mean Train  Stdev Train        Test Accuracies  Max Test  Mean Test  Stdev Test
0         0.1   10  0.005   [0.97, 0.948, 0.963]      0.970       0.960        0.009      [0.933, 1.0, 1.0]     1.000      0.978       0.032
1         0.1   10  0.050  [0.993, 0.985, 0.993]      0.993       0.990        0.004      [1.0, 1.0, 0.933]     1.000      0.978       0.032
2         0.1   10  0.500  [0.978, 0.978, 0.978]      0.978       0.978        0.000        [1.0, 1.0, 1.0]     1.000      1.000       0.000
3         0.1   50  0.005  [0.993, 0.993, 0.978]      0.993       0.988        0.007        [1.0, 1.0, 1.0]     1.000      1.000       0.000
4         0.1   50  0.050   [0.97, 0.978, 0.993]      0.993       0.980        0.010      [1.0, 1.0, 0.933]     1.000      0.978       0.032
5         0.1   50  0.500  [0.978, 0.978, 0.993]      0.993       0.983        0.007      [1.0, 1.0, 0.933]     1.000      0.978       0.032
6         0.1  100  0.005  [0.993, 0.985, 0.993]      0.993       0.990        0.004        [1.0, 1.0, 1.0]     1.000      1.000       0.000
7         0.1  100  0.050   [0.97, 0.985, 0.993]      0.993       0.983        0.010    [1.0, 0.867, 0.933]     1.000      0.933       0.054
8         0.1  100  0.500    [1.0, 0.993, 0.985]      1.000       0.993        0.006      [0.8, 0.933, 1.0]     1.000      0.911       0.083
9         0.2   10  0.005  [0.975, 0.958, 0.975]      0.975       0.969        0.008    [0.9, 0.933, 0.967]     0.967      0.933       0.027
10        0.2   10  0.050  [0.992, 0.983, 0.992]      0.992       0.989        0.004      [1.0, 1.0, 0.967]     1.000      0.989       0.016
11        0.2   10  0.500  [0.983, 0.983, 0.983]      0.983       0.983        0.000    [0.9, 0.967, 0.933]     0.967      0.933       0.027
12        0.2   50  0.005  [0.992, 0.992, 0.992]      0.992       0.992        0.000    [0.967, 0.967, 1.0]     1.000      0.978       0.016
13        0.2   50  0.050    [1.0, 0.992, 0.975]      1.000       0.989        0.010    [0.933, 0.933, 1.0]     1.000      0.955       0.032
14        0.2   50  0.500  [0.983, 0.983, 0.975]      0.983       0.980        0.004  [0.867, 0.933, 0.967]     0.967      0.922       0.042
15        0.2  100  0.005  [0.983, 0.983, 0.992]      0.992       0.986        0.004        [1.0, 1.0, 1.0]     1.000      1.000       0.000
16        0.2  100  0.050  [0.966, 0.975, 0.983]      0.983       0.975        0.007  [0.967, 0.967, 0.967]     0.967      0.967       0.000
17        0.2  100  0.500  [0.992, 0.992, 0.983]      0.992       0.989        0.004  [0.933, 0.933, 0.967]     0.967      0.944       0.016
```
