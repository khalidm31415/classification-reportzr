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
report = svc_reporter.run_experiment(samples, labels, test_sizes=[0.1, 0.2], repetition=5)
print(report)
```

prints

```
   Test Size   C  gamma  kernel                     Train Accuracies  \
0        0.1  10  0.005    poly  [0.881, 0.896, 0.888, 0.881, 0.873]
1        0.1  10  0.005     rbf     [0.978, 0.978, 0.97, 0.97, 0.97]
2        0.1  10  0.005  linear  [0.978, 0.978, 0.978, 0.978, 0.978]
3        0.1  10  0.050    poly  [0.978, 0.978, 0.978, 0.978, 0.978]
4        0.1  10  0.050     rbf  [0.993, 0.985, 0.993, 0.985, 0.985]

   Max Train  Mean Train  Stdev Train                    Test Accuracies  \
0      0.896       0.884        0.008        [0.933, 0.8, 0.8, 1.0, 1.0]
1      0.978       0.973        0.004  [0.933, 0.867, 0.933, 0.8, 0.933]
2      0.978       0.978        0.000      [0.933, 1.0, 1.0, 0.933, 1.0]
3      0.978       0.978        0.000          [1.0, 1.0, 1.0, 1.0, 1.0]
4      0.993       0.988        0.004      [0.933, 1.0, 0.933, 1.0, 1.0]

   Max Test  Mean Test  Stdev Test  \
0     1.000      0.907       0.090
1     0.933      0.893       0.053
2     1.000      0.973       0.033
3     1.000      1.000       0.000
4     1.000      0.973       0.033

                                Experiment Times
0   [0.00086, 0.00076, 0.0007, 0.00071, 0.00069]
1  [0.00075, 0.00075, 0.00073, 0.00074, 0.00074]
2  [0.00048, 0.00046, 0.00046, 0.00045, 0.00046]
3  [0.00046, 0.00049, 0.00048, 0.00048, 0.00047]
4  [0.00061, 0.00058, 0.00057, 0.00059, 0.00059]
```
