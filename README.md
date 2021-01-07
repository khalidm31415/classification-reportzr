# Classification Reportzr
Automate machine learning classification task report for Pak Zuherman

## Install
```bash
pip install classification-reportzr
```

## Usage
### Setting-up the experiment
```python
from sklearn import datasets
from sklearn.svm import SVC

from classification_reportzr.reporterzr import Reporterzr

digits = datasets.load_digits()
samples, labels = digits.data[:-1], digits.target[:-1]

svc_kwargs = {'C':100.0, 'gamma':0.001}
svc_reporter = Reporterzr(EstimatorClass=SVC, estimator_kwargs=svc_kwargs, samples=samples, labels=labels, random_state=3)

# `test_sizes` defaults to [0.1, ..., 0.9]
svc_reporter.run_experiment(test_sizes=[0.1, 0.2])
```

### Get Accuracy Report
```python
print(svc_reporter.get_accuracy_report())
```
prints
```
   train_accuracy  test_accuracy  test_size
0             1.0       0.994444        0.1
1             1.0       0.988889        0.2
```

### Get Classification Report
```python
print(svc_reporter.get_classification_report(test_size=0.1, split='train'))
```
prints
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       160
           1       1.00      1.00      1.00       164
           2       1.00      1.00      1.00       159
           3       1.00      1.00      1.00       164
           4       1.00      1.00      1.00       163
           5       1.00      1.00      1.00       164
           6       1.00      1.00      1.00       163
           7       1.00      1.00      1.00       161
           8       1.00      1.00      1.00       156
           9       1.00      1.00      1.00       162

    accuracy                           1.00      1616
   macro avg       1.00      1.00      1.00      1616
weighted avg       1.00      1.00      1.00      1616
```

### Present All Classification Report
```python
svc_reporter.present_all_classification_report()
```
prints
```
Test size: 0.1
==================================================
Classification report on train data
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       160
           1       1.00      1.00      1.00       164
           2       1.00      1.00      1.00       159
           3       1.00      1.00      1.00       164
           4       1.00      1.00      1.00       163
           5       1.00      1.00      1.00       164
           6       1.00      1.00      1.00       163
           7       1.00      1.00      1.00       161
           8       1.00      1.00      1.00       156
           9       1.00      1.00      1.00       162

    accuracy                           1.00      1616
   macro avg       1.00      1.00      1.00      1616
weighted avg       1.00      1.00      1.00      1616

==================================================
Classification report on test data
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        18
           1       1.00      1.00      1.00        18
           2       1.00      1.00      1.00        18
           3       1.00      1.00      1.00        19
           4       1.00      1.00      1.00        18
           5       1.00      0.94      0.97        18
           6       1.00      1.00      1.00        18
           7       1.00      1.00      1.00        18
           8       1.00      1.00      1.00        17
           9       0.95      1.00      0.97        18

    accuracy                           0.99       180
   macro avg       0.99      0.99      0.99       180
weighted avg       0.99      0.99      0.99       180

================================================== 
 ================================================== 



Test size: 0.2
==================================================
Classification report on train data
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       142
           1       1.00      1.00      1.00       145
           2       1.00      1.00      1.00       142
           3       1.00      1.00      1.00       146
           4       1.00      1.00      1.00       145
           5       1.00      1.00      1.00       146
           6       1.00      1.00      1.00       145
           7       1.00      1.00      1.00       143
           8       1.00      1.00      1.00       138
           9       1.00      1.00      1.00       144

    accuracy                           1.00      1436
   macro avg       1.00      1.00      1.00      1436
weighted avg       1.00      1.00      1.00      1436

==================================================
Classification report on test data
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        36
           1       1.00      1.00      1.00        37
           2       1.00      1.00      1.00        35
           3       1.00      0.97      0.99        37
           4       1.00      1.00      1.00        36
           5       0.97      0.94      0.96        36
           6       1.00      1.00      1.00        36
           7       1.00      1.00      1.00        36
           8       1.00      1.00      1.00        35
           9       0.92      0.97      0.95        36

    accuracy                           0.99       360
   macro avg       0.99      0.99      0.99       360
weighted avg       0.99      0.99      0.99       360

================================================== 
 ================================================== 
```